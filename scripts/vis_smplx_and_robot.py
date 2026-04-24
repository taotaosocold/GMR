"""
在 MuJoCo Viewer 中同時可視化：
  - SMPL-X 人體骨骼（全局位置 + 全局朝向坐標軸）
  - 機器人 XML 默認姿態（骨骼 + 坐標軸）

兩個骨骼並列顯示：機器人在原點，人體偏移到 +y 方向 1.5m 處。

座標軸箭頭顏色：
  紅 = X 軸方向
  綠 = Y 軸方向
  藍 = Z 軸方向

用法：
  python scripts/vis_smplx_and_robot.py \\
      --smplx_file /path/to/motion.npz \\
      --robot casbot_skeleton

交互操作（鍵盤）：
  ← / →   上一幀 / 下一幀
  Space    播放 / 暫停
  Home     跳到第 0 幀
  End      跳到最後一幀
  Q / Esc  退出
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer

GMR_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT

SMPLX_FOLDER = GMR_ROOT / "assets" / "body_models"

# ── 人體骨骼連線（parent → child）────────────────────────────────────────────
SMPLX_BONES = [
    ("pelvis",    "left_hip"),
    ("pelvis",    "right_hip"),
    ("pelvis",    "spine1"),
    ("spine1",    "spine2"),
    ("spine2",    "spine3"),
    ("spine3",    "neck"),
    ("neck",      "head"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("left_ankle",     "left_foot"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("right_ankle",    "right_foot"),
    ("spine3",         "left_collar"),
    ("left_collar",    "left_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("spine3",         "right_collar"),
    ("right_collar",   "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
]

SMPLX_MAIN_JOINTS = {
    "pelvis", "spine1", "spine2", "spine3", "neck", "head",
    "left_hip", "left_knee", "left_ankle", "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "left_collar", "left_shoulder", "left_elbow", "left_wrist",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist",
}

# ── SMPL-X y-up → MuJoCo z-up 坐標系轉換 ────────────────────────────────────
# y-up → z-up: new_x=x, new_y=-z, new_z=y
_R_YUP_ZUP = np.array([[1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]], dtype=np.float64)

# 對應四元數（繞 x 軸 +90°，wxyz）
_Q_YUP_ZUP = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])


def _yup_pos_to_zup(pos):
    return _R_YUP_ZUP @ np.asarray(pos, dtype=np.float64)


def _qmul(a, b):
    """四元數乘法，wxyz 格式。"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _yup_quat_to_zup(q_wxyz):
    """把 y-up world 的四元數轉到 z-up world：q_zup = Q_yup2zup * q_yup。"""
    return _qmul(_Q_YUP_ZUP, np.asarray(q_wxyz, dtype=np.float64))


# ── 自定義幾何注入 ─────────────────────────────────────────────────────────────
def _make_rot_mat(direction):
    """
    構造旋轉矩陣，使 +z 列方向對齊 direction 向量。
    （MuJoCo Arrow/Capsule 沿自身 z 軸延伸）
    """
    z = direction / np.linalg.norm(direction)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]).astype(np.float32)


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    g.type      = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:3]  = [radius, radius, radius]
    g.pos[:]    = pos
    g.mat[:]    = np.eye(3, dtype=np.float32)
    g.rgba[:]   = rgba
    g.emission  = 0.3
    g.dataid    = -1
    g.objtype   = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid     = -1
    g.category  = mujoco.mjtCatBit.mjCAT_DECOR
    g.segid     = -1
    scene.ngeom += 1


def add_arrow(scene, from_pos, to_pos, radius, rgba):
    """從 from_pos 到 to_pos 畫箭頭。"""
    if scene.ngeom >= scene.maxgeom:
        return
    from_pos = np.asarray(from_pos, dtype=np.float64)
    to_pos   = np.asarray(to_pos,   dtype=np.float64)
    diff     = to_pos - from_pos
    length   = np.linalg.norm(diff)
    if length < 1e-9:
        return

    g = scene.geoms[scene.ngeom]
    g.type      = mujoco.mjtGeom.mjGEOM_ARROW
    g.size[:3]  = [radius, radius, length / 2]
    g.pos[:]    = (from_pos + to_pos) / 2
    g.mat[:]    = _make_rot_mat(diff)
    g.rgba[:]   = rgba
    g.emission  = 0.3
    g.dataid    = -1
    g.objtype   = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid     = -1
    g.category  = mujoco.mjtCatBit.mjCAT_DECOR
    g.segid     = -1
    scene.ngeom += 1


def add_capsule(scene, p1, p2, radius, rgba):
    """用 Capsule 繪製骨骼連線。"""
    if scene.ngeom >= scene.maxgeom:
        return
    p1   = np.asarray(p1, dtype=np.float64)
    p2   = np.asarray(p2, dtype=np.float64)
    diff = p2 - p1
    length = np.linalg.norm(diff)
    if length < 1e-9:
        return

    g = scene.geoms[scene.ngeom]
    g.type      = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.size[:3]  = [radius, radius, length / 2]
    g.pos[:]    = (p1 + p2) / 2
    g.mat[:]    = _make_rot_mat(diff)
    g.rgba[:]   = rgba
    g.emission  = 0.1
    g.dataid    = -1
    g.objtype   = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid     = -1
    g.category  = mujoco.mjtCatBit.mjCAT_DECOR
    g.segid     = -1
    scene.ngeom += 1


def draw_coord_frame(scene, pos, quat_wxyz, axis_len=0.07, radius=0.004):
    """在 pos 處繪製 xyz 三色坐標軸箭頭（紅=X，綠=Y，藍=Z）。"""
    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    axis_colors = [
        [1.0, 0.0, 0.0, 1.0],   # X 紅
        [0.0, 0.85, 0.0, 1.0],  # Y 綠
        [0.1, 0.3, 1.0, 1.0],   # Z 藍
    ]
    for i, rgba in enumerate(axis_colors):
        d = np.zeros(3); d[i] = axis_len
        d = rot.apply(d)
        add_arrow(scene, pos, np.asarray(pos) + d, radius, rgba)


# ── 獲取機器人默認姿態 ─────────────────────────────────────────────────────────
def get_robot_bodies(xml_path):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)
    data.qpos[:] = 0
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        qz = model.qpos0[2]
        data.qpos[2] = qz if abs(qz) > 0.01 else 0.868
    mujoco.mj_forward(model, data)

    bodies = {}
    for i in range(1, model.nbody):
        name  = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        pid   = model.body_parentid[i]
        pname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, pid)
                 if pid > 0 else None)
        bodies[name] = {
            "pos":    data.xpos[i].copy(),
            "quat":   data.xquat[i].copy(),   # wxyz
            "parent": pname,
        }
    return bodies


# ── 每幀渲染回調 ───────────────────────────────────────────────────────────────
def build_scene_callback(frames, robot_bodies, state, human_offset, axis_len, show_axes,
                         ik_rot_offsets=None):
    """
    ik_rot_offsets: dict, human_joint_name → np.array(4, wxyz)
        当提供时，同时画出 IK 目标坐标轴（白色）= q_human × rot_offset
        用于验证 rot_offset 设置是否合理
    """

    def callback(scene):
        scene.ngeom = 0   # 清空上一幀自定義幾何
        frame = frames[state["frame"]]

        # ── 人體骨骼 ───────────────────────────────────────────────────────
        jpos_zup  = {}
        jquat_zup = {}

        for jname in SMPLX_MAIN_JOINTS:
            if jname not in frame:
                continue
            pos_yup  = np.array(frame[jname][0], dtype=np.float64)
            quat_yup = np.array(frame[jname][1], dtype=np.float64)

            pos_z  = _yup_pos_to_zup(pos_yup) + human_offset
            quat_z = _yup_quat_to_zup(quat_yup)

            jpos_zup[jname]  = pos_z
            jquat_zup[jname] = quat_z

            # 節點球顏色：左藍 / 右橙 / 中紫
            if jname.startswith("left_"):
                rgba = [0.2, 0.55, 1.0, 0.95]
            elif jname.startswith("right_"):
                rgba = [1.0, 0.5, 0.05, 0.95]
            else:
                rgba = [0.75, 0.35, 0.95, 0.95]
            add_sphere(scene, pos_z, 0.022, rgba)

            if show_axes:
                draw_coord_frame(scene, pos_z, quat_z, axis_len, 0.0035)

            # IK 目标坐标轴（白色粗箭头）= q_human_yup × rot_offset
            if ik_rot_offsets and jname in ik_rot_offsets:
                rot_off = ik_rot_offsets[jname]
                q_human = R.from_quat([quat_yup[1], quat_yup[2], quat_yup[3], quat_yup[0]])
                q_off   = R.from_quat([rot_off[1],  rot_off[2],  rot_off[3],  rot_off[0]])
                ik_quat_xyzw = (q_human * q_off).as_quat()
                ik_quat_wxyz = np.array([ik_quat_xyzw[3], ik_quat_xyzw[0],
                                         ik_quat_xyzw[1], ik_quat_xyzw[2]])
                draw_coord_frame(scene, pos_z, ik_quat_wxyz, axis_len * 1.4, 0.006)

        # 骨骼連線
        for p_name, c_name in SMPLX_BONES:
            if p_name not in jpos_zup or c_name not in jpos_zup:
                continue
            if "left" in p_name or "left" in c_name:
                bone_rgba = [0.2, 0.55, 1.0, 0.7]
            elif "right" in p_name or "right" in c_name:
                bone_rgba = [1.0, 0.5, 0.05, 0.7]
            else:
                bone_rgba = [0.75, 0.35, 0.95, 0.7]
            add_capsule(scene, jpos_zup[p_name], jpos_zup[c_name], 0.008, bone_rgba)

        # ── 機器人骨骼 ─────────────────────────────────────────────────────
        for bname, bdata in robot_bodies.items():
            pos  = bdata["pos"]
            quat = bdata["quat"]

            if "left" in bname:
                rgba = [0.1, 0.9, 0.45, 0.7]
            elif "right" in bname:
                rgba = [1.0, 0.25, 0.25, 0.7]
            elif "head" in bname:
                rgba = [1.0, 0.88, 0.05, 0.7]
            else:
                rgba = [0.65, 0.65, 0.65, 0.6]

            add_sphere(scene, pos, 0.020, rgba)

            if show_axes:
                draw_coord_frame(scene, pos, quat, axis_len, 0.003)

        # 機器人骨骼連線（從父子關係自動生成）
        for bname, bdata in robot_bodies.items():
            pname = bdata["parent"]
            if pname and pname in robot_bodies:
                p1 = robot_bodies[pname]["pos"]
                p2 = bdata["pos"]
                if "left" in bname:
                    bone_rgba = [0.1, 0.9, 0.45, 0.5]
                elif "right" in bname:
                    bone_rgba = [1.0, 0.25, 0.25, 0.5]
                else:
                    bone_rgba = [0.55, 0.55, 0.55, 0.45]
                add_capsule(scene, p1, p2, 0.007, bone_rgba)

    return callback


# ── 主程序 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo Viewer：SMPL-X 人體骨骼 + 機器人 XML 默認姿態"
    )
    parser.add_argument("--smplx_file", required=True,
                        help="AMASS/SMPL-X .npz 文件路徑")
    parser.add_argument("--robot", default="casbot_skeleton",
                        help="機器人名稱，需在 ROBOT_XML_DICT 中")
    parser.add_argument("--frame", default=0, type=int,
                        help="起始幀索引")
    parser.add_argument("--fps",   default=30, type=int)
    parser.add_argument("--human_offset_y", default=1.5, type=float,
                        help="人體相對機器人的 y 偏移（m）")
    parser.add_argument("--axis_len", default=0.08, type=float,
                        help="坐標軸箭頭長度（m）")
    parser.add_argument("--no_axes", action="store_true",
                        help="隱藏坐標軸，只顯示骨骼")
    parser.add_argument("--show_ik_targets", action="store_true",
                        help="疊加顯示 IK 目標坐標軸（q_human × rot_offset），用於驗證 rot_offset 設置")
    args = parser.parse_args()

    # ── 加載 IK 配置（可选）────────────────────────────────────────────────
    ik_rot_offsets = None   # human_joint_name → np.array(4, wxyz)
    if args.show_ik_targets:
        if "smplx" not in IK_CONFIG_DICT or args.robot not in IK_CONFIG_DICT["smplx"]:
            print(f"[WARN] 找不到 smplx→{args.robot} 的 IK 配置，不顯示 IK 目標")
        else:
            ik_cfg_path = IK_CONFIG_DICT["smplx"][args.robot]
            with open(ik_cfg_path) as f:
                ik_cfg = json.load(f)
            # 建立 human_joint → rot_offset 映射（合并 table1 和 table2）
            ik_rot_offsets = {}
            for table_name in ["ik_match_table1", "ik_match_table2"]:
                if table_name not in ik_cfg:
                    continue
                for robot_body, entry in ik_cfg[table_name].items():
                    human_joint = entry[0]
                    rot_off_wxyz = np.array(entry[4], dtype=np.float64)
                    ik_rot_offsets[human_joint] = rot_off_wxyz
            print(f"  載入 IK 配置: {ik_cfg_path}")
            print(f"  找到 {len(ik_rot_offsets)} 個關節的 rot_offset：{list(ik_rot_offsets.keys())}")

    # ── 加載數據 ──────────────────────────────────────────────────────────
    print("加載 SMPL-X 數據...")
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.fps
    )
    print("-" * 150)
    print(frames[0])
    n_frames = len(frames)
    print(f"  人體身高={human_height:.3f}m  幀數={n_frames}  fps={aligned_fps}")

    if args.robot not in ROBOT_XML_DICT:
        print(f"[ERROR] robot '{args.robot}' 不在 ROBOT_XML_DICT")
        sys.exit(1)
    xml_path = ROBOT_XML_DICT[args.robot]
    print(f"加載機器人: {xml_path.name}")
    robot_bodies = get_robot_bodies(xml_path)

    # ── 啟動 MuJoCo viewer ──────────────────────────────────────────────
    robot_model = mujoco.MjModel.from_xml_path(str(xml_path))
    robot_data  = mujoco.MjData(robot_model)
    robot_data.qpos[:] = 0
    if robot_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        qz = robot_model.qpos0[2]
        robot_data.qpos[2] = qz if abs(qz) > 0.01 else 0.868
    mujoco.mj_forward(robot_model, robot_data)

    state = {
        "frame":   max(0, min(args.frame, n_frames - 1)),
        "playing": False,
        "exit":    False,
    }

    human_offset = np.array([0.0, args.human_offset_y, 0.0])

    scene_cb = build_scene_callback(
        frames, robot_bodies, state,
        human_offset, args.axis_len, not args.no_axes,
        ik_rot_offsets=ik_rot_offsets,
    )

    # MuJoCo 鍵碼常量
    KEY_LEFT  = 263
    KEY_RIGHT = 262
    KEY_SPACE = 32
    KEY_Q     = 81
    KEY_ESC   = 256
    KEY_HOME  = 268
    KEY_END   = 269

    def key_callback(keycode):
        if keycode == KEY_RIGHT:
            state["frame"] = min(state["frame"] + 1, n_frames - 1)
        elif keycode == KEY_LEFT:
            state["frame"] = max(state["frame"] - 1, 0)
        elif keycode == KEY_SPACE:
            state["playing"] = not state["playing"]
            print(f"\n  {'▶ 播放' if state['playing'] else '⏸ 暫停'}")
        elif keycode == KEY_HOME:
            state["frame"] = 0
        elif keycode == KEY_END:
            state["frame"] = n_frames - 1
        elif keycode in (KEY_Q, KEY_ESC):
            state["exit"] = True
            return
        print(f"\r  Frame {state['frame']:4d}/{n_frames-1}", end="", flush=True)

    print()
    print("場景說明：")
    print(f"  原點附近  → 機器人 [{args.robot}] 默認姿態（綠=左，紅=右，灰=其他）")
    print(f"  +y {args.human_offset_y}m 處 → SMPL-X 人體骨骼，已轉 z-up（藍=左，橙=右，紫=中）")
    print(f"  坐標軸箭頭（細）：紅=X  綠=Y  藍=Z  ← 世界空間朝向")
    if ik_rot_offsets:
        print(f"  坐標軸箭頭（粗白）← IK 目標朝向 = q_human × rot_offset")
        print(f"  [驗證方法] 粗白箭頭應和機器人對應節點的坐標軸方向吻合")
    print()
    print("鍵盤操作：← →=切幀  Space=播放/暫停  Home/End=首/末幀  Q/Esc=退出")
    print()

    FRAME_DT = 1.0 / aligned_fps
    last_t   = time.time()

    with mujoco.viewer.launch_passive(
        robot_model, robot_data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        # 初始相機：略微俯視，兩個骨骼都在視野中
        viewer.cam.distance  = 5.0
        viewer.cam.azimuth   = 140.0
        viewer.cam.elevation = -18.0
        viewer.cam.lookat[:] = [0.0, args.human_offset_y / 2, 0.9]

        # 顯示機器人半透明（讓內部骨骼線也可見）
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        while viewer.is_running() and not state["exit"]:
            now = time.time()

            # 自動播放
            if state["playing"] and now - last_t >= FRAME_DT:
                state["frame"] = (state["frame"] + 1) % n_frames
                last_t = now
                print(f"\r  Frame {state['frame']:4d}/{n_frames-1}", end="", flush=True)

            # 注入自定義幾何
            viewer.user_scn.ngeom = 0
            scene_cb(viewer.user_scn)

            viewer.sync()
            time.sleep(0.005)

    print("\n退出。")


if __name__ == "__main__":
    main()
