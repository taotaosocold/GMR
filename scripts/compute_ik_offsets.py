"""
从 SMPL-X 默认姿态（poses=0, trans=0）和机器人 XML 默认姿态自动计算
rot_offset 和 scale，并写回 IK 配置 JSON。

核心公式：
    q_human_zup = Q_YUP2ZUP × q_human_yup      # 先把 SMPL-X y-up 四元数转到 z-up
    rot_offset  = inv(q_human_zup) × q_robot_zup

其中：
  q_human_yup  = SMPL-X 参考帧各关节全局四元数（y-up 世界，原始输出）
  q_robot_zup  = 机器人各 body 在 FK 下的全局四元数（z-up MuJoCo 世界）
  Q_YUP2ZUP    = [cos45°, sin45°, 0, 0]（绕 x 轴 +90°，y-up→z-up 世界变换）

⚠️  重要前提：
  此方法仅在机器人 XML 对不同 body 定义了不同的 <body quat="..."> 偏移时有效。
  若所有 body 在 joints=0 时 xquat 均为 [1,0,0,0]，说明 XML 未定义局部坐标轴，
  此时各关节算出来的 rot_offset 相同，无法区分 pitch/yaw/roll，需改用解析法。

  建议先运行 --dry_run 检查机器人各 body 在 joints=0 时的 xquat 是否有差异，
  再决定是否使用本方法。

用法：
  # 使用机器人 joints=0 默认姿态
  python scripts/compute_ik_offsets.py \\
      --smplx_file /home/casbot/Desktop/hjq/motion/text.npz \\
      --robot marathon_001

  # 指定参考帧（默认自动选最直立帧）
  python scripts/compute_ik_offsets.py \\
      --smplx_file /path/to/tpose.npz \\
      --robot marathon_001 \\
      --ref_frame 0

  # 让机器人用自定义 qpos 进入 T-pose 再计算（.npy 文件，shape=(nq,)）
  python scripts/compute_ik_offsets.py \\
      --smplx_file /path/to/tpose.npz \\
      --robot marathon_001 \\
      --qpos_file /path/to/robot_tpose.npy

  # 不更新 scale，只更新 rot_offset
  python scripts/compute_ik_offsets.py \\
      --smplx_file /path/to/tpose.npz \\
      --robot marathon_001 \\
      --no_scale

  # 输出到新文件而不覆盖原始 JSON
  python scripts/compute_ik_offsets.py \\
      --smplx_file /path/to/tpose.npz \\
      --robot marathon_001 \\
      --output /tmp/new_config.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

GMR_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

SMPLX_FOLDER = GMR_ROOT / "assets" / "body_models"


# ── 四元数工具（wxyz 格式） ───────────────────────────────────────────────────

# y-up → z-up 世界坐标系变换：绕 x 轴 +90°，wxyz = [cos45°, sin45°, 0, 0]
_Q_YUP2ZUP = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_inv(q):
    """单位四元数的逆 = 共轭"""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def norm_quat(q):
    return np.array(q) / np.linalg.norm(q)


def wxyz_to_euler(q):
    rot = R.from_quat([q[1], q[2], q[3], q[0]])  # wxyz → xyzw
    return rot.as_euler("xyz", degrees=True)


def human_yup_to_zup(q_wxyz):
    """把 SMPL-X y-up 世界的四元数转换到 z-up 世界：q_zup = Q_YUP2ZUP × q_yup"""
    return norm_quat(quat_mul(_Q_YUP2ZUP, np.asarray(q_wxyz, dtype=np.float64)))


# ── 加载机器人默认姿态 ────────────────────────────────────────────────────────

def load_robot_bodies(xml_path, qpos_override=None):
    """
    返回 dict: body_name → {"pos": np.array(3), "quat": np.array(4, wxyz)}
    qpos_override: 若为 None 则使用 model.qpos0（XML 中 <key> 或 default）
    """
    model = mj.MjModel.from_xml_path(str(xml_path))
    data  = mj.MjData(model)

    if qpos_override is not None:
        if len(qpos_override) != model.nq:
            print(f"  [WARN] qpos_override 长度 {len(qpos_override)} ≠ model.nq {model.nq}，忽略")
            data.qpos[:] = model.qpos0.copy()
        else:
            data.qpos[:] = qpos_override
    else:
        data.qpos[:] = model.qpos0.copy()

    mj.mj_forward(model, data)

    bodies = {}
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name is None or name == "world":
            continue
        bodies[name] = {
            "pos":  data.xpos[i].copy(),
            "quat": data.xquat[i].copy(),   # wxyz
        }
    return bodies, model.nq


# ── 加载 SMPL-X 参考帧 ────────────────────────────────────────────────────────

def load_human_frame(smplx_file, ref_frame_idx=-1):
    """
    返回 (frame_dict, human_height)
    frame_dict: joint_name → {"pos": np.array(3), "quat": np.array(4, wxyz)}
    坐标系：SMPL-X 原始 y-up（不做转换，由 rot_offset 公式自然处理）
    """
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        str(smplx_file), SMPLX_FOLDER
    )
    frames, _ = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )

    if ref_frame_idx >= 0:
        idx = min(ref_frame_idx, len(frames) - 1)
        print(f"  使用指定帧 #{idx}")
    else:
        # 选骨盆"最直立"的帧：骨盆局部 y 轴在 y-up 世界中 y 分量最大
        best_idx, best_score = 0, -1e9
        for fi, frame in enumerate(frames):
            if "pelvis" not in frame:
                continue
            q = np.array(frame["pelvis"][1])   # wxyz
            rot = R.from_quat([q[1], q[2], q[3], q[0]])
            score = rot.apply([0, 1, 0])[1]    # 骨盆 local-y 在世界 y 方向的投影
            if score > best_score:
                best_score = score
                best_idx   = fi
        idx = best_idx
        print(f"  自动选帧 #{idx}，骨盆直立得分 = {best_score:.4f}")

    frame = frames[idx]
    result = {}
    for jname, data in frame.items():
        result[jname] = {
            "pos":  np.array(data[0]),
            "quat": np.array(data[1]),   # wxyz
        }
    return result, human_height


# ── 计算 offset 和 scale ──────────────────────────────────────────────────────

def compute_rot_offset(q_human_yup, q_robot_zup):
    """
    正确公式：直接用 y-up 人体四元数的逆乘 z-up 机器人四元数
        rot_offset = inv(q_human_yup) × q_robot_zup

    原因：motion_retarget.py 中计算 IK 目标时：
        ik_target = q_human_yup × rot_offset   （右乘，scipy Rotation）
    因此：rot_offset = inv(q_human_yup) × q_robot_desired

    在 T-pose 时 q_human_yup=[1,0,0,0]，所以：
        rot_offset = q_robot_desired（即机器人在 T-pose 等价姿态下的 xquat）

    ⚠️ 注意：不需要额外的 Q_YUP2ZUP 转换，因为 y-up→z-up 的变换
    已经内含在 rot_offset 的数值中（motion_retarget.py 直接用 y-up 原始四元数）。
    """
    return norm_quat(quat_mul(quat_inv(q_human_yup), q_robot_zup))


def compute_scale(robot_bodies, human_frame, robot_root, human_root,
                  robot_body, human_joint):
    """scale = |robot_body_pos - robot_root_pos| / |human_joint_pos - human_root_pos|"""
    if robot_root not in robot_bodies or human_root not in human_frame:
        return None
    if robot_body not in robot_bodies or human_joint not in human_frame:
        return None

    r_dist = np.linalg.norm(
        robot_bodies[robot_body]["pos"] - robot_bodies[robot_root]["pos"]
    )
    h_dist = np.linalg.norm(
        human_frame[human_joint]["pos"] - human_frame[human_root]["pos"]
    )
    if h_dist < 1e-6:
        return None
    return r_dist / h_dist


# ── 打印辅助 ──────────────────────────────────────────────────────────────────

def _arm_dir_str(quat_wxyz, local_axis=np.array([1, 0, 0])):
    """把 local_axis 旋转到世界系，返回方向字符串（用于肉眼检验）"""
    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    d = rot.apply(local_axis)
    return f"[{d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}]"


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从参考姿态计算 IK rot_offset 和 scale，写回 JSON"
    )
    parser.add_argument("--smplx_file", required=True,
                        help="AMASS/SMPL-X .npz（建议 poses=0, trans=0 的 T-pose 文件）")
    parser.add_argument("--robot", required=True,
                        help="机器人名称，需在 ROBOT_XML_DICT 中")
    parser.add_argument("--ref_frame", default=-1, type=int,
                        help="SMPL-X 参考帧索引；-1 = 自动选最直立帧")
    parser.add_argument("--qpos_file", default=None,
                        help="机器人参考 qpos .npy 文件；不提供则使用 XML 默认 qpos0")
    parser.add_argument("--no_scale", action="store_true",
                        help="不更新 human_scale_table，仅更新 rot_offset")
    parser.add_argument("--no_rot", action="store_true",
                        help="不更新 rot_offset，仅更新 scale")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印机器人各 body 的 xquat，不计算也不写文件。"
                             "用于诊断 joints=0 时各 body 是否有区分度")
    parser.add_argument("--output", default=None,
                        help="输出 JSON 路径；不提供则覆盖原始配置文件")
    args = parser.parse_args()

    # ── 检查 robot ────────────────────────────────────────────────────────────
    if args.robot not in ROBOT_XML_DICT:
        print(f"[ERROR] robot '{args.robot}' 不在 ROBOT_XML_DICT"); sys.exit(1)
    if args.robot not in IK_CONFIG_DICT["smplx"]:
        print(f"[ERROR] robot '{args.robot}' 不在 IK_CONFIG_DICT['smplx']"); sys.exit(1)

    xml_path       = ROBOT_XML_DICT[args.robot]
    ik_config_path = IK_CONFIG_DICT["smplx"][args.robot]

    # ── 读取 JSON ─────────────────────────────────────────────────────────────
    with open(ik_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    robot_root = cfg["robot_root_name"]
    human_root = cfg["human_root_name"]

    # ── 加载机器人 ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"【1】机器人：{args.robot}  XML: {xml_path.name}")
    print("="*70)

    qpos_override = None
    if args.qpos_file:
        qpos_override = np.load(args.qpos_file)
        print(f"  使用自定义 qpos: {args.qpos_file}")
    else:
        print("  使用 XML 默认 qpos0")

    robot_bodies, nq = load_robot_bodies(xml_path, qpos_override)
    print(f"  nq={nq}，共 {len(robot_bodies)} 个 body")

    # ── dry_run：只打印 robot body xquat 分布 ────────────────────────────────
    if args.dry_run:
        print("\n" + "="*70)
        print("【dry_run】机器人各 body 在 joints=0 时的 xquat（判断是否有区分度）")
        print("="*70)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        n_identity = 0
        for bname, bdata in robot_bodies.items():
            q = bdata["quat"]
            is_id = np.allclose(q, identity, atol=1e-4)
            if is_id:
                n_identity += 1
            flag = "  ← [1,0,0,0]" if is_id else ""
            print(f"  {bname:40s}  {np.round(q, 4).tolist()}{flag}")
        print(f"\n  共 {len(robot_bodies)} 个 body，其中 {n_identity} 个 xquat = [1,0,0,0]")
        if n_identity == len(robot_bodies):
            print("\n  ⚠️  所有 body 的 xquat 均为 [1,0,0,0]。")
            print("     说明 XML 中各 body 没有定义局部坐标系偏移（<body quat>）。")
            print("     此时 T-pose 比较法无法区分不同关节，需改用解析法。")
        elif n_identity > len(robot_bodies) // 2:
            print(f"\n  ⚠️  超过一半的 body xquat = [1,0,0,0]，区分度有限，结果仅供参考。")
        else:
            print(f"\n  ✓  各 body 有不同 xquat，T-pose 比较法可用。")
        return

    if robot_root not in robot_bodies:
        print(f"[ERROR] robot_root '{robot_root}' 不在 XML body 列表中"); sys.exit(1)

    # ── 加载人体 ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"【2】SMPL-X 参考帧：{args.smplx_file}")
    print("="*70)

    human_frame, human_height = load_human_frame(args.smplx_file, args.ref_frame)
    print(f"  人体身高 = {human_height:.3f} m，共 {len(human_frame)} 个关节")

    if human_root not in human_frame:
        print(f"[ERROR] human_root '{human_root}' 不在 SMPL-X 关节列表中"); sys.exit(1)

    # ── 收集所有 (robot_body → human_body) 映射 ──────────────────────────────
    # table1 和 table2 共享 rot_offset/scale，只计算一次
    all_mappings = {}   # robot_body → human_body
    for table_name in ["ik_match_table1", "ik_match_table2"]:
        if table_name not in cfg:
            continue
        for robot_body, entry in cfg[table_name].items():
            human_body = entry[0]
            if robot_body not in all_mappings:
                all_mappings[robot_body] = human_body

    # ── 计算 ──────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("【3】计算 rot_offset 和 scale")
    print("="*70)

    computed_rot    = {}    # robot_body  → np.array(4, wxyz)
    raw_scales      = {}    # human_body  → list of float（左右平均用）

    for robot_body, human_body in all_mappings.items():
        missing = []
        if robot_body not in robot_bodies:
            missing.append(f"robot '{robot_body}'")
        if human_body not in human_frame:
            missing.append(f"human '{human_body}'")
        if missing:
            print(f"  [SKIP] {', '.join(missing)} 数据缺失")
            continue

        q_h_yup = human_frame[human_body]["quat"]   # y-up 世界（原始，motion_retarget.py 直接使用）
        q_r     = robot_bodies[robot_body]["quat"]   # z-up MuJoCo 世界

        rot_off = compute_rot_offset(q_h_yup, q_r)
        computed_rot[robot_body] = rot_off

        scale = compute_scale(robot_bodies, human_frame,
                              robot_root, human_root,
                              robot_body, human_body)
        if scale is not None:
            raw_scales.setdefault(human_body, []).append(scale)

        euler = wxyz_to_euler(rot_off)
        print(f"\n  {robot_body:40s} ← {human_body}")
        print(f"    human  quat (y-up)  : {np.round(q_h_yup, 4).tolist()}")
        print(f"    robot  quat (z-up)  : {np.round(q_r, 4).tolist()}")
        print(f"    rot_offset          : {np.round(rot_off, 6).tolist()}")
        print(f"    euler (xyz deg)     : {np.round(euler, 1)}")
        if scale is not None:
            print(f"    scale               : {scale:.4f}")

    # 对 left/right 对称关节取平均 scale（减少误差）
    avg_scales = {hb: float(np.mean(vals)) for hb, vals in raw_scales.items()}

    # ── 写回 JSON ─────────────────────────────────────────────────────────────
    if not args.no_rot:
        for table_name in ["ik_match_table1", "ik_match_table2"]:
            if table_name not in cfg:
                continue
            for robot_body, entry in cfg[table_name].items():
                if robot_body in computed_rot:
                    entry[4] = np.round(computed_rot[robot_body], 8).tolist()

    if not args.no_scale:
        for human_body, scale in avg_scales.items():
            if human_body in cfg.get("human_scale_table", {}):
                cfg["human_scale_table"][human_body] = round(scale, 4)

    out_path = args.output or str(ik_config_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    # ── 摘要 ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("【4】摘要")
    print("="*70)

    if not args.no_rot:
        print("\nrot_offset（已写入）：")
        for rb, off in computed_rot.items():
            hb = all_mappings[rb]
            euler = wxyz_to_euler(off)
            print(f"  {rb:40s} ← {hb}")
            print(f"    {np.round(off, 6).tolist()}  euler={np.round(euler,1)} deg")

    if not args.no_scale:
        print("\nhuman_scale_table（已更新）：")
        for hb, sc in avg_scales.items():
            in_table = "✓" if hb in cfg.get("human_scale_table", {}) else "（不在 scale table 中，未写入）"
            print(f"  {hb:25s} scale={sc:.4f}  {in_table}")

    print(f"\n已写入: {out_path}")
    print("="*70)

    # ── 对齐检验提示 ─────────────────────────────────────────────────────────
    print("\n【提示】用 vis_smplx_and_robot.py 可视化验证：")
    print(f"  python scripts/vis_smplx_and_robot.py \\")
    print(f"      --smplx_file {args.smplx_file} \\")
    print(f"      --robot {args.robot}")
    print("  对比双方骨骼在各轴上的朝向是否一致，若手臂方向不匹配则需")
    print("  通过 --qpos_file 提供让机器人进入 T-pose 的关节角度后重新计算。")


if __name__ == "__main__":
    main()
