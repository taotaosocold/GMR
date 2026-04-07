import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def load_bvh_file(bvh_file, format="lafan1"):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    #解析bvh文件
    data = read_bvh(bvh_file)
    # 前向运动学+坐标系变换，返回的是一个(positions, rotations)元组
    # 通过global_data[0][frame, i]来获得第frame帧的第i个关节全局位置，通过global_data[2][frame, i]来获得第frame帧的第i个关节全局朝向
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)
    # bvh保准坐标系是Y-up即y轴朝上，但机器人是Z-up。所以这里做了一个坐标转换
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # 然后将这个值变成四元数
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    # 后续每一帧所有朝向都和这个四元数做一个运算，变成z轴朝上
    frames = []
    # 遍历拜访每一帧
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = [position, orientation]
        # 其中对于不同的数据格式，会有一个处理
        if format == "lafan1":
            # Add modified foot pose
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]
        elif format == "nokov":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
        else:
            raise ValueError(f"Invalid format: {format}")
            
        frames.append(result)
    
    # human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m
    # 最后的frames是一个列表[0,1,2,3,...]每一个索引表示对应帧，然后每一个索引是一个字典，键为节点名称，键值又是列表为[positions, rotations]为全局值。
    return frames, human_height


