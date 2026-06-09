# 自动生成ik_config文件的human_scale与pos/quat_offset

## 安装要求
```bash
pip install lxml
pip install matplotlib
```

## 基础配置
-pose_inits中添加_tpose.json文件(设置机器人的初始位姿为T-pose) \
-ik_configs中添加bvh/smplx_to_robot_origin.json文件（主要需要joint_match）\
将人形机器人与human_data在T-pose下完全对齐

## 具体使用
BVH格式：
```bash
python ik_config_manager/generate_keypoint_mapping_bvh.py \
    --bvh_file ik_config_manager/TPOSE.bvh \
    --robot unitree_g1 \
    --loop \
    --robot_qpos_init ik_config_manager/pose_inits/unitree_g1_tpose.json \
    --ik_config_in general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json \
    --ik_config_out general_motion_retargeting/ik_configs/bvh_lafan1_to_g1_auto.json
```

SMPLX格式：
```bash
python ik_config_manager/generate_keypoint_mapping_smplx.py \
    --smplx_file ik_config_manager/SMPLX_TPOSE_UNIFIED_AMASS.npz \
    --robot unitree_g1 \
    --loop \
    --robot_qpos_init ik_config_manager/pose_inits/unitree_g1_tpose.json \
    --ik_config_in general_motion_retargeting/ik_configs/smplx_to_g1.json \
    --ik_config_out general_motion_retargeting/ik_configs/smplx_to_g1_auto.json
```


## 思路
目标：计算出GMR所需要的缩放比例(scale)、pos_offsets和rot_offsets
以SMPLX格式为例子

核心思路：将人体与机器人摆出同一个姿态(项目中是T-pose姿态)，然后将人体姿态缩放调整为机器人的姿态，此时的缩放比例、pos_offsets和rot_offsets就是既得的参数

前提：人体与机器人首先要摆成同一个姿态，项目中使用了T-pose姿态。SMPLX_TPOSE_UNIFIED_AMASS.npz文件是关于人体的运动序列(类似AMASS)，而其中的第一帧就是T-pose的姿势。 而在robot_qpos_init这个参数中会指定json文件，就是初始化机器人的状态，要让其调整为T-pose的姿势，里面的joint_pos和root_rot要手动调节朝向和关节角度要摆成T-pose，其中joint_pos可以使用可视化rviz来手动调节得到角度，root_rot可以直接使用unitre_g1_tpose中的参数即可。

第一步：计算缩放比例
目的：解决“人腿长、机器人腿短”的物理尺寸差异
1. 首先把机器人的“根”关节强行平移与人体的“根”关节完全重合
2. 其他的关节点位置随着根关节进行同样平移
3. 代码中强制规定左和右的缩放比例必须用同一个变量，防止缩放后出现大小腿现象
4. 将缩放比例作为一个随机变量，然后对人体进行缩放，并使用前向运动学算出每个关节的位置，然后和机器人的关节位置进行对较来作损失，然后梯度回传来更新这个比例参数

第二步：计算rot_offsets
经过第一步后虽然关节的位置大致是对齐了，但是每个连杆的朝向还是错的（比如人体默认手掌是朝里的，但是机器人可能默认手掌超外；也可能人体默认足脚尖与面朝相同，而机器人脚尖与面朝相反），这些都要矫正为人体的朝向。
而实际上我们是有了机器人每个关节的绝对朝向和人体关节的绝对朝向，所以rot_offsets就是这两个绝对朝向的旋转差值

第三部：计算pos_offsets
scale会解决长度比例，rot_offsets会解决朝向，但是实际上机器人的关节位置和人体的关节位置还是有位置差异（很有可能是胖瘦的问题，比如机器人的髋很宽或者肩很宽），pos_offsets就是进行微调。
使用前向运动学算出缩放比例后的人体关节位置，和机器人的实际关节位置能算出一个差异，这个差异是全局偏差，于人体的关节朝向矩阵相乘后，就能变成相对于某个关节的局部偏差。