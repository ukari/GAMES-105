from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union
from typing import List
from typing import Tuple

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    end_joint = meta_data.end_joint
    root_joint = meta_data.root_joint
    joint_parent = meta_data.joint_parent
    joint_name = meta_data.joint_name
    joint_initial_position = meta_data.joint_initial_position
    print(joint_initial_position == joint_positions)
    print(joint_positions)
    print(target_pose)
    path, path_name, path_end_to_root, path_start_to_root = meta_data.get_path_from_root_to_end()
    joint_positions, joint_orientations = ccd_once(path_end_to_root, path_start_to_root, joint_positions, joint_orientations, target_pose)
    return joint_positions, joint_orientations

def ccd_once(path_end_to_root: List[int], path_start_to_root: List[int], joint_positions: np.ndarray, joint_orientations: np.ndarray, target_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    positions = joint_positions.copy()
    orientations = joint_orientations
    print(path_end_to_root)
    print(path_start_to_root)
    joints_num = len(joint_positions)
    for i in range(len(path_end_to_root) - 1):
        cur = path_end_to_root[i]
        prev = path_end_to_root[i + 1]
        p0 = joint_positions[prev]
        r0 = joint_positions[cur] - joint_positions[prev]
        rt = target_pose - joint_positions[prev]
        cos_theta = r0.dot(rt) / (np.linalg.norm(r0) * np.linalg.norm(rt))
        theta = -np.rad2deg(np.arccos(cos_theta))
        print(r0, rt, cos_theta, theta)
        s = np.sin(theta/2)
        c = np.cos(theta/2)
        axis = r0 * rt
        quat = np.zeros([4])
        axis_unit = axis / np.linalg.norm(axis)
        quat[0:3] = axis_unit * s
        quat[3] = c
        rotation = R.from_quat(quat)
        print('deg1', rotation.as_euler('XYZ', degrees=True))
        r1 = rotation.apply(r0)
        p1 = r1 + p0
        print('r0', r0, 'r1', r1,'rt', rt)
  #      print()
        positions[cur] = p1
        orientations[cur] = (R.from_quat(orientations[cur]) * rotation).as_quat()
    return positions, orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations