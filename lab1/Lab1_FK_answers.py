from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from enum import Enum
from typing import Union
from typing import List
from typing import Tuple

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data

def load_bones_source(bvh_file_path):
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        size = len(lines)
        i = 0
        j = 0
        for i in range(size):
            if lines[i].startswith('HIERARCHY'):
                break
        for j in range(i+1, size):
            if lines[j].startswith('MOTION'):
                break
        source = []
        for line in lines[i+1:j]:
            for word in line.split():
                source.append(word)
    return source

class TOKEN:
    pass

class ROOT(TOKEN):
    pass

class LPAR(TOKEN):
    pass

class RPAR(TOKEN):
    pass

@dataclass
class STR(TOKEN):
    value: str

@dataclass
class INT(TOKEN):
    value: int

@dataclass
class FLOAT(TOKEN):
    value: float

class OFFSET(TOKEN):
    pass

class CHANNELS(TOKEN):
    pass

class JOINT(TOKEN):
    pass

class END(TOKEN):
    pass

def intp(word):
    try: 
        int(word)
        return True
    except: return False

def floatp(word):
    try:
        float(word)
        return True
    except:
        return False

def token(word: str) -> TOKEN:
    if word == 'ROOT':
        return ROOT()
    if word == '{':
        return LPAR()
    if word == '}':
        return RPAR()
    if word == 'OFFSET':
        return OFFSET()
    if word == 'CHANNELS':
        return CHANNELS()
    if word == 'JOINT':
        return JOINT()
    if word == 'End':
        return END()
    if intp(word):
        return INT(int(word))
    if floatp(word):
        return FLOAT(float(word))
    if type(word) == str:
        return STR(word)
    else:
        raise Exception('unexcept token word ' + word)

def token_pass(source: List[str]) -> List[TOKEN]:
    tokens = []
    for word in source:
        x = token(word)
        if (isinstance(x, TOKEN)):
            tokens.append(x)
        else:
            raise Exception('tokenize error')
    return tokens

@dataclass
class End:
    name: str
    offset: List[float]

@dataclass
class Joint:
    name: str
    offset: List[float]
    channels: List[str]
    joints: Union[End, List[Joint]]

@dataclass
class Root:
    name: str
    offset: List[float]
    channels: List[str]
    joints: Union[End, List[Joint]]

# V variable, E terminal, S start variable, R relation
# RootJoint -> root name { Offset Channels EndOrJoints }
# EndOrJoints -> End | Joints
# End -> end name { Offset }
# Joints -> Joint, Joints
# Joint -> joint name { Offset Channels EndOrJoints }
# Offset -> offset Floats
# Floats -> float | float, floats
# Channels -> channel int Strings
# Strings -> string | string, Strings

def make_consume_parser(t: type):
    def consume_parser(tokens: List[TOKEN]) -> Tuple[List[TOKEN]]:
        token, *xs = tokens
        if (isinstance(token, t)):
            return (xs,)
        else:
            raise Exception('incorrect ' + t.__name__ + ' token ' + str(token) + ', ' + str(tokens))
    return consume_parser

def parseRootSymbol(tokens: List[TOKEN]) -> Tuple[List[TOKEN]]:
    return make_consume_parser(ROOT)(tokens)

def parseStr(tokens: List[TOKEN]) -> Tuple[List[TOKEN], str]:
    token, *xs = tokens
    if (isinstance(token, STR)):
        return (xs, token.value)
    else:
        raise Exception('incorrect STR token ' + str(token) + ', ' + str(tokens))

def parseFloat(tokens: List[TOKEN]) -> Tuple[List[TOKEN], float]:
    token, *xs = tokens
    if (isinstance(token, FLOAT)):
        return (xs, token.value)
    else:
        raise Exception('incorrect FLOAT token ' + str(token) + ', ' + str(tokens))

def parseInt(tokens: List[TOKEN]) -> Tuple[List[TOKEN], int]:
    token, *xs = tokens
    if (isinstance(token, INT)):
        return (xs, token.value)
    else:
        raise Exception('incorrect INT token ' + str(token))

def parseLPar(tokens: List[TOKEN]) -> Tuple[List[TOKEN]]:
    return make_consume_parser(LPAR)(tokens)

def parseRPar(tokens: List[TOKEN]) -> Tuple[List[TOKEN]]:
    return make_consume_parser(RPAR)(tokens)

def parseOffsetSymbol(tokens: List[TOKEN]) -> Tuple[List[TOKEN]]:
    return make_consume_parser(OFFSET)(tokens)

def parseOffset(tokens: List[TOKEN]) -> Tuple[List[TOKEN], List[float]]:
    (r0,) = parseOffsetSymbol(tokens)
    (r1, x) = parseFloat(r0)
    (r2, y) = parseFloat(r1)
    (r3, z) = parseFloat(r2)
    return (r3, [x, y, z])

def parseChannelsSymbol(tokens: List[TOKEN]) -> Tuple[List[TOKEN]]:
    return make_consume_parser(CHANNELS)(tokens)

def parseChannels(tokens: List[TOKEN]) -> Tuple[List[TOKEN], List[str]]:
    (r0,) = parseChannelsSymbol(tokens)
    (r1, num) = parseInt(r0)
    channels = []
    for _ in range(num):
        (r1, channelName) = parseStr(r1)
        channels.append(channelName)
    return (r1, channels)

def parseJointSymbol(tokens: List[TOKEN]) -> Tuple[List[TOKEN]]:
    return make_consume_parser(JOINT)(tokens)

def parseJoint(tokens: List[TOKEN]) -> Tuple[List[TOKEN], Joint]:
    (r0,) = parseJointSymbol(tokens)
    (r1, name) = parseStr(r0)
    (r2,) = parseLPar(r1)
    (r3, offset) = parseOffset(r2)
    (r4, channels) = parseChannels(r3)
    (r5, joints) = parseJoints(r4, name)
    (r6,) = parseRPar(r5)
    return (r6, Joint(name, offset, channels, joints))

def parseEndSymbol(tokens: List[TOKEN]) -> Tuple[List[TOKEN]]:
    return make_consume_parser(END)(tokens)

def parseEnd(tokens: List[TOKEN], parent_name: str) -> Tuple[List[TOKEN], End]:
    (r0,) = parseEndSymbol(tokens)
    (r1, _name) = parseStr(r0)
    (r2,) = parseLPar(r1)
    (r3, offset) = parseOffset(r2)
    (r4,) = parseRPar(r3)
    return (r4, End(parent_name + '_end', offset))

def parseJoints(tokens: List[TOKEN], parent_name: str) -> Tuple[List[TOKEN], Union[End, List[Joint]]]:
    joints = []
    r1 = tokens
    try:
        while True:
            (r1, joint) = parseJoint(r1)
            joints.append(joint)
    except:
        if len(joints) > 0:
            return (r1, joints)
        else:
            try:
                (r0, end) = parseEnd(tokens, parent_name)
                return (r0, end)
            except:
                raise Exception('can not parse joints ' + str(r1))

def parseRoot(tokens: List[TOKEN]) -> Root:
    (r0,) = parseRootSymbol(tokens)
    (r1, name) = parseStr(r0)
    (r2,) = parseLPar(r1)
    (r3, offset) = parseOffset(r2)
    (r4, channels) = parseChannels(r3)
    (r5, joints) = parseJoints(r4, name)
    (_r6,) = parseRPar(r5)
    return Root(name, offset, channels, joints)

def evalRoot(root: Root) -> Tuple[List[str], List[int], np.ndarray]:
    (joint_name, joint_parent, joint_offset) = evalJoints(([root.name], [-1], [root.offset]), 0, root.joints)
    offset = np.ndarray((len(joint_offset), 3), buffer=np.array(joint_offset), dtype=float)
    return (joint_name, joint_parent, offset)

def evalEnd(input: Tuple[List[str], List[int], List[List[float]]], parent: int, end: End) -> Tuple[List[str], List[int], List[List[float]]]:
    (joint_name, joint_parent, joint_offset) = input
    return ([*joint_name, end.name], [*joint_parent, parent], [*joint_offset, end.offset])

def evalJoint(input: Tuple[List[str], List[int], List[List[float]]], parent: int, joint: Joint) -> Tuple[List[str], List[int], List[List[float]]]:
    (joint_name, joint_parent, joint_offset) = input
    cur = len(joint_parent)
    return evalJoints(([*joint_name, joint.name], [*joint_parent, parent], [*joint_offset, joint.offset]), cur, joint.joints)

def evalJoints(input: Tuple[List[str], List[int], List[List[float]]], parent: int, joints: Union[End, List[Joint]]) -> Tuple[List[str], List[int], List[List[float]]]:
    if isinstance(joints, End):
        return evalEnd(input, parent, joints)
    elif isinstance(joints, list):
        res = input
        for joint in joints:
            res = evalJoint(res, parent, joint)
        return res
    else:
        raise Exception('incorrect joints type')

def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    source = load_bones_source(bvh_file_path)
    tokens = token_pass(source)
    root = parseRoot(tokens)
    res = evalRoot(root)
    return res

'''
def acc_orientation(joint_parent: List[int], rotations: np.ndarray, cur: int, acc: np.ndarray) -> np.ndarray:
    if (cur == -1):
        return acc
    else:
        return acc_orientation(joint_parent, rotations, joint_parent[cur], acc * rotations[cur])
'''

def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    # root num 1 channels 6
    # joints num 19 channels 19*3= 57
    # end num 5 channels 0
    
    frame_motion_data = motion_data[frame_id]
    joint_translation = frame_motion_data[0:3]
    
    #joint_positions = np.ndarray((len(positions), 3), buffer=np.array(positions), dtype=float)
    #print(joint_positions)
    joints_num = len(joint_parent)

    ends_map = [True] * joints_num
    for i in range(joints_num):
        par = joint_parent[i]
        if (par >= 0):
            ends_map[par] = False
    rotations = np.ndarray((joints_num, 3), buffer=np.array([0,0,0] * joints_num), dtype=float)
    p = 3
    for i in range(joints_num):
        if not ends_map[i]:
            p_end = p + 3
            rotations[i] = frame_motion_data[p : p_end]
            p = p_end
    orientations = []
    for i in range(joints_num):
        par = joint_parent[i]
        rotation = R.from_euler('XYZ', rotations[i], degrees=True).as_quat()
        if (par == -1):
            orientations.append(rotation)
        else:
            orientations.append((R.from_quat(orientations[par]) * R.from_quat(rotation)).as_quat())

    positions = []
    for i in range(joints_num):
        par = joint_parent[i]
        v = np.array(joint_offset[i]).transpose()
        if (par == -1):
            positions.append(joint_translation + np.array(joint_offset[i]))
        else:
            m = R.from_quat(orientations[par]).as_matrix()
            positions.append(m.dot(v) + positions[par])

    joint_orientations = np.ndarray((joints_num, 4), buffer=np.array(orientations), dtype=float)
    joint_positions = np.ndarray((joints_num, 3), buffer=np.array(positions), dtype=float)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
    """
    motion_data = None
    return motion_data
