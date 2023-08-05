import cv2
import numpy as np
from transforms3d.euler import euler2mat, mat2euler, quat2euler
from transforms3d.quaternions import quat2mat, mat2quat


def pos_euler_to_pose_4x4(pos, euler):
    pose = np.zeros((4, 4))
    mat = euler2mat(*euler)
    pose[0:3, 0:3] = mat[:, :]
    pose[0:3, -1] = pos[:]
    pose[-1, -1] = 1
    return pose

def pos_quat_to_pose_4x4(pos, quat):
    pose = np.zeros((4, 4))
    mat = quat2mat(quat)
    pose[0:3, 0:3] = mat[:, :]
    pose[0:3, -1] = pos[:]
    pose[-1, -1] = 1
    return pose

def rotvec2rotmat(rotvec):
    return np.array(cv2.Rodrigues(rotvec)[0]).astype(np.float32)

def rotvec2euler(rotvec):
    return np.array(mat2euler(rotvec2rotmat(rotvec))).astype(np.float32)

def rotvec2quat(rotvec):
    return np.array(mat2quat(rotvec2rotmat(rotvec))).astype(np.float32)

def rotmat2rotvec(rotmat):
    return np.array(cv2.Rodrigues(rotmat)[0]).astype(np.float32).reshape(-1)


def transform_timestamp_key(p):
    """Transform the timestamps as keys in the dictionary."""
    res = {}
    for key in p.keys():
        res_item = {}
        for subdict in p[key]:
            ts = subdict['timestamp']
            res_item[ts] = subdict
        res[key] = res_item
    return res


def convert_tcp(tcp_pose):
    """ Convert tcp into xyzrpy. """
    return np.concatenate((tcp_pose[:3], quat2euler(tcp_pose[3:]))).astype(np.float32)


def unify_joint(joint_info, cfg):
    """ Unify joint interface. """
    if cfg == 1 or cfg == 2:
        """ Flexiv: 7 joints, with vel """
        return np.array(joint_info[:7]).astype(np.float32), np.array(joint_info[7:14]).astype(np.float32)
    elif cfg == 3 or cfg == 4:
        """ UR: 6 joints, without vel """
        return np.concatenate((joint_info[:6], [0])).astype(np.float32), np.zeros(7).astype(np.float32)
    elif cfg == 5:
        """ Franka: 7 joints, with vel """
        return np.array(joint_info[:7]).astype(np.float32), np.array(joint_info[7:14]).astype(np.float32)
    elif cfg == 6 or cfg == 7:
        """ Kuka: 7 joints, without vel """
        return np.array(joint_info[:7]).astype(np.float32), np.zers(7).astype(np.float32)
    else:
        raise AttributeError('Invalid cfg.')


def fetch_tactile(tactiles, timestamp):
    """ Fetch the most relevent tactile information from the record. """
    idx = np.argmin(np.abs(tactiles[:, 97] - timestamp))
    return np.array(tactiles[idx, :96]).astype(np.int64)
