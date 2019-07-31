from urdf_parser_py.urdf import URDF
import numpy as np
from scipy.spatial.transform import Rotation as R
from load_data import get_pose_and_control
import math


class RobotModel():
    def __init__(self, urdf_path, base_frame, end_effector_frame, camera_model, constant_params=None):
        self.robot = URDF.from_xml_file(urdf_path)
        self.base_frame_ee = base_frame
        self.end_effector_frame = end_effector_frame
        self.camera_model = camera_model
        chain_names = self.robot.get_chain(base_frame, end_effector_frame, joints=True, links=False, fixed=False)
        self.joint_chain = [self.robot.joint_map[jn] for jn in chain_names]
        if constant_params is not None:
            self.constant_params = constant_params
        else:
            self.constant_params = {}


def limits_joint(joint_name, robot_model):
    joint = robot_model.robot.joint_map[joint_name]

    if joint.type == "continuous":
        return None
    return (joint.limit.lower, joint.limit.upper)

def joints_lower_limits(joint_names, rm):
    lower_lims = []
    for jn in joint_names:
        joint_lims = limits_joint(jn, rm)
        if joint_lims is not None:
            lower_lims.append(joint_lims[0])
        else:
            lower_lims.append(None)

    return lower_lims

def joints_upper_limits(joint_names, rm):
    upper_lims = []
    for jn in joint_names:
        joint_lims = limits_joint(jn, rm)
        if joint_lims is not None:
            upper_lims.append(joint_lims[1])
        else:
            upper_lims.append(None)

    return upper_lims

def joint_types(joint_names, rm):
    return {jn: rm.robot.joint_map[jn].type for jn in joint_names}


def rotation_matrix(angle, rotation_axis):
    """ Takes an angle (radians) and axis of rotation (3-D vector) and outputs a rotation matrix"""
    normed_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rot_vec = R.from_rotvec(normed_axis * angle)
    return R.as_dcm(rot_vec)


def transformation_matrix(joint, param):

    if joint.type == "revolute" or joint.type == "continuous": # Rotational Joint
        translation = np.array(joint.origin.xyz).reshape(-1, 1) # Translate along link
        rot = rotation_matrix(param, np.array(joint.axis)) # Rotate around axis
        rotation_translation = np.hstack((rot, translation))

        zero_row = np.zeros((1,4))

        T = np.vstack((rotation_translation, zero_row))
        T[3,3] = 1

        return T

    if joint.type == "prismatic": # Translational Joint
        origin = np.array(joint.origin.xyz)
        axis = np.array(joint.axis)
        translation = origin + param * axis # First translate up link, then move the amount along the axis

        rotation_translation = np.hstack((np.eye(3), translation.reshape(-1,1)))

        T = np.vstack((rotation_translation, np.zeros((1,4))))
        T[3,3] = 1
        return T

    # Otherwise the joint type must be the identity
    return np.eye(4)


def forward_kinematics_matrix(joint_params, joint_param_names, robot_model):
    T = np.eye(4)
    for joint in robot_model.joint_chain:
        if joint.name in joint_param_names:
            joint_idx = np.where(joint_param_names == joint.name)[0][0]
            Tn = transformation_matrix(joint, joint_params[joint_idx].item())
        elif joint.name in robot_model.constant_params:
            Tn = transformation_matrix(joint, robot_model.constant_params[joint.name])
        else:
            raise ValueError("{} not in param/constant map.\n Param Names: {}\n Constant Map: {}".format(
                joint.name, joint_param_names, robot_model.constant_params))

        T = np.matmul(T, Tn)
    return T


def forward_kinematics(joint_params, joint_param_names, robot_model):
    fk_matrix = forward_kinematics_matrix(joint_params, joint_param_names, robot_model)
    return np.matmul(fk_matrix, np.array([0.0,0.0,0.0,1.0]))[0:3]


if __name__ == "__main__":
    rm = RobotModel("config/pr2.xml", 'base_link', 'r_gripper_tool_frame', None)
    arm_joint_names = np.genfromtxt("config/arm_joint_names.txt", np.str)
    pose, vels = get_pose_and_control(["demos/reach_blue_cube/demo_2019_07_01_12_21_27/kinect2_qhd_image_color_rect_1561980089061739626.jpg",
    "demos/reach_blue_cube/demo_2019_07_01_12_21_27/kinect2_qhd_image_color_rect_1561980089061739626.jpg"], 0, arm_joint_names)
    print(pose)
    fk_mat = forward_kinematics_matrix(pose, arm_joint_names, joint_param_map, rm)
    print(fk_mat)
    print("FK Shape", fk_mat.shape)
    print("Pose shape", pose.shape)
    ee_pos = np.matmul(fk_mat, np.array([0,0,0,1]))
    print(ee_pos)
