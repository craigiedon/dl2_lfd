import sys, os

import tf
import math
import numpy as np

import cv2
import geometry_msgs

from model import load_model, ImageOnlyNet
from load_data import cv_to_nn_input, nn_input_to_imshow, load_demos, unnorm_pose, wrap_pose
from torchvision.transforms import Compose, Normalize
from helper_funcs.utils import load_json, byteify
from frozenResnetTrainer import ResnetJointPredictor
# from helper_funcs.transforms import get_trans
# from autoencoder import EncodeDecodePredictor

# ROS
import rospy # Ros Itself
from geometry_msgs.msg import PoseStamped # Combined timestamp and position/quatern-rot full pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState
import moveit_commander

import torch

IMAGE_SIZE = 224


def send_arm_goal(j_pos, arm_publisher, joint_names):

    jtps = [JointTrajectoryPoint(
        positions=j_pos,
        velocities=[0.0] * len(j_pos),
        accelerations=[0.0] * len(j_pos),
        time_from_start=rospy.Duration(2))]

    jt = JointTrajectory(joint_names=joint_names,points=jtps)
    jt.header.stamp = rospy.Time.now()

    print("Sending arm goal...")

    arm_publisher.publish(jt)


def act(last_state, model, arm_publisher, joint_names):
    if last_state.image is None or last_state.joint_pos is None:
        print('No image / position. Skipping')
        return

    with torch.no_grad():
        torch_im = torch.from_numpy(last_state.image.transpose(2, 0, 1)).to(dtype=torch.float)
        torch_pos = wrap_pose(torch.FloatTensor(last_state.joint_pos)) # device="cuda")

        cv2.imshow('Input', cv2.cvtColor(nn_input_to_imshow(torch_im), cv2.COLOR_RGB2BGR))
        cv2.waitKey(100)

        # Additional step for resnet model
        normalizer = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        torch_im = normalizer(torch_im)

        next_pos_normed = model(torch.unsqueeze(torch_im, 0))[0].squeeze()
        next_pos = unnorm_pose(next_pos_normed).tolist()

        # print("Joint Names: {}".format(joint_names))
        # print("Curret Pos: {}".format(torch_pos))

        print("Current Pos Norm-Unnormed {}".format(unnorm_pose(torch_pos)))
        print("Next Pos Prediction: {}".format(next_pos))

    send_arm_goal(next_pos, arm_publisher, joint_names)

def open_loop_act(last_id, last_state, data_set, arm_publisher, joint_names):
    if last_id >= len(data_set):
        print("Gone through dataset, waiting here")
        return

    if last_state.image is None or last_state.joint_pos is None:
        print('No image / position. Skipping')
        return

    torch_im = torch.from_numpy(last_state.image.transpose(2, 0, 1)).to(dtype=torch.float)
    cv2.imshow('Input', cv2.cvtColor(nn_input_to_imshow(torch_im), cv2.COLOR_RGB2BGR))
    cv2.waitKey(100)

    (img, pos), next_pos = data_set[last_id]
    data_pos_torch = torch.FloatTensor(pos)
    print("Raw pos {}".format(data_pos_torch))


    send_arm_goal(data_pos_torch, arm_publisher, joint_names)

def setup_moveit_group(group_name):
    moveit_commander.roscpp_initialize(sys.argv)

    group = moveit_commander.MoveGroupCommander(group_name)
    group.set_goal_tolerance(0.001)
    group.set_planning_time(2.)
    group.set_max_velocity_scaling_factor(0.5)
    group.set_max_acceleration_scaling_factor(0.5)

    return group


def move_to_position_rotation(move_group, position, rpy):
    quat = tf.transformations.quaternion_from_euler(*rpy)
    desired_pose = geometry_msgs.msg.Pose(
        position=geometry_msgs.msg.Point(*position),
        orientation=geometry_msgs.msg.Quaternion(*quat))
    success = move_group.go(desired_pose, wait=True)
    return success

class RobotStateCache(object):
    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.bridge = CvBridge()
        self.image = None
        self.joint_pos = None


    def update_img(self, img_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
        x, y, width, height = 200, 120, 590, 460 
        cropped_im = img[y:y + height, x:x + width]
        resized_im = cv2.resize(cropped_im, (IMAGE_SIZE, IMAGE_SIZE))
        self.image = cv_to_nn_input(resized_im)
    

    def update_joint_pos(self, joint_state):
        j_pos = joint_state.position
        all_names = joint_state.name
        self.joint_pos = [j_pos[all_names.index(jn)] for jn in self.joint_names]


def reset_pose():
    rospy.init_node('pr2_mover', anonymous=True)
    r = rospy.Rate(1)

    exp_config = byteify(load_json("config/experiment_config.json"))


    left_command = rospy.Publisher('/l_arm_controller/command', JointTrajectory, queue_size=10)
    right_command = rospy.Publisher('/r_arm_controller/command', JointTrajectory, queue_size=10)


    right_joints = [
        "r_shoulder_pan_joint",
        "r_shoulder_lift_joint",
        "r_upper_arm_roll_joint",
        "r_elbow_flex_joint",
        "r_forearm_roll_joint",
        "r_wrist_flex_joint",
        "r_wrist_roll_joint"]

    left_start = [ 0.2242,  0.3300,  1.4105, -0.8090,  0.1163, -0.9732,  0.2731]
    right_start = [-0.7920,  0.5493, -1.1246, -1.0972, -0.8366, -1.0461, -0.0410]

    rospy.sleep(3)
    send_arm_goal(left_start, left_command, exp_config["nn_joint_names"])
    send_arm_goal(right_start, right_command, right_joints)
    rospy.sleep(3)

    print('Robot policy Prepared.')
    # model.eval()
    # while not rospy.is_shutdown():
    #     r.sleep()

    print('Done.')

if __name__ == '__main__':
    reset_pose()
