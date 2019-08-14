import sys, os

import tf
import math
import numpy as np

import cv2
import geometry_msgs

from model import load_model
from load_data import cv_to_nn_input

# ROS
import rospy # Ros Itself
from geometry_msgs.msg import PoseStamped # Combined timestamp and position/quatern-rot full pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState
import moveit_commander

import torch

IMAGE_SIZE = 128


def send_left_arm_goal(j_pos, arm_publisher):
    joint_names = ['l_upper_arm_roll_joint',
                   'l_shoulder_pan_joint',
                   'l_shoulder_lift_joint',
                   'l_forearm_roll_joint',
                   'l_elbow_flex_joint',
                   'l_wrist_flex_joint',
                   'l_wrist_roll_joint']

    jtps = [JointTrajectoryPoint(
        positions=j_pos,
        velocities=[0.0] * len(j_pos),
        accelerations=[0.0] * len(j_pos),
        time_from_start=rospy.Duration(1.0 / 20.0))]

    jt = JointTrajectory(joint_names=joint_names,points=jtps)
    jt.header.stamp = rospy.Time.now()

    arm_publisher.publish(jt)


def act(last_img, last_pos, model, arm_publisher, vel_dt):
    if last_img is None:
        print('No image. Skipping')
        return

    torch_im = last_img.to(torch.device("cuda"))
    torch_pos = torch.FloatTensor(last_pos, device="cuda") ## TODO: Convert to torch. Look it up

    # cv2.imshow('Input', self.last_img)
    # cv2.waitKey(100)
    output_vels = model(torch_im, last_pos).tolist()
    print('output velocities', output_vels)

    new_pos = (torch_pos + output_vels * vel_dt).tolist()

    send_left_arm_goal(new_pos, arm_publisher)

def setup_moveit_group(group_name):
    moveit_commander.roscpp_initialize(sys.argv)

    group = moveit_commander.MoveGroupCommander(group_name)
    group.set_goal_tolerance(0.001)
    group.set_planning_time(2.)
    group.set_max_velocity_scaling_factor(0.5)
    group.set_max_acceleration_scaling_factor(0.5)

    return group


def move_to_position_rotation(move_group, position, rpy):
    desired_pose = geometry_msgs.msg.Pose(position, rpy)
    success = move_group.go(desired_pose, wait=True)
    return success

class RobotStateCache(object):
    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.bridge = CvBridge()


    def update_img(self, img_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
        x, y, width, height = 200, 120, 590, 460 
        cropped_im = img[y:y + height, x:x + width]
        resized_im = cv2.resize(raw, (IMAGE_SIZE, IMAGE_SIZE))
        self.last_img =  cv_to_nn_input(resized_im)
    

    def update_joint_pos(self, joint_state):
        j_pos = joint_state.position
        all_names = joint_state.name
        self.last_joint_pos = [j_pos[all_names.index(jn)] for jn in self.joint_names]


def main():
    rospy.init_node('pr2_mover', anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)
    r = rospy.Rate(0.5)

    pr2_left = setup_moveit_group("left_arm")
    # pr2_right = setup_moveit_group("right_arm")

    left_command = rospy.Publisher('/l_arm_controller/command_filter', JointTrajectory, queue_size=1)

    joint_names = ['l_upper_arm_roll_joint',
                   'l_shoulder_pan_joint',
                   'l_shoulder_lift_joint',
                   'l_forearm_roll_joint',
                   'l_elbow_flex_joint',
                   'l_wrist_flex_joint',
                   'l_wrist_roll_joint']

    state_cache = RobotStateCache(joint_names)
    image_sub = rospy.Subscriber('/kinect2/qhd/image_color_rect', Image, state_cache.update_img)
    joints_sub = rospy.Subscriber('/kinect2/qhd/joint_states', JointState, state_cache.update_joint_pos)
    print('Subscribed to data.')

    device = torch.device("cuda")
    model = load_model(model_path, device, height, width, joint_names)


    ### Setup robot in initial pose using the moveit controllers
    move_to_position_rotation(pr2_right, [0.576, -0.462, 0.910], [-1.765, 0.082, 1.170])

    right = pr2_right.get_current_pose().pose
    left_pos = [right.position.x + 0.1, right.position.y + 0.49, right.position.z + 0.05]
    left_rpy = [math.pi/2.0, 0.0, -math.pi/2.0]
    success = move_to_position_rotation(pr2_left, left_pos, left_rpy)


    print('Robot policy Prepared.')
    while not rospy.is_shutdown():
        print('Check some things...')
        # act(state_cache.last_img, state_cache.last_joint_pos, model, left_command, 0.05)
        r.sleep()

    print('Done.')


if __name__ == '__main__':
    main()
