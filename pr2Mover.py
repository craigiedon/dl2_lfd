import sys, os

import tf
import math
import numpy as np

import cv2
import geometry_msgs

from model import load_model
from load_data import cv_to_nn_input, nn_input_to_imshow, load_demos, unnorm_pose, wrap_pose
from torchvision.transforms import Compose
from helper_funcs.utils import load_json, byteify
from helper_funcs.transforms import Crop, Resize

# ROS
import rospy # Ros Itself
from geometry_msgs.msg import PoseStamped # Combined timestamp and position/quatern-rot full pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState
import moveit_commander

import torch

IMAGE_SIZE = 128


def send_arm_goal(j_pos, arm_publisher, joint_names):

    jtps = [JointTrajectoryPoint(
        positions=j_pos,
        velocities=[0.0] * len(j_pos),
        accelerations=[0.0] * len(j_pos),
        time_from_start=rospy.Duration(1.0))]

    jt = JointTrajectory(joint_names=joint_names,points=jtps)
    jt.header.stamp = rospy.Time.now()

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

        next_pos_normed = model(torch.unsqueeze(torch_im, 0), torch.unsqueeze(torch_pos, 0))[0]
        next_pos = unnorm_pose(next_pos_normed).tolist()

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


    # (img, pos_normed), next_pos_normed = data_set[last_id]

    # pos = unnorm_pose(pos_normed)
    # next_pos = unnorm_pose(next_pos_normed)

    # print("First data pos")
    # print(pos)

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


def sanity_check():
    rospy.init_node('pr2_mover', anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)
    r = rospy.Rate(0.5)

    exp_config = byteify(load_json("config/experiment_config.json"))

    pr2_left = setup_moveit_group("left_arm")
    pr2_right = setup_moveit_group("right_arm")

    left_command = rospy.Publisher('/l_arm_controller/command', JointTrajectory)
    right_command = rospy.Publisher('/r_arm_controller/command', JointTrajectory)


    right_joints = [
        "r_shoulder_pan_joint",
        "r_shoulder_lift_joint",
        "r_upper_arm_roll_joint",
        "r_elbow_flex_joint",
        "r_forearm_roll_joint",
        "r_wrist_flex_joint",
        "r_wrist_roll_joint"]


    device = torch.device("cpu")
    im_params = exp_config["image_config"]
    im_trans = Compose([
        Crop(im_params["crop_top"], im_params["crop_left"],
            im_params["crop_height"], im_params["crop_width"]),
        Resize(im_params["resize_height"], im_params["resize_width"])])

    model = load_model("saved_models/gear_unconstrained_9.pt", device, im_params["resize_height"], im_params["resize_width"], exp_config["nn_joint_names"])

    state_cache = RobotStateCache(exp_config["nn_joint_names"])
    image_sub = rospy.Subscriber('/kinect2/qhd/image_color_rect', Image, state_cache.update_img)
    joints_sub = rospy.Subscriber('/joint_states', JointState, state_cache.update_joint_pos)
    print("Subscribed")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, train_loader = load_demos(
        exp_config["demo_folder"],
        im_params["file_glob"],
        exp_config["batch_size"],
        exp_config["nn_joint_names"],
        im_trans,
        False,
        device,
        from_demo=0,
        to_demo=1)

    print("Starting pos: {}".format(unnorm_pose(train_set[0][0][1])))

    left_start = [ 0.2242,  0.3300,  1.4105, -0.8090,  0.1163, -0.9732,  0.2731]
    right_start = [-0.7920,  0.5493, -1.1246, -1.0972, -0.8366, -1.0461, -0.0410]

    send_arm_goal(left_start, left_command, exp_config["nn_joint_names"])
    send_arm_goal(right_start, right_command, right_joints)



    # ### Setup robot in initial pose using the moveit controllers
    # right_success = move_to_position_rotation(pr2_right, [0.576, -0.462, 0.910], [-1.765, 0.082, 1.170])

    # right = pr2_right.get_current_pose().pose
    # left_pos = [right.position.x + 0.1, right.position.y + 0.49, right.position.z + 0.05]
    # left_rpy = [math.pi/2.0, 0.0, -math.pi/2.0]


    # print("Desired left: pos {}, rot {}".format(left_pos, left_rpy))
    # left_success = move_to_position_rotation(pr2_left, left_pos, left_rpy)
    # print("Left Success message: {}".format(left_success))
    # print("Right Success message: {}".format(right_success))


    print('Robot policy Prepared.')
    model.eval()
    while not rospy.is_shutdown():
        print("step")
        act(state_cache, model, left_command, exp_config["nn_joint_names"])
        r.sleep()

    print('Done.')

# def main():
#     rospy.init_node('pr2_mover', anonymous=True)
#     moveit_commander.roscpp_initialize(sys.argv)
#     r = rospy.Rate(0.5)

#     pr2_left = setup_moveit_group("left_arm")
#     pr2_right = setup_moveit_group("right_arm")

#     left_command = rospy.Publisher('/l_arm_controller/command', JointTrajectory, queue_size=1)


#     state_cache = RobotStateCache(joint_names)
#     image_sub = rospy.Subscriber('/kinect2/qhd/image_color_rect', Image, state_cache.update_img)
#     joints_sub = rospy.Subscriber('/joint_states', JointState, state_cache.update_joint_pos)
#     print('Subscribed to data.')

#     """
#     device = torch.device("cuda")
#     model = load_model(model_path, device, height, width, joint_names)
#     """


#     ### Setup robot in initial pose using the moveit controllers
#     move_to_position_rotation(pr2_right, [0.576, -0.462, 0.910], [-1.765, 0.082, 1.170])

#     right = pr2_right.get_current_pose().pose
#     left_pos = [right.position.x + 0.1, right.position.y + 0.49, right.position.z + 0.05]
#     left_rpy = [math.pi/2.0, 0.0, -math.pi/2.0]
#     success = move_to_position_rotation(pr2_left, left_pos, left_rpy)


#     print('Robot policy Prepared.')
#     while not rospy.is_shutdown():
#         print('Check some things...')
#         act(state_cache, None, left_command, 0.05)
#         r.sleep()

#     print('Done.')


if __name__ == '__main__':
    sanity_check()
