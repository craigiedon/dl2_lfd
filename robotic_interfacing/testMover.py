import sys, time, collections
from collections import deque
import copy, math
import matplotlib.pyplot as plt
import rospy
import tf
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg as gm
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
from moveit_commander.conversions import pose_to_list
import numpy as np
from load_data import load_rgbd_demos, nn_input_to_imshow
from helper_funcs.transforms import get_trans, get_grayscale_trans
from helper_funcs.utils import load_json
import torch
from model import ImagePlusPoseNet, PosePlusStateNet
from mdn import approx_ml
from cv_bridge import CvBridge
import cv2
from scipy.spatial.transform import Rotation as R

def move_to_pos_rpy(group, pos, rpy):
    quat = tf.transformations.quaternion_from_euler(*rpy)
    return move_to_pos_quat(group, pos, quat)

def move_to_pos_quat(group, pos, quat):
    pose_goal = gm.Pose(gm.Point(*pos), gm.Quaternion(*quat))

    group.set_pose_target(pose_goal)
    plan_success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    return plan_success


class RobotStateCache(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_im = None
        self.depth_im = None
        self.l_pose = None
        self.r_pose = None


    def update_rgb_img(self, img_msg):
        self.rgb_im = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
    
    def update_depth_img(self, depth_msg):
        self.depth_im = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough") / 10000.0 * 255.0
        self.depth_im = np.expand_dims(self.depth_im, 2)
        # print("min", np.min(self.depth_im), "max", np.max(self.depth_im))

    def update_l_pose(self, pose_stamped):
        # print("Updating l pose!")
        pos = pose_stamped.pose.position
        quat = pose_stamped.pose.orientation
        rpy = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz') / np.pi
        self.l_pose = [pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]]


    def update_r_pose(self, pose_stamped):
        # print("Updating r pose!")
        pos = pose_stamped.pose.position
        quat = pose_stamped.pose.orientation
        rpy = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz') / np.pi
        self.r_pose = [pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]]


def main(model_path):
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]
    device = torch.device("cpu")
    rgb_trans = get_trans(im_params, distorted=False)
    # depth_trans = get_grayscale_trans(im_params)

    # train_set, train_loader = load_rgbd_demos(
    #     exp_config["demo_folder"],
    #     im_params["file_glob"],
    #     exp_config["batch_size"],
    #     "l_wrist_roll_link",
    #     rgb_trans,
    #     depth_trans,
    #     True,
    #     device,
    #     from_demo=0,
    #     to_demo=1)

    model = ImagePlusPoseNet((im_params["resize_height"], im_params["resize_width"]), 100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Imporant if you have dropout / batchnorm layers!

    state_cache = RobotStateCache()
    rgb_sub = rospy.Subscriber('/kinect2/qhd/image_color_rect', Image, state_cache.update_rgb_img)
    # depth_sub = rospy.Subscriber('/kinect2/qhd/image_depth_rect', Image, state_cache.update_depth_img)
    l_pose_sub = rospy.Subscriber('/l_wrist_roll_link_as_posestamped', gm.PoseStamped, state_cache.update_l_pose)
    r_pose_sub = rospy.Subscriber('/r_wrist_roll_link_as_posestamped', gm.PoseStamped, state_cache.update_r_pose)
    print("Subscribed")


    rospy.init_node("testMover")

    moveit_commander.roscpp_initialize(sys.argv)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    l_group = moveit_commander.MoveGroupCommander('left_arm')
    l_group.set_pose_reference_frame("base_link")

    r_group = moveit_commander.MoveGroupCommander('right_arm')
    r_group.set_pose_reference_frame("base_link")

    # print(l_group.get_end_effector_link())
    # print(l_group.get_pose_reference_frame())

    r = rospy.Rate(1.5)

    # left_start_pos = np.array([0.68368, 0.1201, 0.8733])
    # left_start_quat = np.array([0.49261, -0.5294, -0.4433, 0.529611])


    left_start_pos = np.array([6.88e-01, 1.62e-01, 8.38e-01] )# + (np.random.rand(3) - 0.5) * 0.1
    # left_start_quat = np.array([3.746527957264187553e-02, -1.138874327630349376e-02, -5.064174634791493990e-01, 8.613988634984809378e-01])
    left_start_rpy = np.array([0.0, 0.0, -np.pi / 2.0])

    
    right_start_pos = np.array([4.350030651697819328e-01, -3.619320769156886275e-01, 9.532789909342967993e-01]) #+ (np.random.rand(3) - 0.5) * 0.1
    # right_start_quat = np.array([7.092342993967834519e-02,-1.200804160961578410e-01,5.726577827747867389e-01, 8.078450498599533125e-01])
    # right_start_rpy = tf.transformations.euler_from_quaternion(right_start_quat)
    right_start_rpy = np.array([0.0, 0.0, np.pi / 2.0])


    move_to_pos_rpy(l_group, left_start_pos, left_start_rpy)
    move_to_pos_rpy(r_group, right_start_pos, right_start_rpy)

    rospy.sleep(3)

    # Closed Loop
    while not rospy.is_shutdown():
        print("Step")
        if state_cache.l_pose is not None and state_cache.r_pose is not None and state_cache.rgb_im is not None:
            with torch.no_grad():
                current_pose = torch.tensor(state_cache.l_pose, dtype=torch.float)
                goal_pose = torch.tensor(state_cache.r_pose, dtype=torch.float)
                print("Current: {}".format(current_pose))

                next_pose = model(rgb_trans(state_cache.rgb_im).unsqueeze(0), current_pose.unsqueeze(0))[0]
                # pis, stds, mus = model( current_pose.unsqueeze(0), goal_pose.unsqueeze(0))
                # next_pose = approx_ml(pis[0], stds[0], mus[0], num_samples=1000)

                print("Next {}".format(next_pose))
                next_position = next_pose[0:3].numpy().astype(np.float64)
                next_rpy = next_pose[3:].numpy().astype(np.float64) * np.pi
                # print("For sanity {}\n{}\n{}".format(next_pose[3:], next_pose[3:].numpy(), next_pose[3:].numpy().astype(np.float64)))
                # print("Next rpy {}".format(next_rpy))
                next_quat = R.from_euler("xyz", next_rpy).as_quat()

                move_to_pos_quat(l_group, next_position, next_quat)
        r.sleep()



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testMover.py <model_path>")
        sys.exit(0)
    
    model_path = sys.argv[1]
    main(model_path)
