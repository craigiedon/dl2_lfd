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
from model import ZhangNet
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
        self.pose = None


    def is_empty(self):
        return self.rgb_im is None or self.depth_im is None or self.pose is None

    def update_rgb_img(self, img_msg):
        self.rgb_im = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
    
    def update_depth_img(self, depth_msg):
        self.depth_im = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough") / 10000.0 * 255.0
        self.depth_im = np.expand_dims(self.depth_im, 2)
        # print("min", np.min(self.depth_im), "max", np.max(self.depth_im))

    
    def update_pose_hist(self, pose_stamped):
        pos = pose_stamped.pose.position
        quat = pose_stamped.pose.orientation
        rpy = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz') / np.pi

        self.pose = [pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]]

        self.last_quat = [quat.x, quat.y, quat.z, quat.w]
        self.last_pos = [pos.x, pos.y, pos.z]
        # right_start_pos = np.array([0.576, -0.456, 0.86])
        # left_start_pos = right_start_pos + np.array([0.1, 0.49, 0.05])

        # print("Start pos: ", left_start_pos)
        # print("Pos received: ", [pos.x, pos.y, pos.z])
        # print("Quat received: ", [quat.x, quat.y, quat.z, quat.w])


def main(model_path):
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]
    device = torch.device("cpu")
    rgb_trans = get_trans(im_params, distorted=False)
    depth_trans = get_grayscale_trans(im_params)

    train_set, train_loader = load_rgbd_demos(
        exp_config["demo_folder"],
        im_params["file_glob"],
        exp_config["batch_size"],
        "l_wrist_roll_link",
        rgb_trans,
        depth_trans,
        True,
        device,
        from_demo=0,
        to_demo=1)

    model = ZhangNet(im_params["resize_height"], im_params["resize_width"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    state_cache = RobotStateCache()
    rgb_sub = rospy.Subscriber('/kinect2/qhd/image_color_rect', Image, state_cache.update_rgb_img)
    depth_sub = rospy.Subscriber('/kinect2/qhd/image_depth_rect', Image, state_cache.update_depth_img)
    joints_sub = rospy.Subscriber('/l_wrist_roll_link_as_posestamped', gm.PoseStamped, state_cache.update_pose_hist)
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

    r = rospy.Rate(3)
    right_start_pos = np.array([0.576, -0.456, 0.86])
    right_start_quat = np.array([-0.656, -0.407, 0.379, 0.510])

    left_start_pos = right_start_pos + np.array([0.1, 0.49, 0.05])
    left_start_rpy = np.array([np.pi/2.0, 0.0, -np.pi/2.0])

    move_to_pos_rpy(l_group, left_start_pos, left_start_rpy)
    move_to_pos_quat(r_group, right_start_pos, right_start_quat)

    # Open Loop: Just replicating the training run
    # i = 0
    # while not rospy.is_shutdown():
    #     l_goal = train_set[i][3].numpy().astype(np.float64)
    #     l_pos = l_goal[0:3]
    #     l_rpy = l_goal[3:] * np.pi

    #     l_quat = R.from_euler("xyz", l_rpy).as_quat()

    #     print("pos", l_pos)
    #     print("rpy", l_rpy)
    #     print("quat", l_quat)

    #     move_to_pos_quat(l_group, l_pos, l_quat)
        
    #     if i + 5 < len(train_set) - 1:
    #         i += 5

    #     r.sleep()

    # Closed Loop
    pose_history = deque(maxlen=5)
    while not rospy.is_shutdown():
        print("Step")
        if not state_cache.is_empty():
            torch_rgb = rgb_trans(state_cache.rgb_im)
            torch_depth = depth_trans(state_cache.depth_im)
            pose_history.append(state_cache.pose)
            # plt.show()
            # print("Depth im:", torch_depth.squeeze())
            # color_path = "demos/gear_good/demo_2019_06_26_15_58_59/kinect2_qhd_image_color_rect_1561561145828572837.jpg"
            # sanity_im = rgb_trans(cv2.imread(color_path))

            # plt.subplot(1,2, 1)
            # plt.imshow(nn_input_to_imshow(torch_rgb))
            # plt.subplot(1,2,2)
            # plt.imshow(nn_input_to_imshow(sanity_im))
            # plt.show()


            if len(pose_history) == 5:
                # Ready to go!
                torch_pose_hist = torch.tensor(pose_history)
                # print("Pose History: ", torch_pose_hist)
                with torch.no_grad():
                    # current_pose = torch_pose_hist[4]
                    # current_position = current_pose[0:3].numpy().astype(np.float64)
                    # current_rpy = current_pose[3:].numpy().astype(np.float64)
                    # current_quat = R.from_euler('xyz', current_rpy * np.pi).as_quat()


                    # print("Current pos: ", current_position)
                    move_to_pos_quat(l_group, state_cache.last_pos, state_cache.last_quat)

                    next_pose, _ = model(torch_rgb.unsqueeze(0), torch_depth.unsqueeze(0), torch_pose_hist.unsqueeze(0))
                    next_position = next_pose[0][0:3].numpy().astype(np.float64)
                    next_rpy = next_pose[0][3:].numpy().astype(np.float64) * np.pi
                    next_quat = R.from_euler("xyz", next_rpy).as_quat()

                    # print("Pred Pos: {}, Quat: {}".format(next_position, next_quat))
                    move_to_pos_quat(l_group, next_position, next_quat)
                    # move_to_pos_quat(l_group, next_position, current_quat)
        r.sleep()



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testMover.py <model_path>")
        sys.exit(0)
    
    model_path = sys.argv[1]
    main(model_path)