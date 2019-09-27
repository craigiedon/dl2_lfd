#!/usr/bin/env python
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from dmp import DMP, load_dmp
from math import ceil
import rospy
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import glob
from matplotlib import pyplot as plt
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose, Point, Quaternion
import copy
import tf

class StartTracker():
    def __init__(self):
        self.header = None

    def cache_header(self, msg):
        self.header = msg.header


def move_to_pos_rpy(group, pos, rpy):
    quat = tf.transformations.quaternion_from_euler(*rpy)
    return move_to_pos_quat(group, pos, quat)

def move_to_pos_quat(group, pos, quat):
    pose_goal = Pose(Point(*pos), Quaternion(*quat))

    group.set_pose_target(pose_goal)
    plan_success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    return plan_success

def rollout_to_waypoints(y_r):
    waypoints = []
    for y in y_r:
        # Convert the eulers to quaternions first...
        point = Point(*y[0:3])
        quat = Quaternion(*tf.transformations.quaternion_from_euler(*y[3:]))
        pose = Pose(point, quat)
        waypoints.append(copy.deepcopy(pose))
    return waypoints


def play_back(dmp):
    rospy.init_node('dmp_playback',anonymous=True)

    start_track = StartTracker()

    rospy.Subscriber('joint_states',JointState, start_track.cache_header)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    l_group = moveit_commander.MoveGroupCommander('left_arm')
    l_group.set_pose_reference_frame("base_link")

    r_group = moveit_commander.MoveGroupCommander('right_arm')
    r_group.set_pose_reference_frame("base_link")

    ## Setup initial starting pos here first

    left_start_pos = np.array([0.68368, 0.1201, 0.8733])
    left_start_quat = np.array([0.49261, -0.5294, -0.4433, 0.529611])

    right_start_pos = np.array([0.57622, -0.45568, 0.85959])
    right_start_quat = np.array([-0.6559, -0.4068, 0.3795, 0.5101])
    right_start_rpy = tf.transformations.euler_from_quaternion(right_start_quat)

    move_to_pos_quat(l_group, left_start_pos, left_start_quat)
    move_to_pos_quat(r_group, right_start_pos, right_start_quat)


    rate = rospy.Rate(0.5)

    original_start = dmp.y0
    original_goal = dmp.goal


    while not rospy.is_shutdown():
        if start_track.header is not None:
            print("Moving back to original position")
            move_to_pos_rpy(r_group, right_start_pos, right_start_rpy)
            move_to_pos_rpy(l_group, original_start[0:3], original_start[3:])
            rospy.sleep(1)

            print("Offsetting start and goal by a small amount")
            new_start_pos = original_start[0:3] + (np.random.rand(3) - 0.5) * np.array([0.005, 0.05, 0.05])
            new_start_rpy = original_start[3:] + (np.random.rand(3) - 0.5) * np.pi * 0.05

            # goal_offset_pos = (np.random.rand(3) - 0.5) * np.array([0.005, 0.05, 0.05])
            # # goal_offset_rpy = (np.random.rand(3) - 0.5) * np.pi * 0.05
            # goal_offset = np.concatenate((goal_offset_pos, goal_offset_rpy))

            new_goal_pos = original_goal[0:3] # + goal_offset_pos
            new_goal_rpy = original_goal[3:]  # + goal_offset_rpy

            move_to_pos_rpy(l_group, new_start_pos, new_start_rpy)
            # move_to_pos_rpy(r_group, right_start_pos + goal_offset_pos, right_start_rpy + goal_offset_rpy)
            rospy.sleep(1)

            orig_rollout = dmp.rollout(tau=1.0)[0]
            dmp.y0 = np.concatenate((new_start_pos, new_start_rpy))
            dmp.goal = np.concatenate((new_goal_pos, new_goal_rpy))


            print("Rolling out from set start / end")
            y_r = dmp.rollout(tau=1.0)[0]
            waypoints = rollout_to_waypoints(y_r)
            plot_orig_v_offset(orig_rollout, y_r, dmp.y0, dmp.goal)

            print("Executing...")
            reset_pos = dmp.goal[0:3] + np.array([0.0, 0.0, 0.15])
            reset_quat = tf.transformations.quaternion_from_euler(*dmp.goal[3:])

            plan, _ = l_group.compute_cartesian_path(waypoints,0.01,0.0)
            # l_group.execute(plan,wait=True)
            l_group.stop()
            l_group.clear_pose_targets()
            print("Finished executing DMP")
            rospy.sleep(2)
            print("Unhooking the gear from the peg")
            move_to_pos_quat(l_group, reset_pos, reset_quat)
            break
        rate.sleep()


    def callback(self,msg):
        self.header = msg.header



def plot_orig_v_offset(original_rollout, offset_rollout, offset_start, offset_goal):
    dims = original_rollout.shape[1]
    for d in range(dims):
        plt.subplot(2,ceil(dims / 2),d + 1)
        dmp_timescale = np.linspace(0, 1, original_rollout.shape[0])
        plt.plot(dmp_timescale, original_rollout[:, d], label="Original")

        raw_timescale = np.linspace(0, 1, offset_rollout.shape[0])
        plt.plot(raw_timescale, offset_rollout[:, d], label="Raw")
        plt.plot(0, [offset_start[d]], marker='o', markersize=3, color="red")
        plt.plot(1, [offset_goal[d]], marker='o', markersize=3, color="red")
        plt.legend()
    plt.show()

if __name__ == '__main__':
    dmp = load_dmp("./saved_dmps/dmp-0-l_wrist_roll_link.npy")
    play_back(dmp)
