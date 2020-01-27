#!/usr/bin/env python
import sys
from os.path import dirname, realpath

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
from std_msgs.msg import Float64, Empty, Header
import copy
import tf

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

def toPose(pos, rpy):
    point = Point(*pos)
    quat = Quaternion(*tf.transformations.quaternion_from_euler(*rpy))
    return Pose(point, quat)

# Set up moveit stuff, and ensure global positioning is used...
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

l_group = moveit_commander.MoveGroupCommander('left_arm')
l_group.set_pose_reference_frame("base_link")

r_group = moveit_commander.MoveGroupCommander('right_arm')
r_group.set_pose_reference_frame("base_link")

# l_start_pos, l_start_rpy = np.array([0.632, 0.4, 0.80]), np.array([0.0, 0.0, -np.pi / 2.0])
l_start_pos, l_start_rpy = np.array([0.2, 0.7, 0.8]), np.array([0.0, 0.0, -np.pi / 4.0])
r_start_pos, r_start_rpy = np.array([0.3, -0.4, 0.784]), np.array([0.0, 0.0, np.pi / 2.0])

l_goal_pos, l_goal_rpy = np.array([0.5, 0.4, 0.8]), np.array([0.0, 0.0, -np.pi / 4.0])

# move_to_pos_rpy(l_group, l_start_pos, l_start_rpy)
move_to_pos_rpy(l_group, l_goal_pos, l_goal_rpy)
move_to_pos_rpy(r_group, r_start_pos, r_start_rpy)

move_to_pos_rpy(l_group, l_goal_pos, l_goal_rpy)