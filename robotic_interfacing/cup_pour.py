#!/usr/bin/env python
import sys
from os.path import dirname, realpath
# sys.path.append(dirname(dirname(realpath(__file__))))

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

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

l_group = moveit_commander.MoveGroupCommander('left_arm')
l_group.set_pose_reference_frame("base_link")

r_group = moveit_commander.MoveGroupCommander('right_arm')
r_group.set_pose_reference_frame("base_link")


# Set start points for both arms

l_start_pos, l_start_rpy = np.array([0.632, 0.242, 0.877]), np.array([0.0, 0.0, 0.0])

r_start_pos, r_start_rpy = np.array([0.424, -0.39, 0.984]), np.array([0.0, 0.0, 0.0])

# Set "above  cup" motion
l_up_pos = l_start_pos + np.array([0.0, 0.0, 0.1])

# Set "over cup" motion
l_over_pos = np.array([r_start_pos[0], r_start_pos[1], l_up_pos[2]])

# Set "pour" motion
l_pour_pos, l_pour_rpy = l_over_pos + np.array([0.0, 0.0, 0.1]), np.array([0.0, 1.6, 0.0])

# Execute plan slowly!
waypoints = [
    toPose(l_start_pos, l_start_rpy),
    toPose(l_up_pos, l_start_rpy),
    toPose(l_over_pos, l_start_rpy),
    toPose(l_pour_pos, l_pour_rpy)
]

plan, _ = l_group.compute_cartesian_path(waypoints,0.01,0.0)
print(plan)

# record_publisher.publish()
# l_group.execute(plan,wait=True)
# l_group.stop()
# l_group.clear_pose_targets()
# record_publisher.publish()