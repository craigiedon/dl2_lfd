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

# Set up moveit stuff, and ensure global positioning is used...
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

l_group = moveit_commander.MoveGroupCommander('left_arm')
l_group.set_pose_reference_frame("base_link")

r_group = moveit_commander.MoveGroupCommander('right_arm')
r_group.set_pose_reference_frame("base_link")