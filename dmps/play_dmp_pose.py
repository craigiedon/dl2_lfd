#!/usr/bin/env python
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from dmp import DMP, load_dmp
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


def move_to_pos_quat(group, pos, quat):
    pose_goal = gm.Pose(gm.Point(*pos), gm.Quaternion(*quat))

    group.set_pose_target(pose_goal)
    plan_success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    return plan_success


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

    move_to_pos_quat(l_group, left_start_pos, left_start_quat)
    move_to_pos_quat(r_group, right_start_pos, right_start_quat)


    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        if start_track.header is not None:
            y_r, _, _ = dmp.rollout()
            waypoints = []
            for y in range(y_r):
                # Convert the eulers to quaternions first...
                point = Point(*y[0:3])
                quat = Quaternion(tf.transformations.quaternion_from_euler(*y[3:]))
                pose = Pose(point, quat)
                waypoints.append(copy.deepcopy(pose))

            plan, _ = l_group.compute_cartesian_path(waypoints,0.01,0.0)

            l_group.execute(plan,wait=True)
            print("Finished executing DMP")
            break
        rate.sleep()


    def callback(self,msg):
        self.header = msg.header


if __name__ == '__main__':
    dmp = load_dmp("./saved_dmps/dmp-0-l_wrist_roll_link.npy")
    play_back(dmp)
