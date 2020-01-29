#!/usr/bin/env python
import sys
from os.path import dirname, realpath

from dmps.dmp import load_dmp_demos, DMP
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

import torch
from os.path import join
from nns.dmp_nn import DMPNN
from helper_funcs.conversions import np_to_pgpu
from loadAndChartDMP import chart_given_rollout


def move_to_pos_rpy(group, pos, rpy):
    pos = pos.astype(float)
    rpy = rpy.astype(float)
    quat = tf.transformations.quaternion_from_euler(*rpy)
    return move_to_pos_quat(group, pos, quat)

def move_to_pos_quat(group, pos, quat):
    pos = pos.astype(float)
    quat = quat.astype(float)

    pose_goal = Pose(Point(*pos), Quaternion(*quat))

    group.set_pose_target(pose_goal)
    plan_success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    return plan_success

def rollout_to_waypoints(y_r):
    y_r = y_r.astype(float)
    waypoints = []
    for y in y_r:
        # Convert the eulers to quaternions first...
        point = Point(*y[0:3])
        quat = Quaternion(*tf.transformations.quaternion_from_euler(*y[3:]))
        pose = Pose(point, quat)
        waypoints.append(copy.deepcopy(pose))
    return waypoints


def play_back(weights, y_start, y_goal, right_start, orig_rollout):
    rospy.init_node('dmp_playback', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    l_group = moveit_commander.MoveGroupCommander('left_arm')
    l_group.set_pose_reference_frame("base_link")

    r_group = moveit_commander.MoveGroupCommander('right_arm')
    r_group.set_pose_reference_frame("base_link")

    basis_fs = 30
    dt = 0.01

    dmp = DMP(basis_fs, dt, 6)

    print("Ys just before rollout:")
    print("Y_start", y_start)
    print("Y_goal", y_goal)
    rollout = dmp.rollout_torch(y_start.unsqueeze(0), y_goal.unsqueeze(0), weights, 1.0)[0][0].detach().cpu()

    # chart_given_rollout(orig_rollout, rollout)

    # You will need to move with respect to rpy, but remember its normed, so undo the norm first!
    rollout[:, 3:] = rollout[:, 3:] * np.pi
    y_start[3:] = y_start[3:] * np.pi
    y_goal[3:] = y_goal[3:] * np.pi
    right_start[3:] = right_start[3:] * np.pi


    rate = rospy.Rate(0.5)

    # First, move to the start position (left and right arms)
    print("Moving to start positions")
    move_to_pos_rpy(l_group, y_start[0:3].cpu().numpy(), y_start[3:].cpu().numpy())
    move_to_pos_rpy(r_group, right_start[0:3].cpu().numpy(), right_start[3:].cpu().numpy())
    print("Finished moving to start positions")

    print("Rolling out waypoints")
    waypoints = rollout_to_waypoints(rollout.cpu().numpy())
    # orig_rollout[:, 3:] = orig_rollout[:, 3:] * np.pi
    # waypoints = rollout_to_waypoints(orig_rollout.cpu().numpy())
    print("Executing")

    plan, _ = l_group.compute_cartesian_path(waypoints, 0.01, 0.0)
    l_group.execute(plan, wait=True)
    l_group.stop()
    l_group.clear_pose_targets()
    
    print("Finished Executing DMP")
    rospy.sleep(3)

    # print("Returning to start pos")
    # move_to_pos_rpy(l_group, y_start[0:3].cpu().numpy(), y_start[3:].cpu().numpy())
    # move_to_pos_rpy(r_group, right_start[0:3].cpu().numpy(), right_start[3:].cpu().numpy())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python play_constrained_dmp.py <data-folder> <model-folder>")
        sys.exit(0)

    data_folder = sys.argv[1]
    model_folder = sys.argv[2]

    start_state = np_to_pgpu(load_dmp_demos(data_folder)[0])[0]
    pose_hists = np_to_pgpu(load_dmp_demos(data_folder)[1])[0]

    y_start = start_state[0]
    y_goal = start_state[-1]
    right_start = start_state[1]

    basis_fs = 30
    dt = 0.01
    model = DMPNN(start_state.numel(), 1024, start_state.shape[1], basis_fs).cuda()
    model.load_state_dict(torch.load(join(model_folder, "learned_model_epoch_final.pt")))
    model.eval()
    print("Start states: ", start_state)
    weights = model(start_state.unsqueeze(0))

    play_back(weights, y_start, y_goal, right_start, pose_hists)
