#!/usr/bin/env python

import pickle
from dmp import DMP
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import glob
from matplotlib import pyplot as plt

class Playback():
    def __init__(self):
        rospy.init_node('dmp_playback', anonymous=True)
        self.traj_pub = rospy.Publisher('motion_segment', JointTrajectory, queue_size=1)
        rospy.Subscriber('joint_states', JointState, self.callback)
        self.header = None

    def load_dmp(self, i):
        with open("dmp{}.npy".format(i), "r") as f:
            obj = pickle.load(f)
        return obj

    def callback(self, msg):
        self.header = msg.header

    def spin(self):
        rate = rospy.Rate(0.5)
        dmp_count = 1
        n_dmps = len(glob.glob('./*.npy'))
        while not rospy.is_shutdown():
            if self.header is not None:
                dmp = self.load_dmp(dmp_count)
                y_r, dy_r, _ = dmp.rollout()
                jt = JointTrajectory()
                jt.header = self.header
                jt.joint_names = dmp.joint_names
                jtp = JointTrajectoryPoint()

                for i in range(y_r.shape[0]):
                    jtp.positions = y_r[i, :].tolist()
                    jtp.velocities = dy_r[i,:].tolist()
                    jtp.time_from_start = rospy.Duration(1)
                    jt.points.append(jtp)
                self.traj_pub.publish(jt)
                print('Publishing dmp {}'.format(dmp_count))
                dmp_count += 1

                if dmp_count > n_dmps:
                    break
            rate.sleep()


if __name__ == "__main__":
    pb = Playback()
    pb.spin()
