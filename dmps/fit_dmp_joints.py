#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from dmp import DMP, imitate_path, plot_rollout
from moveit_msgs.msg import DisplayTrajectory
import matplotlib.pyplot as plt
import pickle


class Imitate():
    def __init__(self, arm_prefix):
        rospy.init_node('dmp_fitter', anonymous=True)
        rospy.Subscriber('joint_states', JointState, self.callback)

        self.joint_names = [arm_prefix + joint for joint in
                            ["_shoulder_pan_joint",
                             "_shoulder_lift_joint",
                             "_upper_arm_roll_joint",
                             "_elbow_flex_joint",
                             "forearm_roll_joint",
                             "_wrist_flex_joint",
                             "_wrist_roll_joint"]]

        self.js_prev = None 
        self.js_current = None
        self.traj_pub = rospy.Publisher('motion_segment', JointTrajectory, queue_size=1)
        self.moving = False
        self.traj = []
        self.dmp_count = 0

    def callback(self, msg):
        if self.js_prev is not None:
            self.js_prev = msg
            self.js_current = msg
        else:
            self.js_prev = self.js_current
            self.js_current = msg

        # If your joints have changed by more than a small threshold amount, then you are moving! Start recording joint angles
        if np.sum(np.abs(np.array(self.js_prev.position) - np.array(self.js_current.position))) > 0.01:
            print('Moving')
            j_list = [msg.position[msg.name.index(j)] for j in self.joint_names]
            self.traj.append(j_list)

        elif len(self.traj) > 0:
            # Fit DMP to motion segment
            path = np.array(self.traj)
            dmp = DMP(path[0, :], path[-1, :], num_basis_funcs=500, dt=0.01, d=path.shape[1], jnames=self.joint_names)
            _, weights = imitate_path(path + 1e-5 * np.random.randn(path.shape[0], path.shape[1]), dmp)
            print("Learned weights: {}".format(weights))
            dmp.weights = weights

            self.dmp_count += 1

            # Save it with pickle
            with open("./dmp{}.npy".format(self.dmp_count), "w") as f:
                pickle.dump(dmp, f, pickle.HIGHEST_PROTOCOL)

            # Plot the path imitated by the learned dmp (rolled out) against the actual trajectory path...
            y_r, dy_r, _ = dmp.rollout()
            plot_rollout(y_r, path)

            jt = build_jt(msg.header, self.joint_names, y_r, dy_r)
            self.traj_pub.publish(jt)
            self.traj = []
        else:
            print("Stationary")

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    plt.ion()
    im = Imitate('l')
    im.spin()


def build_jt(header, joint_names, y_r, dy_r):
    jt = JointTrajectory()
    jt.header = header
    jt.joint_names = joint_names

    jtp = JointTrajectoryPoint()
    for i in range(y_r.shape[0]):
        jtp.positions = y_r[i, :].tolist()
        jtp.velocities = dy_r[i, :].tolist()
        jtp.time_from_start = rospy.Duration(1.0)
        jt.points.append(jtp)
    
    return jt
