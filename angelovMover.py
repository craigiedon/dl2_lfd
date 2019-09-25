import sys, os

import robotcontroller as rc
import tf
import math
import numpy as np

import cupy, chainer, cv2

import insert_gear_policy as igp

# ROS
import rospy # Ros Itself
from geometry_msgs.msg import PoseStamped # Combined timestamp and position/quatern-rot full pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class GearInsertPolicy(object):
    """docstring for GearInsertPolicy
    """
    def __init__(self, pr2_left=None, pr2_right=None):
        super(GearInsertPolicy, self).__init__()

        if pr2_left is None:
            self.pr2_left = rc.PR2RobotController('left_arm', add_table=True)
        else:
            self.pr2_left = pr2_left

        if pr2_right is None:
            self.pr2_right = rc.PR2RobotController('right_arm', add_table=False)
        else:
            self.pr2_right = pr2_right

        self.pr2_left.remove_and_deattach_gears()

        self.current_action = 0
        self.last_img = None
        self.subscribe_to_data()
        self.load_model(gpu_id=0)
        self.prepare()


    def in_initiation_state(self):
        return True

    def terminate(self):
        # Check with another model if we are in a terminal state
        pass

    def load_model(self, filename='mymodel.model', gpu_id=0):
        # Testing policies
        filename='/mnt/7ac4c5b9-8c05-451f-9e6d-897daecb7442/gears/results/results_rmdn_new_subset60/InsertGearPolicyResNet_model_epoch_300.model'
        # Load the model
        self.model = igp.InsertGearPolicyResNet()
        self.model.load_model(filename)
        if gpu_id >= 0:
            self.model.to_gpu(gpu_id)
        print('Model loaded. And loaded to', gpu_id)

    def subscribe_to_data(self):
        self.left_command = rospy.Publisher('/l_arm_controller/command_filter',
                                            JointTrajectory, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/kinect2/qhd/image_color_rect',
                                         Image, self.img_callback)
        print('Subscribed to data.')

    def img_callback(self, im):
        print('New image msg!')
        self.last_img = self.bridge.imgmsg_to_cv2(im, "bgr8")


    def send_left_arm_goal(self, j_positions):
        jt = JointTrajectory()
        jt.header.stamp = rospy.Time.now()
        jt.joint_names = ['l_upper_arm_roll_joint', 'l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_forearm_roll_joint', 
                          'l_elbow_flex_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint']
        jtp = JointTrajectoryPoint()
        jtp.positions = list(j_positions)
        jtp.velocities = [0.0] * len(j_positions)
        jtp.accelerations = [0.0] * len(j_positions)
        jtp.time_from_start = rospy.Duration(0.45)
        jt.points.append(jtp)
        self.left_command.publish(jt)


    def prepare(self):
        self.pr2_right.move_to_p_rpy([0.576, -0.462, 0.910], [-1.765, 0.082, 1.170])

        right = self.pr2_right.get_current_pose()

        self.delta_pose = [0.1, 0.49, 0.05]
        right.position.x += self.delta_pose[0]
        right.position.y += self.delta_pose[1]
        right.position.z += self.delta_pose[2]

        plan = self.pr2_left.plan(right)
        success = self.pr2_left.move_to_p_rpy([right.position.x, right.position.y, right.position.z], 
                                         [math.pi/2., 0, -math.pi/2])

        print('Robot policy Prepared.')



    def act(self):
        if self.last_img is None:
            print('No image. Skipping')
            return

        # cv2.imshow('Input', self.last_img)
        # cv2.waitKey(100)
        img = self.model.prepare(self.last_img)
        output_j = self.model(cupy.asarray([img]))[0]
        output_j = chainer.cuda.to_cpu(output_j.data)
        print('output_j', output_j)

        self.send_left_arm_goal(output_j)


    def reset(self):
        #self.current_action = 0
        # TODO: go back to prepared state
        pass


if __name__ == '__main__':
    policy = GearInsertPolicy()

    r = rospy.Rate(0.5)

    while not rospy.is_shutdown():
        print('Check some things...')
        policy.act()
        r.sleep()

    print('Done.')