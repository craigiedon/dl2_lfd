#! /usr/bin/env python
import cv2
import roslib
# import sensor_msgs
# import sys
import rospy
from std_msgs.msg import String, Empty
from pr2_msgs.msg import AccelerometerState
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge, CvBridgeError
# import tf
import numpy as np
# import os
from os import makedirs
from os.path import join
from datetime import datetime
import message_filters
import argparse

class DataGrabber:

    def __init__(self, image_topics, root_path):

        self.bridge = CvBridge()

        self.recording = False
        self.recordingFolder = None
        self.joint_names_recorded = False
        self.root_path = root_path
        self.image_topics = image_topics

        rospy.loginfo(['Image topics for recording: ', image_topics])

        rospy.Subscriber('/toggle_recording', Empty, self.toggle_recording)

        self.arm_names =  ['l', 'r']
        im_subs = [message_filters.Subscriber(image_topic, Image, queue_size=1) for image_topic in self.image_topics]
        eff_subs = [message_filters.Subscriber(arm + '_wrist_roll_link_as_posestamped', PoseStamped, queue_size=1) for arm in self.arm_names]
        acc_subs = [message_filters.Subscriber('accelerometer/{}_gripper_motor_throttled'.format(arm), AccelerometerState, queue_size=1) for arm in self.arm_names]
        joints_sub = message_filters.Subscriber('joint_states', JointState, queue_size=1)

        synched_sub = message_filters.ApproximateTimeSynchronizer(
                        im_subs + eff_subs + acc_subs + [joints_sub], 
                        queue_size=30, slop=0.05) # was 0.05
        synched_sub.registerCallback(self.demo_callback)

        self.save_rate_publish = rospy.Publisher('/rate_save_data_grabber',
                                             Empty, queue_size=1)

    
    def toggle_recording(self, data):
        if self.recording:
            print('Turning off recording for {}'.format(self.recordingFolder))
            self.recording = False
            self.recordingFolder = None
            # TODO: Should we scp the information off somewhere? 
            # TODO: Delete previous recordings from robot to save space?
        else:
            print('Turning on recording')
            self.recording = True
            self.recordingFolder = join(self.root_path, 
                                'demos/demo_{}'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
            makedirs(self.recordingFolder)
            # print('Current Directory:', os.getcwd())
            print('Folder made at {}'.format(self.recordingFolder))
            self.joint_names_recorded = False


    def demo_callback(self, *data):
        if not self.recording:
            return

        try:
            rospy.logdebug('RECORDING NEW DATA!!!')
            # Split datarecording
            ims = data[:len(self.image_topics)]
            effs = data[len(self.image_topics):len(self.image_topics) + len(self.arm_names)]
            accs = data[len(self.image_topics) + len(self.arm_names):len(self.image_topics) + 2*len(self.arm_names)]
            joint_state = data[-1]
            
            # Get timestamp
            im = ims[0]
            t_stamp = int(im.header.stamp.to_nsec())
            out_folder = self.recordingFolder # Make sure to safe folder, as it's async and can be removed mid execution.

            print(self.image_topics, len(data), len(ims), len(self.image_topics))
            # Save images
            for im, image_topic in zip(ims, self.image_topics):
                if 'depth' not in image_topic:
                    cv_image = self.bridge.imgmsg_to_cv2(im, 'bgr8')
                    image_name = image_topic.strip('/').replace('/', '_') + '_{}.jpg'
                    im_path = join(out_folder, image_name.format(t_stamp))
                    cv2.imwrite(im_path, cv_image)
                else:
                    rospy.logdebug('this is a depth image.')
                    # print(im)
                    cv_image = self.bridge.imgmsg_to_cv2(im, 'passthrough')/10000. * 255.
                    rospy.logdebug('Min ', np.min(cv_image), 'max: ', np.max(cv_image))
                    image_name = image_topic.strip('/').replace('/', '_') + '_{}.jpg'
                    im_path = join(out_folder, image_name.format(t_stamp))
                    cv2.imwrite(im_path, cv_image)
            
            # Record joint names
            if not self.joint_names_recorded:
                names_path = join(out_folder, 'joint_names_{}.txt'.format(t_stamp))
                np.savetxt(names_path, joint_state.name, fmt='%s')
                self.joint_names_recorded = True

            # Save joint states
            vel_path = join(out_folder, 'joint_vel_{}.txt'.format(t_stamp))
            position_path = join(out_folder, 'joint_position_{}.txt'.format(t_stamp))
            effort_path = join(out_folder, 'joint_effort_{}.txt'.format(t_stamp))

            np.savetxt(vel_path, joint_state.velocity)
            np.savetxt(position_path, joint_state.position)
            np.savetxt(effort_path, joint_state.effort)

            # Save eff pose
            rospy.logdebug(['End effectors: ', effs])
            for eff, arm in zip(effs, self.arm_names):
                eff_path = join(out_folder, '{}_wrist_roll_link_{}.txt'.format(arm, t_stamp)) 
                pose = [eff.pose.position.x, 
                        eff.pose.position.y, 
                        eff.pose.position.z, 
                        eff.pose.orientation.x, eff.pose.orientation.y,
                        eff.pose.orientation.z, eff.pose.orientation.w]
                np.savetxt(eff_path, pose)

            # Save accelerometer data
            rospy.logdebug(['AccelerometerState: ', accs])
            for acc, arm in zip(accs, self.arm_names):
                acc_path = join(out_folder, '{}_palm_max_acceleration_{}.txt'.format(arm, t_stamp)) 
                max_acc = [max(d.x for d in acc.samples), 
                           max(d.y for d in acc.samples),
                           max(d.z for d in acc.samples)]
                rospy.logdebug(max_acc)
                np.savetxt(acc_path, max_acc)

            self.save_rate_publish.publish()
            rospy.logdebug('Saving at time {}'.format(t_stamp))
        except CvBridgeError as e:
            rospy.logwarn('No transform available')

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imtopic', action='append', help='The ROS image topic path which the grabber will record') # e.g.  default=['/kinect2/sd/image_color_rect'], 
    parser.add_argument('--root_path', default='/home/michael/vive_ws/src/htc_vive_teleop_stuff/scripts', help='The root folder where to save the demo files.')
    args, unknown_args = parser.parse_known_args()

    rospy.init_node('image_grabber', anonymous=True)
    ic = DataGrabber(args.imtopic, args.root_path)
    print('Recording Node Online')
    print("Recording to: {}".format(args.root_path))

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down Recording Node')

