from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
import tf as ros_tf
import geometry_msgs.msg
import rospy
import numpy as np

class RobotModel:
    
    def __init__(self,urdf_path,base_frame,ee_frame,camera_model):
        
        self.robot = URDF.from_xml_file(urdf_path)
        self.kdl_kin = KDLKinematics(self.robot,base_frame,ee_frame)
        self.dim = len(self.kdl_kin.get_joint_names())
        self.K = camera_model
        self.transform = ros_tf.TransformerROS(True, rospy.Duration(100))
        self.update_transforms()
        
        
    def update_transforms(self):
        
        # Hard coded fixed transform to kinect - 
        # TODO: extract this using kdl or urdf parser, stop using tf

        
        self.m = geometry_msgs.msg.TransformStamped()
        self.m.header.frame_id = 'kinect'
        self.m.child_frame_id = 'base_link'

        self.m.transform.translation.x = -0.09925941262730134
        self.m.transform.translation.y = 0.9403368287456643
        self.m.transform.translation.z = 1.4206832337255002

        self.m.transform.rotation.x = 0.6886035023162357
        self.m.transform.rotation.y = -0.6887001480258
        self.m.transform.rotation.z = 0.16050631244663915
        self.m.transform.rotation.w =  0.16048378850163286
        self.transform.setTransform(self.m)
                
    def joint_angle_sampler(self):
        in_image = False
        while not in_image:
            joints = self.kdl_kin.random_joint_angles()
            joint_im = self.project_camera(joints)
#             #215:350,140:370,:
            if (joint_im[0] < 140) | (joint_im[0] > 370) | (joint_im[1] < 215) | (joint_im[1] > 350):
                in_image = False
            else:
                in_image = True
        return joints
    
    def control(self,theta,theta_d,Kp):
        return -Kp*(theta-theta_d)
    
    def roll_out_2d(self,theta,theta_d,Kp,N,dt):
        return [self.project_camera(th) for th in self.roll_out_3d(theta,theta_d,Kp,N,dt)]
        
    def roll_out_3d(self,theta,theta_d,Kp,N,dt):
        th_list = [theta]
        for j in range(N):
            th_list.append(th_list[-1] + self.control(th_list[-1],theta_d,Kp)*dt)
        return th_list

    def FK(self,q):
        return self.kdl_kin.forward(q)
    
    def IK(self,pose):
        return self.kdl_kin.inverse(pose)
    
    def project_camera(self,q):
        
        T = self.FK(q)
        self.update_transforms()
        
        ros_tf.TransformerROS.asMatrix
        
        pose = geometry_msgs.msg.PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = T[0,-1]
        pose.pose.position.y = T[1,-1]
        pose.pose.position.z = T[2,-1]
        q = ros_tf.transformations.quaternion_from_matrix(T)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        
        point_kinect_frame = self.transform.transformPose('kinect',pose)
        
        Xe = np.array(((point_kinect_frame.pose.position.x),(point_kinect_frame.pose.position.y),(point_kinect_frame.pose.position.z)))
        
        xim = self.K.dot(Xe)
        return xim/xim[2]
