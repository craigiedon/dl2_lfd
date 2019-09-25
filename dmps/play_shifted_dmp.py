import pickle
from dmp import DMP
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PointStamped
import glob
from matplotlib import pyplot as plt
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import copy

class PlayBack():

    def __init__(self,group):
        rospy.init_node('dmp_playback',anonymous=True)
        rospy.Subscriber('clicked_point',PointStamped,self.callback)
        self.header = None
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(group)

    def load_dmp(self,i):
        with open('dmp%05d.npy'%i,"r") as f:
            obj = pickle.load(f)
        return obj

    def callback(self,msg):
        self.header = msg.header
        self.spin(msg.point)

    def spin(self,point):
        Ndmp = len(glob.glob('./*.npy'))
        for j in range(1,Ndmp+1):
            dmp = self.load_dmp(j)
            current_pose = self.group.get_current_pose().pose
            #Set start to current pose
            dmp.y0[0] = current_pose.position.x
            dmp.y0[1] = current_pose.position.y
            dmp.y0[2] = current_pose.position.z
	    #dmp.y0[3] = current_pose.orientation.x
	    #dmp.y0[4] = current_pose.orientation.y
	    #dmp.y0[5] = current_pose.orientation.z
	    #dmp.y0[6] = current_pose.orientation.w
            #Set goal to an offset above clicked point
            dmp.goal[0] = point.x
            dmp.goal[1] = point.y
            dmp.goal[2] = point.z + 0.26
            y_r,dy_r,ddy_r = dmp.rollout()
            waypoints = []
            for i in range(y_r.shape[0]):
                pose = geometry_msgs.msg.Pose()
                pose.position.x = y_r[i,0]
                pose.position.y = y_r[i,1]
                pose.position.z = y_r[i,2]
                pose.orientation.x = y_r[i,3]
                pose.orientation.y = y_r[i,4]
                pose.orientation.z = y_r[i,5]
                pose.orientation.w = y_r[i,6]

                waypoints.append(copy.deepcopy(pose))

	    plan,fraction = self.group.compute_cartesian_path(waypoints,0.01,0.0)

            self.group.execute(plan,wait=True) 
	    self.group.stop()
	    
            print('Publishing dmp %d'%j)

if __name__ == '__main__':
    pb = PlayBack('left_arm')
    rospy.spin()