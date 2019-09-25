import sys, time, collections
import copy, math
import rospy
import tf
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String, Header
from moveit_commander.conversions import pose_to_list
import numpy as np

# from moveit_msgs.msgs import GetPositionFK

# Gripper
from pr2_controllers_msgs.msg import Pr2GripperCommandAction, Pr2GripperCommandGoal
import actionlib
from moveit_msgs.srv import GetPositionFK

pose_vec = collections.namedtuple('pose_vec', ['x', 'y', 'z', 'rx', 'ry', 'rz', 'rw'], verbose=False)

def quaternion2list(q):
    return [q.x, q.y, q.z, q.w]

def list2quaternion(l):
    assert len(l) == 4
    return geometry_msgs.msg.Quaternion(*l)

class PR2Gripper(object):
    """docstring for PR2Gripper"""
    def __init__(self, arm):
        super(PR2Gripper, self).__init__()
        self.arm = arm

        self.gripper_action_name = arm[0] + '_gripper_controller/gripper_action' #Specify [r]ight/[l]eft arm controller
        self.gripper_client = actionlib.SimpleActionClient(self.gripper_action_name, Pr2GripperCommandAction)

        if not self.gripper_client.wait_for_server(rospy.Duration(10)):
            rospy.logerr("ArmMover.py: right_gripper_client action server did not come up within timelimit")

    def open(self, dist):
        return self.gripper_controll(dist)

    def close(self, dist, force):
        return self.gripper_controll(dist, max_effort=force)

    def gripper_controll(self, position, max_effort = -1):
        goal = Pr2GripperCommandGoal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        return self.gripper_client.send_goal_and_wait(goal, execute_timeout=rospy.Duration(2.0), 
                                                            preempt_timeout=rospy.Duration(0.1))

class PR2RobotController(object):
    """docstring for PR2RobotController"""
    def __init__(self, group_name, add_table=True):
        super(PR2RobotController, self).__init__()
        self.group_name = group_name

        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)

        if rospy.client._init_node_args is None:
            rospy.init_node('delta_pose_mover', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(self.group_name)

        self.reference_frame = '/odom_combined'
        self.group.set_pose_reference_frame(self.reference_frame)
        self.group.set_goal_tolerance(0.001)

        self.group.set_planning_time(2.)
        self.group.set_max_velocity_scaling_factor(0.5)
        self.group.set_max_acceleration_scaling_factor(0.5)

        self.gripper = PR2Gripper(self.group_name)

        if (add_table):
            self.add_collision_table()
        else:
            self.remove_collision_table()

    def add_collision_sphere(self, point):
        p = geometry_msgs.msg.PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = point[0]
        p.pose.position.y = point[1]
        p.pose.position.z = point[2]
        p.pose.orientation.w = 1
        self.scene.add_sphere("human_sphere", p, radius=0.25)


    def add_and_attach_base_plate(self):
        eef_link = self.group_name[0] +'_wrist_roll_link'
        touch_links = self.robot.get_link_names(group=self.group_name) + [self.group_name[0] + '_gripper_l_finger_tip_link', 
                                                                          self.group_name[0] + '_gripper_r_finger_tip_link']
        # print(touch_links)
        base_plate_pose = geometry_msgs.msg.PoseStamped()
        base_plate_pose.header.frame_id = eef_link
        base_plate_pose.pose.position.x = 0.25
        base_plate_pose.pose.position.y = -0.03
        base_plate_pose.pose.orientation.w = 1.0
        base_plate_name = 'base_plate'
        self.scene.add_box(base_plate_name, base_plate_pose, size=(0.08, 0.08, 0.01))
        rospy.sleep(0.1)
        self.scene.attach_box(eef_link, base_plate_name, touch_links=touch_links)
        rospy.sleep(0.1)
        print('Done attaching base_plate')

    def add_and_attach_gears(self, box_name='gear'):
        eef_link = self.group_name[0] +'_wrist_roll_link'
        touch_links = self.robot.get_link_names(group=self.group_name) + [self.group_name[0] + '_gripper_l_finger_tip_link', 
                                                                          self.group_name[0] + '_gripper_r_finger_tip_link']
        # print(touch_links)
        gear = geometry_msgs.msg.PoseStamped()
        gear.header.frame_id = eef_link
        gear.pose.position.x = 0.23
        gear.pose.orientation.w = 1.0
        gear_name = box_name #'gear'
        self.scene.add_box(gear_name, gear, size=(0.08, 0.01, 0.08))
        rospy.sleep(0.1)
        self.scene.attach_box(eef_link, gear_name, touch_links=touch_links)
        rospy.sleep(0.1)
        print('Done attaching gear')

    def remove_and_deattach_gears(self, box_name='gear'):
        eef_link = self.group_name[0] +'_wrist_roll_link'
        # box_name = 'gear'
        self.scene.remove_attached_object(eef_link, name=box_name)
        self.scene.remove_world_object(box_name)



    def remove_collision_sphere(self, point):
        self.scene.remove_world_object("human_sphere")
        print('Removing human_sphere.')

    def add_collision_table(self):
        rospy.sleep(0.1)

        table_height = 0.025

        p = geometry_msgs.msg.PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = 0.81 # fwd
        p.pose.position.y = 0.6 # left/right
        p.pose.position.z = 0.73 - table_height/2.
        # p.pose.orientation.w = 0.924
        # p.pose.orientation.z = 0.383
        p.pose.orientation.w = 1.
        
        self.scene.add_box("table", p, (0.6, 1.2, table_height))

    def remove_collision_table(self):
        self.scene.remove_world_object("table")
        print('Removing table')

    def add_collision_safety_box(self):
        rospy.sleep(0.1)

        table_height = 1.

        p = geometry_msgs.msg.PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x = 0.5 # fwd
        p.pose.position.y = 0.0 # left/right
        p.pose.position.z = 0.6 - table_height/2.
        # p.pose.orientation.w = 0.924
        # p.pose.orientation.z = 0.383
        p.pose.orientation.w = 1.
        
        self.scene.add_box("safety_box", p, (0.6, 1.2, table_height))
        print('Added safety box.')


    def remove_collision_safety_box(self):
        self.scene.remove_world_object("safety_box")
        print('Removing safety box.')

    def open_gripper(self, dist=0.08):
        self.gripper.open(dist)

    def close_gripper(self, force=50):
        self.gripper.close(dist=0, force=force)

    def plan(self, p, rpy):
        print(p, rpy)
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = p[0]
        pose_goal.position.y = p[1]
        pose_goal.position.z = p[2]

        qt = tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        pose_goal.orientation.x = qt[0]
        pose_goal.orientation.y = qt[1]
        pose_goal.orientation.z = qt[2]
        pose_goal.orientation.w = qt[3]

        return self.group.plan(pose_goal)

    def plan(self, pose):
        print('Target pose', pose)
        return self.group.plan(pose)

    def service_left_fk(self, joint_state):
        rospy.wait_for_service('/pr2_left_arm_kinematics/get_fk')
        try:
            ## include <kinematics_msgs/GetKinematicSolverInfo.h> kinematics_msgs/GetPositionFK.h> moveit_msgs
            moveit_fk = rospy.ServiceProxy('/pr2_left_arm_kinematics/get_fk', GetPositionFK)
            
            # A vector of link name for which forward kinematics must be computed
            fk_link_names = ['l_wrist_roll_link']
            joint_names = ['l_upper_arm_roll_joint', 'l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_forearm_roll_joint', 
                              'l_elbow_flex_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint']
            # joint_positions = []
            # for i in range(7):
            #   joint_names.append('right_arm_j'+str(i)) # your names may vary
            #   joint_positions.append(0.8) # try some arbitrary joint angle
            header = Header(0, rospy.Time.now(), "/base_link")
            rs = moveit_commander.RobotState() # this is a moveir msg? http://docs.ros.org/api/moveit_msgs/html/msg/RobotState.html
            rs.joint_state.name = joint_names
            rs.joint_state.position = joint_state
            rospy.logwarn('Requesting fk.')
            fk_pose = moveit_fk(header, fk_link_names, rs)

            print("FK LOOKUP:", fk_pose) # Lookup the pose
        except rospy.ServiceException, e:
            rospy.logerror("Service call failed: %s"%e)
        #### To test, execute joint_positions here ####
        rospy.loginfo(["POST MOVE:", self.get_current_pose()]) # Verify that the new pose matches your computed pose

        # The 3d/6d pose
        return fk_pose

    # def KDL_fk(self, joint_state):
    #     from urdf_parser_py.urdf import URDF
    #     from pykdl_utils.kdl_kinematics import KDLKinematics
    #     robot = URDF.from_parameter_server()
    #     kdl_kin = KDLKinematics(robot, base_link, end_link)
    #     q = kdl_kin.random_joint_angles()
    #     pose = kdl_kin.forward(q) # forward kinematics (returns homogeneous 4x4 numpy.mat)
    #     q_ik = kdl_kin.inverse(pose, q+0.3) # inverse kinematics
    #     if q_ik is not None:
    #         pose_sol = kdl_kin.forward(q_ik) # should equal pose
    #     J = kdl_kin.jacobian(q)
    #     print 'q:', q
    #     print 'q_ik:', q_ik
    #     print 'pose:', pose
    #     if q_ik is not None:
    #         print 'pose_sol:', pose_sol
    #     print 'J:', J

    #     return pose

    def reset_pose(self):
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 1.0

        # roll = -0.08
        # pitch = -0.58
        # yaw = 0.992
        roll, pitch, yaw = [-1.765, 0.082, 1.170]

        qt = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        plan_success = False

        while not plan_success:
            # x = np.random.uniform(low=0.52, high=0.54)
            # y = np.random.uniform(low=-0.42, high=0.43)
            # z = np.random.uniform(low=0.87, high=0.88)
            x, y, z = [0.576, -0.462, 0.910]

            pose_goal.position.x = x
            pose_goal.position.y = y
            pose_goal.position.z = z

            pose_goal.orientation.x = qt[0]
            pose_goal.orientation.y = qt[1]
            pose_goal.orientation.z = qt[2]
            pose_goal.orientation.w = qt[3]

            self.group.set_pose_target(pose_goal)
            plan_success = self.group.go(wait=True)
            # rospy.sleep(5)

        self.group.stop()
        self.group.clear_pose_targets()
        # print("Reset to initial position {0}".format(pose_goal.position))

    def move_to_p_rpy(self, p, rpy):
        new = geometry_msgs.msg.Pose()
        
        new.position.x = p[0]
        new.position.y = p[1]
        new.position.z = p[2]

        new.orientation = list2quaternion(tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2]))
        print('New target: ', new)
        self.group.set_pose_target(new)
        success = self.group.go(wait=True)
        return success

    def get_current_pose(self):
        return self.group.get_current_pose().pose


    def apply_delta_pose(self, orig, delta):
        new = geometry_msgs.msg.Pose()
        
        new.position.x = orig.position.x + delta.position.x
        new.position.y = orig.position.y + delta.position.y
        new.position.z = orig.position.z + delta.position.z

        # Todo: add orientation offset
        # w, x, y, z = q_mult(orig.orientation, delta.orientation)
        # new.orientation.x = x
        # new.orientation.y = y
        # new.orientation.z = z
        # new.orientation.w = w

        # http://wiki.ros.org/tf2/Tutorials/Quaternions
        new.orientation = list2quaternion(tf.transformations.quaternion_multiply(quaternion2list(delta.orientation), quaternion2list(orig.orientation)))

        return new


    # Tf useful - https://answers.ros.org/question/69754/quaternion-transformations-in-python/?answer=69799#post-id-69799
    def _move_delta_geom_pose(self, d):
        # Calc new pose

        c = self.group.get_current_pose().pose
        
        if self.reference_frame == "/base_link":
            c.position.z -= 0.051
        elif self.reference_frame != "/odom_combined":
            print("WARNING: PLEASE CHECK YOUR REFERENCE FRAME!")

        new = self.apply_delta_pose(c, d)

        # Test:
        # quaternion = (c.orientation.x, c.orientation.y, c.orientation.z, c.orientation.w)
        # euler = tf.transformations.euler_from_quaternion(quaternion)
        # quaternion2 = (new.orientation.x, new.orientation.y, new.orientation.z, new.orientation.w)
        # euler2 = tf.transformations.euler_from_quaternion(quaternion2)
        # print('Euler before', euler, ' Euler after: ', euler2)
        ## End test

        # Move to new pose
        self.group.set_pose_target(new)
        # print('Going to pose')
        success = self.group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        # self.group.stop()
        # self.group.clear_pose_targets()
        # print('Stopped')
        return success

    def move_delta(self, d):
        delta = geometry_msgs.msg.Pose()

        for k in ['x', 'y', 'z']:
            setattr(delta.orientation, k, getattr(d, 'r' + k))
            setattr(delta.position, k, getattr(d, k))
        delta.orientation.w = d.rw

        return self._move_delta_geom_pose(delta)

    def move_delta_t_rpy(self, t, rpy):
        assert len(t) == 3
        assert len(rpy) == 3

        qt = tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        delta = pose_vec(t[0], t[1], t[2], qt[0], qt[1], qt[2], qt[3])

        return self.move_delta(delta)

    def move_delta_t(self, t):
        return self.move_delta_t_rpy(t, [0, 0, 0])


def main3():
    pr2 = PR2RobotController('right_arm', add_table=False)
    # pr2 = PR2RobotController('left_arm', add_table=False)

    pr2.reset_pose()

    for _ in range(100):
        delta_t = np.random.uniform(low=-0.02, high=0.02, size=3)
        delta_rpy = np.random.uniform(low=-math.pi/6., high=math.pi/6., size=3)
        print(delta_t)
        print(delta_rpy)
        # time.sleep(2)
        # pr2.move_delta_t(delta_t)
        success = pr2.move_delta_t_rpy(delta_t, delta_rpy)
        if (not success):
            print('Couldn\'t move to that pose')


def main2():
    pr2 = PR2RobotController('right_arm')
    pr2.reset_pose()


    roll = 0
    pitch = 0
    yaw = math.pi/4
    qt = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    print(qt)
    # #type(pose) = geometry_msgs.msg.Pose
    # pose.orientation.x = qt[0]
    # pose.orientation.y = qt[1]
    # pose.orientation.z = qt[2]
    # pose.orientation.w = qt[3]


    delta = pose_vec(-0.00, 0, 0, qt[0], qt[1], qt[2], qt[3])

    for _ in range(2):
        pr2.move_delta(delta)
    # pr2.move_delta(delta)
    # pr2.move_delta(delta)


def main():
    ## First initialize `moveit_commander`_ and a `rospy`_ node:
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('delta_gripper_mover', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "right_arm"
    group = moveit_commander.MoveGroupCommander(group_name)


    planning_frame = group.get_planning_frame()
    print "============ Reference frame: %s" % planning_frame
    # We can also print the name of the end-effector link for this group:
    eef_link = group.get_end_effector_link()
    print "============ End effector: %s" % eef_link

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print "============ Robot Groups:", robot.get_group_names()

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print "============ Printing robot state"
    print robot.get_current_state()
    print('Current pose:')
    print group.get_current_pose()
    print ""

    group.set_goal_tolerance(0.001)

    #group.set_pose_reference_frame('/r_wrist_roll_link')

    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.w = 1.0
    pose_goal.position.x = 0.4
    pose_goal.position.y = -0.2
    pose_goal.position.z = 0.5
    group.set_pose_target(pose_goal)

    ## Now, we call the planner to compute the plan and execute it.
    print('Going to pose')
    plan = group.go(wait=True)
    # Calling `stop()` ensures that there is no residual movement
    group.stop()
    print('Stopped')
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    group.clear_pose_targets()
    print('Done.')
    print('Pose reference frame: ', group.get_pose_reference_frame())
    print('Tolerance: ', group.get_goal_joint_tolerance())
    print('Planning time: ', group.get_planning_time())
    print group.get_current_pose()

if __name__ == '__main__':
  main3()