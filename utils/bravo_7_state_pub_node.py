import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import JointState
from bravo_7_gym.msg import Bravo7State
from geometry_msgs.msg import WrenchStamped, Wrench
from sensor_msgs.msg import JointState
import numpy as np
from math import pi

class Bravo7StatePublisher:
    def __init__(self):
        self.statePub = rospy.Publisher('/bravo/gym_state', JointState, queue_size=10)

        self.ee_link_name = rospy.get_param("ee_link", "ee_link")
        self.world_frame = rospy.get_param("world_frame", "bravo_base_link")
        self.dt = rospy.get_param("dt",0.001)

        self.joint_topic = rospy.get_param("joint_topic", "/bravo/joint_states")
        self.ft_topic = rospy.get_param("ft_sensor_topic", "bravo/filtered_force_torque")
        self.ee_topic = rospy.get_param("ee_state_topic", "/bravo/ee_state")
        self.b7_topic = rospy.get_param("bravo7_state_topic", "/bravo/gym_state")

        self.bravo7state = Bravo7State()
        self.bravo7state.force = np.zeros(3)
        self.bravo7state.torque = np.zeros(3)
        self.lastTime = None
        self.tDiff = None

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.FTSub = rospy.Subscriber(self.ft_topic, WrenchStamped, self.ft_callback)
        self.jointSub = rospy.Subscriber(self.joint_topic, JointState, self.joint_callback)
        self.eeSub = rospy.Subscriber(self.ee_topic, JointState, self.ee_callback)
        self.b7statePub = rospy.Publisher(self.b7_topic, Bravo7State, queue_size=1)

    def pubState(self):
        self.b7statePub.publish(self.bravo7state)

    def ee_callback(self, msg):
        self.bravo7state.pose = msg.position
        self.bravo7state.vel = msg.velocity

    def joint_callback(self, msg):
        self.bravo7state.q = msg.position[1:]
        self.bravo7state.dq = msg.velocity[1:]
        self.bravo7state.gripper = msg.position[0]

    def ft_callback(self, msg):
        self.bravo7state.force[0] = msg.wrench.force.x
        self.bravo7state.force[1] = msg.wrench.force.y
        self.bravo7state.force[2] = msg.wrench.force.z

        self.bravo7state.torque[0] = msg.wrench.torque.x
        self.bravo7state.torque[1] = msg.wrench.torque.y
        self.bravo7state.torque[2] = msg.wrench.torque.z

    def spin(self):
        while not rospy.is_shutdown():
            self.pubState()
            rospy.Rate(1.0/self.dt).sleep()

if __name__=="__main__":
    rospy.init_node("ee_state_publisher")
    sp = Bravo7StatePublisher()
    sp.spin()