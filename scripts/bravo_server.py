from flask import Flask, request, jsonify
import numpy as np
from absl import app, flags
import time
import rospy
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from std_msgs.msg import Float64MultiArray
#from geometry_msgs.msg import PoseStamped
from bravo_7_gym.msg import Bravo7State
from motion_primitives.motion import Mover

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "robot_ip", "127.0.0.1", "IP address of the bravo robot"
)

class BravoServer:
    """ Handles the robot communications from env to ROS """
    def __init__(self, robot_ip):
        time.sleep(0.5)
        self.robot_ip = robot_ip
        self.cc_activated = False

        self.robot = Mover()
        # read in ros parameters
        self.state_topic = rospy.get_param("compliance_controller/full_state_topic", "/bravo/gym_state")
        self.cc_cmd_topic = rospy.get_param("compliance_controller/cc_cmd_topic", "/bravo/compliance_controller/command")
        self.toggle_cc_topic = rospy.get_param("compliance_controller/toggle_cc_service_topic", "toggle_compliance_controller")

        # impedance service func
        rospy.wait_for_service(self.toggle_cc_topic)
        self.toggle_cc_srv = rospy.ServiceProxy(self.toggle_cc_topic, SetBool)
        
        # robot state sub
        self.b7state_sub = rospy.Subscriber(self.state_topic, Bravo7State, self._stateCB)
        
        # robot cmd pub
        self.b7CC_pub = rospy.Publisher(self.cc_cmd_topic, Float64MultiArray, queue_size=1)

        self.pose = np.zeros((7,))
        
    def _stateCB(self, msg):
        self.pose = msg.pose
        self.vel = msg.vel
        self.force = msg.force
        self.torque = msg.torque
        self.q = msg.q
        self.dq = msg.dq

    def startCC(self):
        """ Start the compliance controller """
        if not self.cc_activated:
            res = self.toggle_cc_srv(True)
            if res.success:
                self.cc_activated = True
            print(res.msg)
        else:
            print("Already Activated")

    def stopCC(self):
        """ Stop the compliance controller """
        if self.cc_activated:
            res = self.toggle_cc_srv(False)
            if res.success:
                self.cc_activated = False
            print(res.msg)
        else:
            print("Already Deactivated")

    def pubCC(self, pose_cmd):
        """ Publish pose goal to compliance controller """
        msg = Float64MultiArray()
        msg.data = pose_cmd
        self.b7CC_pub.publish(msg)

    def moveToPose(self, pose):
        """ Moves the robot to specified pose using moveit """
        self.robot.go_named_group_state("rest")

    def setPoseGoal(self, pose):
        if self.cc_activated:
            self.pubCC(pose)
        else:
            self.moveToPose(pose)

def main(_):
    ROBOT_IP = FLAGS.robot_ip
    
    webapp = Flask(__name__)

    rospy.init_node("bravo_7_control_server")
    robot_server = BravoServer(
        robot_ip=ROBOT_IP
    )

    @webapp.route("/startCC", methods=["POST"])
    def start_CC():
        print("Starting compliance controller")
        robot_server.startCC()
        return "Compliance Controller Activated"

    @webapp.route("/stopCC", methods=["POST"])
    def stop_CC():
        print("Stopping compliance controller")
        robot_server.stopCC()
        return "Compliance Controller Deactivated"

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify(
            {
                "pose": np.array(robot_server.pose).tolist(),
                "vel": np.array(robot_server.vel).tolist(),
                "force": np.array(robot_server.force).tolist(),
                "torque": np.array(robot_server.torque).tolist(),
                "q": np.array(robot_server.q).tolist(),
                "dq": np.array(robot_server.dq).tolist()
            }
        )
    
    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])
        print("Moving to", pos)
        robot_server.setPoseGoal(pos)
        return "Moved"

    webapp.run(host="127.0.0.1")
if __name__=="__main__":
    print("Starting App")
    app.run(main)