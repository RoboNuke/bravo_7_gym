from flask import Flask, request, jsonify
import numpy as np
from absl import app, flags
import time
import copy

import rospy
from std_srvs.srv import SetBool
from std_msgs.msg import Float64MultiArray
from rsa_bravo_msgs.srv import TareForceTorque
from geometry_msgs.msg import PoseStamped
from bravo_7_gym.msg import Bravo7State
from motion_primitives.motion import Mover
from bravo_controllers.srv import StartPlayback

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

        self.ee_frame = rospy.get_param("compliance_controller/ee_frame", "ee_frame")
        self.world_frame = rospy.get_param("compliance_controller/world_frame", "bravo_base_link")
        # impedance service func
        rospy.wait_for_service(self.toggle_cc_topic)
        self.toggle_cc_srv = rospy.ServiceProxy(self.toggle_cc_topic, SetBool)
        
        # robot state sub
        self.b7state_sub = rospy.Subscriber(self.state_topic, Bravo7State, self._stateCB)
        
        # robot cmd pub
        self.b7CC_pub = rospy.Publisher(self.cc_cmd_topic, Float64MultiArray, queue_size=1)

        self.pose = np.zeros((7,))
        print("Servo Obj init")
        
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
        return True

    def moveToPose(self, pose):
        """ Moves the robot to specified pose using moveit """
        #self.robot.go_named_group_state("rest")
        print("Reset Pose:", pose)
        PS = PoseStamped()
        PS.header.frame_id = self.world_frame
        PS.pose.position.x = pose[0]
        PS.pose.position.y = pose[1]
        PS.pose.position.z = pose[2]
        PS.pose.orientation.x = pose[3]
        PS.pose.orientation.y = pose[4]
        PS.pose.orientation.z = pose[5]
        PS.pose.orientation.w = pose[6]
        success = self.robot.go_ee_pose(pose = PS, wait = True)
        return success

    def setPoseGoal(self, pose):
        if self.cc_activated:
            return self.pubCC(pose)
        else:
            return self.moveToPose(pose)
    
    def moveToNamedState(self, name, wait):
        success = self.robot.go_named_group_state(name, wait)
        return success

    def tareFTSensor(self):
        """ Moves to home position, tares ft sensor to zero, moves back to previous position """
        curPos = copy.deepcopy(self.pose)
        self.robot.go_named_group_state("home", wait=True, retry = True)
        print("Waiting for ft tare service...")
        rospy.wait_for_service('/bravo/tare_ft_sensor')
        print("\t Got it!")
        serv = rospy.ServiceProxy('/bravo/tare_ft_sensor', TareForceTorque)
        print("\tGot serv")
        res = serv(np.zeros((6,)))
        if res.success:
            print("\tSuccessfully tared ft sensor")
        else:
            print("\tFT Sensor Tare Failed")
        self.moveToPose(curPos)
        return res.success

def main(_):
    ROBOT_IP = FLAGS.robot_ip
    print("ROBOT_IP:", ROBOT_IP)
    webapp = Flask(__name__)

    rospy.init_node("bravo_7_control_server")
    robot_server = BravoServer(
        robot_ip=ROBOT_IP
    )

    print("ROS Init")
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
        success = robot_server.setPoseGoal(pos)
        return jsonify({"success": success})
    
    @webapp.route("/toNamedPose", methods=["POST"])
    def toNamedPose():
        name = str(request.json['name'])
        wait = bool(request.json['wait'])
        print("Moving to ", name, len(name))
        success = robot_server.moveToNamedState(name, wait)

        print("Robot " + "has" if success else "has not" + " moved to " + name)
        return jsonify({"success": success})
    
    @webapp.route("/loadTraj", methods=["POST"])
    def getLoadedPath():
        filepath = str(request.json['filepath'])
        dt = float(request.json['dt'])
        steps = int(request.json['steps'])
        playback = bool(request.json['playback'])

        print("Waiting for playback_ee_trajectory service...")
        rospy.wait_for_service('/playback_ee_trajectory')
        print("\t Got it!")
        serv = rospy.ServiceProxy('/playback_ee_trajectory', StartPlayback)
        print("Got serv")
        res = serv(filepath, dt, steps, playback)
        print("Successful:", res.success)
        if res.success:
            return jsonify({"traj":res.traj.data})
        
    @webapp.route("/tareFTSensor", methods=["POST"])
    def tareFTSensor():
        print("Starting tare")
        try:
            robot_server.tareFTSensor()
        except Exception as e:
            print(e)
        return "Tared FT Sensor"

    print("Functs init")
    webapp.run(host=ROBOT_IP)
    print("Webapp running")

if __name__=="__main__":
    print("Starting App")
    app.run(main)