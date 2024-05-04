
from mlsocket import MLSocket
import rospy
from std_msgs.msg import Float64MultiArray
import threading

i = 0
def connectionCB(conn):
    global i
    i = i + 1
    cmd = Float64MultiArray()
    while not rospy.is_shutdown():
        try:
            raw = conn.recv(1024)
            cmd.data = raw
            cmdPub.publish(cmd)
        except Exception as e:
            print(f"{i}:{e}")
            return



if __name__ == "__main__":
    rospy.init_node("ee_state_publisher")
    host = rospy.get_param("host", '127.0.0.1')
    cmd_port = rospy.get_param("cmd_port", 65432)
    cmd_topic = rospy.get_param("cmd_topic", "/bravo/compliance_controller/command")

    cmdPub = rospy.Publisher(cmd_topic, Float64MultiArray, queue_size=1)

    
    with MLSocket() as s:
        print("Waiting for connection on", cmd_port,"...")
        s.bind((host, cmd_port))
        while not rospy.is_shutdown():
            s.listen()
            conn, address = s.accept()
            print("Connected to:", address)
            cmdThread = threading.Thread(target=connectionCB, args=(conn,))
            cmdThread.start()