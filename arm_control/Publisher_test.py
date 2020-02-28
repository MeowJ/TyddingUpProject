import rospy
from std_msgs.msg import String
import numpy as np

rospy.loginfo("Start to grasp")

while True:
    pub = rospy.Publisher('arm_to_gripper', String, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    pub.publish('o')
