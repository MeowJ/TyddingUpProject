import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np

rospy.loginfo("Start move arm")
x = 0.0
y = 0.0
flag = True
x_old = 0.5
y_old = 0.5
def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    global x, y, x_old, y_old
    x_old = x
    x = data.data[0]
    y_old = y 
    y = data.data[1]


rospy.init_node('move_arm_eva_node', anonymous=False)
while flag:
    if not rospy. is_shutdown():
	    rospy.Subscriber("eva_move_to", Float64MultiArray, callback, queue_size=1)
	    print("collect data")
    if np.sqrt((x_old-x)**2+(y_old - y)**2) < 0.03:
    	flag = False
