from math import pi, atan2, cos, sin
import numpy as np
import cv2
import time
import rospy
import thread
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension

# 6 object points in real world coordinate
object_points = np.matrix([[-19.5, 0, 1, 0, 0, 0], \
                           [0, 0, 0, -19.5, 0, 1], \
                           [-19, 30.5, 1, 0, 0, 0], \
                           [0, 0, 0, -19, 30.5, 1], \
                           [20, 30.5, 1, 0, 0, 0], \
                           [0, 0, 0, 20, 30.5, 1], \
                           [22, 0, 1, 0, 0, 0], \
                           [0, 0, 0, 22, 0, 1], \
                           [0, 0, 1, 0, 0, 0], \
                           [0, 0, 0, 0, 0, 1], \
                           [0, 30.5, 1, 0, 0, 0], \
                           [0, 0, 0, 0, 30.5, 1]])

# 6 corresponding points in pixel world
pixel_points = np.matrix([[77], [210], [80], [60], [259],[58],[268],[211],[167],[210],[167],[59]])

#(77, 210)
#(80, 60)
#(259, 58)
#(268, 211)
#(167, 210)
#(167, 59)

# 4 points in real world coordinate
object_robot_points = np.matrix([[0, 30.5, 1, 0, 0, 0], \
                          	[0, 0, 0, 0, 30.5, 1], \
                          	[20, 30.5, 1, 0, 0, 0], \
                          	[0, 0, 0, 20, 30.5, 1], \
                          	[22, 0, 1, 0, 0, 0], \
                          	[0, 0, 0, 22, 0, 1], \
                          	[0, 0, 1, 0, 0, 0], \
                          	[0, 0, 0, 0, 0, 1]])

# 4 corresponding points in robot coordinate
botpoints_manual = np.matrix([[-2.0], [24.0], [18.0], [24.0], [20.0], [-6.5], [-2.0], [-6.5]])

imgpoints_aff_LQE = np.matrix([[0]])

# define transformation matrix
Pixel_to_world = np.identity(3)
World_to_robot = np.identity(3)

# initialise data type
target_coords = Float64MultiArray()  # the three absolute target coordinates
target_coords.layout.dim.append(MultiArrayDimension())  # coordinates
target_coords.layout.dim[0].label = "coordinates"
target_coords.layout.dim[0].size = 4

target_coords.layout.dim.append(MultiArrayDimension())  # speed
target_coords.layout.dim[1].label = "speed"
target_coords.layout.dim[1].size = 1

pub = 0
continue_loop = True


def ros_spin(message):
    print(message)
    rospy.spin()
    print("ROS spinning aborted")


def callback(value):
    pass


def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)


def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)
    return values


'''def get_point_aff_LQE(event, x, y, flags, param):
    global imgpoints_aff_LQE

    if event == cv2.EVENT_LBUTTONDBLCLK:
        imgpoints_aff_LQE = np.concatenate((imgpoints_aff_LQE, np.array([[x], [y]])), 0)
        print("(" + str(x) + ", " + str(y) + ")")
'''

def calibration_affine_LQE(image):
    #global imgpoints_aff_LQE, object_points
    global object_points,pixel_points
	
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    '''while imgpoints_aff_LQE.shape[0] < 13:
        cv2.imshow("Calibration", gray)
        cv2.setMouseCallback("Calibration", get_point_aff_LQE)
        time.sleep(0.2)
        if cv2.waitKey(1) & 0xFF is ord('q'):
            break
    imgpoints_aff_LQE = imgpoints_aff_LQE[1:, :]'''

    #Column = np.linalg.pinv(object_points) * imgpoints_aff_LQE
    Column = np.linalg.pinv(object_points) * pixel_points

    World_to_pixel = np.concatenate((Column[0:3, :].T, Column[3:, :].T, np.matrix([[0, 0, 1]])), 0)

    return np.linalg.inv(World_to_pixel)


def robot_calibration_affine_LQE_manual():
    global object_robot_points, botpoints_manual

    Column = np.linalg.pinv(object_robot_points) * botpoints_manual

    World_to_robot = np.concatenate((Column[0:3, :].T, Column[3:, :].T, np.matrix([[0, 0, 1]])), 0)

    return World_to_robot

# from pixel to world coordinate
def P2W(x, y):
    global Pixel_to_world
    Pixel_coord = np.matrix([[x], [y], [1]])
    World_coord = Pixel_to_world * Pixel_coord

    return World_coord[0:2, :]

# from world to robot coordinate
def W2R(x, y):
    global World_to_robot
    World_coord = np.matrix([[x], [y], [1]])
    Robot_coord = World_to_robot * World_coord

    return Robot_coord[0:2, :]

# from world to robot coordinate
def W2R_rot(alpha):
    global World_to_robot
    alpha_0 = pi / 4

    if alpha < 2 * pi / 3:
        alpha_t = 2 * pi / 3
    elif alpha > 4 * pi / 3:
        alpha_t = 4 * pi / 3
    else:
        alpha_t = alpha

    alpha_vector_world = np.matrix([[cos(alpha_t)], [sin(alpha_t)], [0]])
    alpha_vector_robot = World_to_robot[0:3, 0:3] * alpha_vector_world

    alpha_robot = atan2(alpha_vector_robot[1], alpha_vector_robot[0]) - alpha_0

    return alpha_robot


def main():
    # **********************************************************************************************************************************
    camera = cv2.VideoCapture(1)
    # camera parameters setting
    camera.set(3,355)
    camera.set(4,288)
    camera.set(5,30)
    # camera.set(10,0.5)

    # Camera calibration (pixel world to real world) **********************************************************************************
    global Pixel_to_world
    ret, image = camera.read()
    Pixel_to_world = calibration_affine_LQE(image)
    print("Camera Calibration Complete !")


    # robot calibration (real world coordinate to robot coordinate) *******************************************************************
    print("Robot Calibration...")
    global World_to_robot
    World_to_robot = robot_calibration_affine_LQE_manual()
    print("Robot Calibration Complete !")
    cv2.destroyAllWindows()

    # adjust threshold for target object detecting
    setup_trackbars('HSV')

    rospy.init_node('eva_robot', anonymous=False)
    global pub
    time.sleep(2)
    pub = rospy.Publisher('eva_move_to', Float64MultiArray, queue_size=1)
    time.sleep(2)
    thread.start_new_thread(ros_spin, ('ROS spinning...',))

    z = 0
    speed = 0.1
    orientation = 0

    while continue_loop:

        # grab the current frame
        ret, frame = camera.read()

        # get threshold
        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values('HSV')
        greenLower = (v1_min, v2_min, v3_min)
        greenUpper = (v1_max, v2_max, v3_max)
	#orange colour	
	#greenLower = (10, 100, 20)
        #greenUpper = (25, 255, 255)

	#red_colour
	#greenLower = (0, 70, 50)
        #greenUpper = (10, 255, 255)

	#green colour
        #greenLower = (7, 55, 0)
        #greenUpper = (67, 255, 255)

	#range for orange, red and green
        greenLower = (0, 55, 0)
        greenUpper = (67, 255, 255)

        #  image processing
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # fps = cv2.CAP_PROP_FPS
        # print("fps:",fps)

        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, "position: (" + str(x) + " , " + str(y) + ")", (10, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                # sending out message **************************************************************************************************
		pixel_world = P2W(x,y)                
		print(pixel_world[0],pixel_world[1])
		Target_position = W2R(pixel_world[0], pixel_world[1])
                target_slope = W2R_rot(orientation)

                robot_target_x = float(Target_position[0])
                robot_target_y = float(Target_position[1])
                robot_target_z = float(z)
                robot_target_o = float(target_slope)
                robot_speed = float(speed)
                print('Publishing eva_move_to...')
                r = rospy.Rate(2000)  # hz
                if not rospy.is_shutdown():
                    target_coords.data = [robot_target_x, robot_target_y, robot_target_z, robot_target_o,
                                          robot_speed]

                    pub.publish(target_coords)
                    r.sleep()

        if cv2.waitKey(1) & 0xFF is ord('q'):
            camera.release()
            break
        # show the frame to our screen
        cv2.imshow("Original", frame)
        # cv2.imshow("Thresh", thresh)
        cv2.imshow("Mask", mask)

if __name__ == '__main__':
	#try:
	main()
	#except rospy.ROSInterruptException:
		#pass
