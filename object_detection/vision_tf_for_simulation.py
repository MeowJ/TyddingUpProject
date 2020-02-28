#!/usr/bin/env python



import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from utils import label_map_util

from utils import visualization_utils as vis_util

from math import pi, atan2, cos, sin
import time
import rospy
import thread
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Float32

# calibration ***********************************************************************************************************88
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

#********************************************************************************************************
def calibration_affine_LQE(image):
    #global imgpoints_aff_LQE, object_points
    global object_points,pixel_points
    # print(image)
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

#********************************************************************************************************

# # Model preparation 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
    print ('Downloading the model')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
    print ('Download complete')
else:
    print ('Model already exists')

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
###########################################################################################################

import rospy, sys, numpy as np
import moveit_commander
from copy import deepcopy
import geometry_msgs.msg
from ur5_notebook.msg import Tracker
import moveit_msgs.msg
import cv2, cv_bridge
from sensor_msgs.msg import Image


from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
tracker = Tracker()

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
sess = tf.Session(graph=detection_graph)

class vision:
    def __init__(self):
        rospy.init_node("vision", anonymous=False)
        self.track_flag = False
        self.default_pose_flag = True
        self.cx = 400.0
        self.cy = 400.0
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber('/panda/usbcam/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.cxy_pub = rospy.Publisher('coordinate_to_arm_', Tracker, queue_size=1)
        self.pub_x = rospy.Publisher('coordinate_to_arm', Float32, queue_size=1)
        self.pub_y = rospy.Publisher('color_detection_y', Float32, queue_size=1)
        self.pub_x_tf = rospy.Publisher('tf_detection_x', Float32, queue_size=1)
        self.pub_y_tf = rospy.Publisher('tf_detection_y', Float32, queue_size=1)

    def image_callback(self,msg):
        # BEGIN BRIDGE
        global sess
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        ##################################################################
        # ################################################
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # END HSV
        # BEGIN FILTER
        lower_red = np.array([15,0,0])
        upper_red = np.array([36, 255, 255])
        #(15,0,0), (36, 255, 255)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        (_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #area = cv2.contourArea(cnts)
        h, w, d = image.shape
        # print h, w, d  (800,800,3)
        #BEGIN FINDER
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

        # cx range (55,750) cy range( 55, ~ )
        # END FINDER
        # Isolate largest contour
        #  contour_sizes = [(cv2.contourArea(contour), contour) for contour in cnts]
        #  biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            for i, c in enumerate(cnts):
                area = cv2.contourArea(c)
                if area > 1000:
                    self.track_flag = True
                    self.cx = cx
                    self.cy = cy
                    self.error_x = self.cx - w/2
                    self.error_y = self.cy - (h/2+195)
                    tracker.x = cx
                    tracker.y = cy
                    tracker.flag1 = self.track_flag
                    tracker.error_x = self.error_x
                    tracker.error_y = self.error_y
                    #(_,_,w_b,h_b)=cv2.boundingRect(c)
                    #print w_b,h_b
                    # BEGIN circle
                    #cv2.circle(image, (cx, cy), 10, (0,0,0), -1)
                    #cv2.putText(image, "({}, {})".format(int(cx), int(cy)), (int(cx-5), int(cy+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.drawContours(image, cnts, -1, (255, 255, 255),1)
                    #BGIN CONTROL
                    break
                else:
                    self.track_flag = False
                    tracker.flag1 = self.track_flag
                    ###############################################################
        # END BRIDGE
        # BEGIN HSV
        image_np = image
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        
        im_height, im_width,channel = image_np.shape
        tf_y = float(im_height)*(boxes[0][0][0]+boxes[0][0][2])/2
        tf_x = float(im_width)*(boxes[0][0][1]+boxes[0][0][3])/2
        #y = 288*(boxes[0][0][0]+boxes[0][0][2])/2
        #x = 352*(boxes[0][0][1]+boxes[0][0][3])/2
      # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        self.cxy_pub.publish(tracker)
        self.pub_x.publish(tracker.x)
        self.pub_y.publish(tracker.y)
        self.pub_x_tf.publish(tf_x)
        self.pub_y_tf.publish(tf_y)

        cv2.namedWindow("window", 1)
        cv2.imshow("window", image )
        cv2.waitKey(1)

follower=vision()
rospy.spin()
