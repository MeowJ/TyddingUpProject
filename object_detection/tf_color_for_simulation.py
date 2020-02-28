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


# # Model preparation 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#PATH_TO_LABELS = oid_v4_label_map.pbtxt
#MODEL_NAME = 'ssd_mobilenet_v2_oid_v4_2018_12_12'
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
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

#intializing the web camera device

import cv2
cap = cv2.VideoCapture(0)

# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   ret = True
   while (ret):
      ret,image_np = cap.read()
      image = image_np
      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

      # define range of blue color in HSV
      lower_blue = np.array([110,50,50])
      upper_blue = np.array([130,255,255])
      lower_yellow = np.array([20,100,100])
      upper_yellow = np.array([30,255,255])

      # Threshold the HSV image to get only blue colors
      mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

      # Bitwise-AND mask and original image
      res = cv2.bitwise_and(image,image, mask= mask)
      (_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      h, w, d = image.shape
      M = cv2.moments(mask)
           
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
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
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      if M['m00'] > 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        for i, c in enumerate(cnts):
                area = cv2.contourArea(c)
                if area > 1500:
                # BEGIN circle
                    cv2.circle(image, (cx, cy), 10, (0,0,0), -1)
                    cv2.putText(image, "({}, {})".format(int(cx), int(cy)), (int(cx-5), int(cy+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.drawContours(image, cnts, -1, (255, 255, 255),1)
                    #BGIN CONTROL
                    break
                
#      plt.figure(figsize=IMAGE_SIZE)
#      plt.imshow(image_np)
#      cv2.imshow('image',cv2.resize(image_np,(1280,960)))
      cv2.imshow('image',cv2.resize(image_np,(640,480)))
      
#      print(scores)

      (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
      if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        #print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
      else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        #Sprint ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
     

      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break
## COCO-trained models

#NUM_CLASSES = 90
#PATH_TO_LABELS = mscoco_label_map.pbtxt
#MODEL_NAME = ssd_mobilenet_v1_coco_2018_01_28
#MODEL_NAME = ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03
#MODEL_NAME = ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18
#MODEL_NAME = ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18
#MODEL_NAME = ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03
#MODEL_NAME = ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
#MODEL_NAME = ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
#MODEL_NAME = ssd_mobilenet_v2_coco_2018_03_29
#MODEL_NAME = ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
#MODEL_NAME = ssdlite_mobilenet_v2_coco_2018_05_09
#MODEL_NAME = ssd_inception_v2_coco_2018_01_28
#MODEL_NAME = faster_rcnn_inception_v2_coco_2018_01_28
#MODEL_NAME = faster_rcnn_resnet50_coco_2018_01_28
#MODEL_NAME = faster_rcnn_resnet50_lowproposals_coco_2018_01_28
#MODEL_NAME = rfcn_resnet101_coco_2018_01_28
#MODEL_NAME = faster_rcnn_resnet101_coco_2018_01_28
#MODEL_NAME = faster_rcnn_resnet101_lowproposals_coco_2018_01_28
#MODEL_NAME = faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
#MODEL_NAME = faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28
#MODEL_NAME = faster_rcnn_nas_coco_2018_01_28
#MODEL_NAME = faster_rcnn_nas_lowproposals_coco_2018_01_28
#MODEL_NAME = mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
#MODEL_NAME = mask_rcnn_inception_v2_coco_2018_01_28
#MODEL_NAME = mask_rcnn_resnet101_atrous_coco_2018_01_28
#MODEL_NAME = mask_rcnn_resnet50_atrous_coco_2018_01_28

## Kitti-trained models
#PATH_TO_LABELS = kitti_label_map.pbtxt
#NUM_CLASSES = 90
#MODEL_NAME = faster_rcnn_resnet101_kitti_2018_01_28

## Open Images-trained models - NOT TESTED
#PATH_TO_LABELS = oid_v4_label_map.pbtxt
#NUM_CLASSES = 90
#MODEL_NAME = faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28
#MODEL_NAME = faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28
#MODEL_NAME = facessd_mobilenet_v2_quantized_320x320_open_image_v4
#MODEL_NAME = faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12
#MODEL_NAME = ssd_mobilenet_v2_oid_v4_2018_12_12
#MODEL_NAME = ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20


## iNaturalist Species-trained models - NOT TESTED
#PATH_TO_LABELS = fgvc_2854_classes_label_map.pbtxt
#NUM_CLASSES = 90
#MODEL_NAME = faster_rcnn_resnet101_fgvc_2018_07_19
#MODEL_NAME = faster_rcnn_resnet50_fgvc_2018_07_19

## AVA v2.1 trained models - NOT TESTED
#PATH_TO_LABELS = ava_label_map_v2.1.pbtxt
#NUM_CLASSES = 90
#MODEL_NAME = faster_rcnn_resnet101_ava_v2.1_2018_04_30


