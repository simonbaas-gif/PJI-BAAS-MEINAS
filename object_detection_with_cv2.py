# # Imports

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

# ## Time librairies and argument importation

import time
import argparse
import imutils

# ## Arguments parsing

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = False, default = 0, help = "Video type (0 : webcam, 1 : USB camera, <video_file.mp4> : file in the folder")
ap.add_argument("-t", "--timer", required = False, default = 5, help = "Image capture frenquency", type = int)
ap.add_argument("-w", "--width", required = False, default = 1200, help = "Image width", type = int)
ap.add_argument("-d", "--duration", required = False, default = 600, help = "Capture duration", type = int)
args = vars(ap.parse_args())

# ## OpenCV importation

import cv2

if(args["video"] == "0" or args["video"] == "1"):
  args["video"] = int(args["video"])
capture = cv2.VideoCapture(args["video"])

# ## Env setup

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
# MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28' # SLOW
# MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28' # SLOW
# MODEL_NAME = 'inference_graph' # OWN MODEL

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
# PATH_TO_LABELS = 'training/labelmap.pbtxt' # OWN LABELMAP

NUM_CLASSES = 90
# NUM_CLASSES = 5 #OWN CLASSES

# ## Download Model

print("Downloading model")

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())

print ("Loading frozen model into memory")

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name = '')

# ## Loading label map

# Label maps matches the id and the classe name
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = NUM_CLASSES, use_display_name = True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code

# def load_image_into_numpy_array(image):
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)

# ## Detecte the most important obstacle

# def most_important_obstacle(image, vertices):
#   mask = np.zeros_like(image)
#   if len(image.shape) > 2:
#     channel_count = image.shape[2]
#     ignore_mask_color = (255,) * channel_count
#   else:
#     ignore_mask_color = 255
#   cv2.fillPoly(mask, vertices, ignore_mask_color)
#   masked_image = cv2.bitwise_and(image, mask)
#   return masked_image

# # Detection

# PATH_TO_TEST_IMAGES_DIR = 'test_images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)

# ## Variables

# ### Average variables

moyenne_score = 0
compteur = 0
list_objects = []

# ### Time variables

img_cpt = 0
timer = time.time()
start_time = time.time()
end_time = time.time()

with detection_graph.as_default():
  with tf.Session(graph = detection_graph) as sess:
    while (capture.isOpened()):
      ret, image_np = capture.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis = 0)
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
          feed_dict = { image_tensor : image_np_expanded })
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates = True,
          line_thickness = 3)


      # Crée deux zones, à gauche et à droite. Aend_time de détecter dans quelle zone se toruve l'obsacle
      # image_np_width = image_np.shape[0]
      # image_np_height = image_np.shape[1]
      #
      # rows, cols = image_np.shape[:2]
      #
      # left_boundary = [int(cols*0.40), int(rows*0.95)]
      # left_boundary_top = [int(cols*0.40), int(rows*0.20)]
      # right_boundary = [int(cols*0.60), int(rows*0.95)]
      # right_boundary_top = [int(cols*0.60), int(rows*0.20)]
      #
      # bottom_left = [int(cols*0.20), int(rows*0.95)]
      # top_left = [int(cols*0.20), int(rows*0.20)]
      # bottom_right = [int(cols*0.80), int(rows*0.95)]
      # top_right = [int(cols*0.80), int(rows*0.20)]
      # vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
      #
      # cv2.line(image_np,tuple(bottom_left),tuple(bottom_right), (255, 0, 0), 5)
      # cv2.line(image_np,tuple(bottom_right),tuple(top_right), (255, 0, 0), 5)
      # cv2.line(image_np,tuple(top_left),tuple(bottom_left), (255, 0, 0), 5)
      # cv2.line(image_np,tuple(top_left),tuple(top_right), (255, 0, 0), 5)
      #
      # copied = np.copy(image_np)
      # interested = most_important_obstacle(copied,vertices)
      #
      # print(image_np_width,image_np_height)
      #
      # ymin = int((boxes[0][0][0]*image_np_width))
      # xmin = int((boxes[0][0][1]*image_np_height))
      # ymax = int((boxes[0][0][2]*image_np_width))
      # xmax = int((boxes[0][0][3]*image_np_height))
      #
      # Result = np.array(image_np[ymin:ymax,xmin:xmax])
      #
      # ymin_str='y min  = %.2f '%(ymin)
      # ymax_str='y max  = %.2f '%(ymax)
      # xmin_str='x min  = %.2f '%(xmin)
      # xmax_str='x max  = %.2f '%(xmax)
      #
      # cv2.putText(image_np,ymin_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
      # cv2.putText(image_np,ymax_str, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
      # cv2.putText(image_np,xmin_str, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
      # cv2.putText(image_np,xmax_str, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
      #
      # print(scores.max())
      # print("left_boundary[0],right_boundary[0] :", left_boundary[0], right_boundary[0])
      # print("left_boundary[1],right_boundary[1] :", left_boundary[1], right_boundary[1])
      # print("xmin, xmax :", xmin, xmax)
      # print("ymin, ymax :", ymin, ymax)
      #
      # if scores.max() > 0.78:
      #    print("inif")
      # if(xmin >= left_boundary[0]):
      #    print("Move Left - 1st !")
      #    cv2.putText(image_np,'Move Left !', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
      # elif(xmax <= right_boundary[0]):
      #    print("Move Right - 2nd !")
      #    cv2.putText(image_np,'Move Right !', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
      # elif(xmin <= left_boundary[0] and xmax >= right_boundary[0]):
      #    print("STOP ! - 3nd !")
      #    cv2.putText(image_np,' Stop !', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
      #    cv2.line(image_np,tuple(left_boundary),tuple(left_boundary_top), (255, 0, 0), 5)
      # cv2.line(image_np,tuple(right_boundary),tuple(right_boundary_top), (255, 0, 0), 5)
      # cv2.write(image_np)
      # end_time de la partie concernant les deux zones


      objects = [] # List of objects detected for each frame
      threshold = 0.5 # Threshold of detection 
      for index, value in enumerate(classes[0]):
        object_dict = {}
        if scores[0, index] > threshold:
          object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                    scores[0, index]
          objects.append(object_dict)
        list_objects.append(objects)
      # print(list_objects)

      cv2.imshow("Obstacles detection", cv2.resize(image_np, (800, 600)))
      if cv2.waitKey(1) & 0xFF == ord('q'):
        print('window closed')
        cv2.destroyAllWindows()
        break

      # Time section
      # Indicate the duration (argument --duration)
      if (end_time - start_time) >= args["duration"]:
        print('time is over')
        break
      # Save pictures every x secondes (argument --timer)
      if time.time() - timer >= args["timer"]:
        img_name = "images_save/image_np_{}.png".format(img_cpt)
        cv2.imwrite(img_name, image_np)
        # print("{} written!".format(img_cpt))
        timer = time.time()
        end_time = time.time()
        # print (end_time - start_time)
      img_cpt += 1
      # End time section

    # Release the capture and destroy the window
    capture.release()
    cv2.destroyAllWindows()

# ## Calculate the average of detection for each element

def avg_for_each_vehicles(tab):
  res = {}
  number = {}
  for list_elem in tab:
    for elem in list_elem:
      for lab in elem:
        score = elem[lab]
        if lab in res:
          res[lab] += score
          number[lab] += 1
        else:
          res[lab] = score
          number[lab] = 1
  for elem in res:
    res[elem] = res[elem] / number[elem]
  return res

print("Average : ")
print(avg_for_each_vehicles(list_objects))
print("List Objects : ")
print(list_objects)

# ## Create the graph picture

def plot(tab):
  fig = plt.figure()
  x = []
  y = []
  for el in tab:
    x.append(el)
    y.append(tab[el])
  width = 0.5
  plt.bar(x, y, width, color='r' )
  plt.xlabel("Type of vehicles")
  plt.ylabel("Average of detection")
  plt.savefig('AverageForEachVehicles.png')
  # plt.show()

plot(avg_for_each_vehicles(list_objects))
