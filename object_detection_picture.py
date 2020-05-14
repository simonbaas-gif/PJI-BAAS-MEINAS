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

# ## Env setup

sys.path.append("..")
from object_detection.utils import ops as utils_ops

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

# ## Load a (frozen) Tensorflow model into memory.

print ("Loading frozen model into memory")

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map

# Label maps matches the id and the classe name
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# # Detection

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 11) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# ## Variables

# ### Average variables

moyenne_score = 0
compteur = 0
list_objects = []
img_number = 1

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
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
          line_thickness=3)

      objects = [] # Lis of objects detected for each frame
      threshold = 0.5 # Threshold of detection
      for index, value in enumerate(classes[0]):
        object_dict = {}
        if scores[0, index] > threshold:
          object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                    scores[0, index]
          objects.append(object_dict)
      list_objects.append(objects)
      plt.figure(figsize=IMAGE_SIZE)
      # plt.show(image_np)
      plt.imsave('test_images/result/image' + str(img_number) + ".jpg", image_np, format='jpg')
      img_number += 1

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

print("Average of detection : ")
print(avg_for_each_vehicles(list_objects))
print("List Objects : ")
print(list_objects)

# ## Create the graph picture

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

# ## For 10 pictures which are describe, calculate the average of the number of classes detected

descriptionImage = []
#Image 1
descriptionImage.append({b'car':4, b'person':1})
#Image 2
descriptionImage.append({b'car':7, b'person':2, b'motorcycle':1})
#Image 3
descriptionImage.append({b'car':1, b'truck':1})
#Image 4
descriptionImage.append({b'car':1, b'truck':1, b'person':5, b'motorcycle':1})
#Image 5
descriptionImage.append({b'car':11, b'truck':1, b'motorcycle':5, b'person':5})
#Image 6
descriptionImage.append({b'car':1, b'person':2})
#Image 7
descriptionImage.append({b'car':3, b'truck':1, b'person':1})
#Image 8
descriptionImage.append({b'car':1, b'truck':1, b'person':1})
#Image 9
descriptionImage.append({b'truck':1})
#Image 10
descriptionImage.append({b'bus':1})

nbElements = 23

def testImage(tab):
    res = 0
    for i in range(len(descriptionImage)):
        for el in descriptionImage[i]:
            #car
            cpt = 0
            for obj in tab[i]:
                for key in obj:
                    if key == el:
                        cpt += 1
            res += cpt / descriptionImage[i][el]
    res = res / nbElements
    return res

print("Average of classes number : ")
print(testImage(list_objects))