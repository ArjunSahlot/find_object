#
#  Find object
#  This is a game in which you have to find the objects listed on the screen. The objects are detected by using tensorflow object detection.
#  Copyright Arjun Sahlot 2021
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import os
import tarfile
import time
import random

import cv2
import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf
from Find_Object.models.research.object_detection.utils import label_map_util
from Find_Object.models.research.object_detection.utils import visualization_utils as vis_util

print("The game will start in about 2 minutes. Get Ready!")

PARENT = os.path.dirname(__file__)
PATH = os.path.join(PARENT, "models", "research", "object_detection")


objects = [
    "bottle",
    "toothpaste",
    "cell phone",
    "chair",
    "scissors",
    "knife",
    "toothbrush",
    "banana",
    "kite",
    "bed",
    "dining table",
    "pizza",
    "cow",
    "remote",
    "couch",
    "bench",
    "frisbee",
    "toilet",
    "laptop",
    "bird",
    "oven",
    "baseball bat",
    "teddy bear",
    "donut",
    "keyboard",
    "tv",
    "cup",
]

cap = cv2.VideoCapture(0)

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = os.path.join(PATH, MODEL_NAME, 'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(PATH, 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = os.path.join(PATH, 'test_images')
TEST_IMAGE_PATHS = [os.path.join(PATH, PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

IMAGE_SIZE = (12, 8)


def is_done(items):
    for done in items.values():
        if not done:
            return False
    return True


def display_items(frame, current_items):
    start_x, start_y = 580, 40
    i = 0
    current_image = frame
    for name, done in current_items.items():
        if not done:
            current_image = cv2.putText(current_image, name, (start_x, start_y + i * 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            current_image = cv2.putText(current_image, name, (start_x, start_y + i * 35), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 3, cv2.LINE_AA)
            current_image = cv2.line(current_image, (start_x, start_y + i * 35 - 8), (start_x + len(name)*15, start_y + i * 35 - 8), (0, 255, 0), 5)

        i += 1

    return current_image


items = {i: False for i in random.sample(objects, 5)}

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        start_time = time.time()
        new_names = []
        finished = False
        while True:
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            names = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)[1]
            for name in names:
                if name in items.keys():
                    items[name] = True
                elif name not in objects and name != "person" and name not in new_names:
                    new_names.append(name)

            image = display_items(cv2.resize(image_np, (800, 600)), items)

            if is_done(items):
                if not finished:
                    finish_time = round(time.time()-start_time, 3)
                    finished = True
                image = cv2.putText(image, str(finish_time), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            else:
                image = cv2.putText(image, str(round(time.time()-start_time, 3)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            cv2.imshow('object detection', image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
if len(new_names) != 0:
    print("NEW NAMES WERE FOUND:")
for name in new_names:
    print(name)
