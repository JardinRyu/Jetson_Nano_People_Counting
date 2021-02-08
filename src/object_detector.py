import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.tensorrt as trt
import time
import sys

""" TensorFlow detection using TRT optimized graph"""
class ObjectDetection():
    def __init__(self, detection_model_path = './data/ssd_mobilenet_v1_coco_trt_graph.pb'):
        self.detection_model_path = detection_model_path
        self.labels = self._getLabels()

    def _getLabels(self):
        labels = {}
        with open('./data/coco_classes.json') as fh:
            for line in fh:
                label, des = line.strip().split(': ', 1)
                labels[label] = des.strip()
        return labels

    def detect(self, image):
        
        image_expanded = np.expand_dims(image, axis=0)
        
        (boxes, scores, classes, num_detections) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        
        boxes[0, :, [0, 2]] = (boxes[0, :, [0, 2]]*image.shape[0])
        boxes[0, :, [1, 3]] = (boxes[0, :, [1, 3]]*image.shape[1])
        return np.squeeze(boxes).astype(int), np.squeeze(scores), classes

    def _setupTensors(self):
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')        
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')        
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def _getTRTGraph(self):
        with tf.gfile.FastGFile(self.detection_model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    def initializeSession(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.tf_sess = tf.Session(config=tf_config)
        tf.import_graph_def(self._getTRTGraph(), name='')
        self._setupTensors()
        print ("Successfully initialized TF session")

    def __del__(self):
        tf.reset_default_graph()
        self.tf_sess.close()
        print ("Cleanly exited ObjectDetector")

