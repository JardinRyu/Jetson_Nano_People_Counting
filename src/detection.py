import tensorflow as tf
import numpy as np
import cv2
import time
import sys

from tensorflow.python.compiler.tensorrt import trt_convert as trt

""" TensorFlow detection using TRT optimized graph"""
class Detector():
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

    def _setupTensors(self):
        self.image_tensor = self.tf_sess.graph.get_tensor_by_name('image_tensor:0')        
        self.boxes = self.tf_sess.graph.get_tensor_by_name('detection_boxes:0')        
        self.scores = self.tf_sess.graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.tf_sess.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.tf_sess.graph.get_tensor_by_name('num_detections:0')

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

    def detect(self, image):
        #image = cv2.resize(image, (300, 300))
        image_expanded = np.expand_dims(image, axis=0)
        
        (boxes, scores, classes, num_detections) = self.tf_sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        
        boxes[0, :, [0, 2]] = (boxes[0, :, [0, 2]]*image.shape[0])
        boxes[0, :, [1, 3]] = (boxes[0, :, [1, 3]]*image.shape[1])
        
        boxes = np.squeeze(boxes).astype(int)
        scores = np.squeeze(scores)

        det_boxes = boxes[np.argwhere(scores>0.3).reshape(-1)]
        det_scores = scores[np.argwhere(scores>0.3).reshape(-1)]

        det_n = len(det_boxes)
        result = []
        classes = classes[0]

        if det_n > 0:
            for i in range(det_n):
                if int(classes[i]) == 1:
                    box = det_boxes[i] # xmin = box[0], ymin = box[2], xmax = box[1], ymax = box[3]                
                    bbox = [box[1], box[0], box[3], box[2]]
                    result.append(bbox)
            
        return result