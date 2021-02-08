import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.tensorrt as trt
import time
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from src.usb_camera import USBCamera
from src.object_detector import ObjectDetection

""" Jetson Live Object Detector """
class JetsonLiveObjectDetection():
    def __init__(self, model, debug=False, fps = 10.):
        self.debug = debug
        self.camera = USBCamera(300, 300)
        self.model = model
        self.rate = float(1. / fps)
        self.detector = ObjectDetection('./data/' + self.model)

    def _visualizeDetections(self, img, scores, boxes, classes, num_detections):
        cols = img.shape[1]
        rows = img.shape[0]
        detections = []

        for i in range(num_detections):
            bbox = [float(p) for p in boxes[i]]
            score = float(scores[i])
            classId = int(classes[i])
            if score > 0.5:
                x = int(bbox[1] * cols)
                y = int(bbox[0] * rows)
                right = int(bbox[3] * cols)
                bottom = int(bbox[2] * rows)
                thickness = int(4 * score)
                cv2.rectangle(img, (x, y), (right, bottom), (125,255, 21), thickness=thickness)
                detections.append(self.detector.labels[str(classId)])

        print ("Debug: Found objects: " + str(' '.join(detections)) + ".")

        cv2.imshow('Detection', img)

    def start(self):
        print ("Starting Live object detection, may take a few minutes to initialize...")
        #self.camera.startStreaming()
        self.detector.initializeSession()

        if not self.camera.isOpened():
            print ("Camera has failed to open")
            exit(-1)
        elif self.debug:
            cv2.namedWindow("Detection", cv2.WINDOW_AUTOSIZE)
    
        while True:
            curTime = time.time()

            img = self.camera.getFrame()
            img = img[:, :, 0:3]
            (boxes, scores, classes) = self.detector.detect(img)
            det_boxes = boxes[np.argwhere(scores>0.3).reshape(-1)]
            det_scores = scores[np.argwhere(scores>0.3).reshape(-1)]
    
            #scores, boxes, classes, num_detections = self.detector.detect(img)

            if len(det_boxes) > 0:
                for i in range(len(det_boxes)):
                    box = det_boxes[i]
                    cropped = img[box[0]:box[2], box[1]:box[3], :]
                    cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

            sec = time.time() - curTime

            if sec != 0:
                fps = 1 / (sec)
                str = 'FPS: %0.1f' % fps
                text_fps_x = len(img[0]) - 150
                text_fps_y = 20
                cv2.putText(img, str, (text_fps_x, text_fps_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            
            cv2.imshow('Detection', img)

            if cv2.waitKey(1) == ord('q'):
                break
            
        cv2.destroyAllWindows()
        self.camera.__del__()
        self.detector.__del__()
        print ("Exiting...")
        return



if __name__ == "__main__":
    debug = True
    model = 'ssd_mobilenet_v1_coco_trt_graph.pb'
    
    live_detection = JetsonLiveObjectDetection(model=model, debug=debug)
    live_detection.start()
    

