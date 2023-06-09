#!/usr/bin/env python3
""" Task 0. Initialize Yolo """


import tensorflow.keras as K


class Yolo:
    """
    Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Initialize Yolo """

        self.model = K.models.load_model(model_path)
        with open(classes_path) as f:
            self.class_names = [i.strip() for i in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
