#!/usr/bin/env python3
""" Task 2. Process Outputs """


import tensorflow.keras as K
import numpy as np


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

    def process_outputs(self, outputs, image_size):
        """
        outputs is a list of numpy.ndarrays containing the predictions
        from the Darknet model for a single image

        Returns a tuple of (boxes, box_confidences, box_class_probs)
        """

        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            boxes.append(output[..., 0:4])
            box_confidences.append(output[..., 4:5])
            box_class_probs.append(output[..., 5:])

        for i, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape

            c = np.zeros((grid_height, grid_width, anchor_boxes), dtype=int)

            idx_y = np.arange(grid_height)
            idx_y = idx_y.reshape(grid_height, 1, 1)

            idx_x = np.arange(grid_width)
            idx_x = idx_x.reshape(1, grid_width, 1)

            Cx = c + idx_x
            Cy = c + idx_y

            centx = (box[..., 0])
            centy = (box[..., 1])
            bx = ((1 / (1 + np.exp(-centx))) + Cx) / grid_width
            by = ((1 / (1 + np.exp(-centy))) + Cy) / grid_height

            tx = (box[..., 2])
            ty = (box[..., 3])
            tw = np.exp(tx) * self.anchors[i, :, 0] /\
                self.model.input.shape[1].value
            th = np.exp(ty) * self.anchors[i, :, 1] /\
                self.model.input.shape[2].value

            x1 = (bx - tw / 2) * image_width
            y1 = (by - th / 2) * image_height
            x2 = (bx + tw / 2) * image_width
            y2 = (by + th / 2) * image_height

            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

        for i, box in enumerate(box_confidences):
            box_confidences[i] = (1 / (1 + np.exp(-box)))
        for i, box in enumerate(box_class_probs):
            box_class_probs[i] = (1 / (1 + np.exp(-box)))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Returns a tuple of (filtered_boxes, box_classes, box_scores)
        """

        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i, box in enumerate(boxes):
            box_conf = box_confidences[i]
            box_class_prob = box_class_probs[i]

            box_scores_per_ouput = box_conf * box_class_prob
            box_class = np.argmax(box_scores_per_ouput, axis=-1)
            box_score = np.max(box_scores_per_ouput, axis=-1)

            mask = box_score >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores
