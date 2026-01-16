import torch
from torchvision import transforms
import cv2
import numpy as np


class Augmentation(object):
    def __init__(self, size: int = 304, mean: tuple = (104, 117, 123)):
        self.mean = mean
        self.size = size

    def __call__(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        height, width, _ = image.shape
        boxes[:, 0::2] *= width
        boxes[:, 1::2] *= height

        image = image.astype(np.float32)

        if np.random.randint(2):
            image = self._distort(image)

        if np.random.randint(2):
            image, boxes = self._expand(image, boxes, self.mean)

        if np.random.randint(2):
            image, boxes, labels = self._crop(image, boxes, labels)

        if np.random.randint(2):
            image, boxes = self._mirror(image, boxes)

        image = cv2.resize(image, (self.size, self.size))

        new_h, new_w, _ = image.shape
        boxes[:, 0::2] /= new_w
        boxes[:, 1::2] /= new_h

        image -= self.mean
        return image, boxes, labels

    def _distort(self, img):
        return img * (np.random.uniform(0.7, 1.3))

    def _mirror(self, img, boxes):
        _, width, _ = img.shape
        img = img[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        return img, boxes

    def _expand(self, img, boxes, mean):
        h, w, c = img.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, w * ratio - w)
        top = np.random.uniform(0, h * ratio - h)
        expand_img = np.zeros((int(h * ratio), int(w * ratio), c), dtype=img.dtype)
        expand_img[:, :, :] = mean
        expand_img[int(top):int(top + h), int(left):int(left + w)] = img
        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        return expand_img, boxes

    def _crop(self, image, boxes, labels):
        height, width, _ = image.shape

        for _ in range(50):
            mode = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
            if mode is None:
                return image, boxes, labels

            for _ in range(50):
                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                if h / w < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(0, width - w)
                top = np.random.uniform(0, height - h)

                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2

                if not mask.any():
                    continue

                current_boxes = boxes[mask].copy()
                current_labels = labels[mask]

                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:4] = np.minimum(current_boxes[:, 2:4], rect[2:4])
                current_boxes[:, 2:4] -= rect[:2]

                return image[rect[1]:rect[3], rect[0]:rect[2]], current_boxes, current_labels

        return image, boxes, labels