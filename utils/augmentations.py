import torch
import cv2
import numpy as np


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union


class PhotoMetricDistort(object):
    def __init__(self):
        self.pd = [
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue
        ]

    def brightness(self, im):
        if np.random.randint(2):
            delta = np.random.uniform(-32, 32)
            im += delta
        return im

    def contrast(self, im):
        if np.random.randint(2):
            alpha = np.random.uniform(0.5, 1.5)
            im *= alpha
        return im

    def saturation(self, im):
        if np.random.randint(2):
            im[:, :, 1] *= np.random.uniform(0.5, 1.5)
        return im

    def hue(self, im):
        if np.random.randint(2):
            im[:, :, 0] += np.random.uniform(-18, 18)
            im[:, :, 0][im[:, :, 0] > 360.0] -= 360.0
            im[:, :, 0][im[:, :, 0] < 0.0] += 360.0
        return im

    def __call__(self, im):
        im = im.astype(np.float32)
        im = self.brightness(im)
        if np.random.randint(2):
            distort = self.contrast
            ops = [self.saturation, self.hue]
        else:
            distort = None
            ops = [self.saturation, self.hue, self.contrast]

        if distort:
            im = distort(im)

        im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        for op in ops:
            if op != self.contrast:
                im = op(im)
        im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)

        if not distort:
            im = self.contrast(im)

        return np.clip(im, 0, 255)


class Augmentation(object):
    def __init__(self, size=304, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.size = size
        self.distort = PhotoMetricDistort()

    def __call__(self, image, boxes, labels):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.distort(image)

        image = image.astype(np.float32) / 255.0

        if np.random.randint(2):
            image, boxes = self._expand(image, boxes)

        image, boxes, labels = self._random_crop(image, boxes, labels)

        if np.random.randint(2):
            image, boxes = self._mirror(image, boxes)

        image = cv2.resize(image, (self.size, self.size))
        image = (image - self.mean) / self.std

        return image, np.clip(boxes, 0, 1), labels

    def _mirror(self, img, boxes):
        img = img[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = 1.0 - boxes[:, 2::-2]
        return img, boxes

    def _expand(self, img, boxes):
        h, w, c = img.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, w * ratio - w)
        top = np.random.uniform(0, h * ratio - h)
        expand_img = np.full((int(h * ratio), int(w * ratio), c), self.mean, dtype=img.dtype)
        expand_img[int(top):int(top + h), int(left):int(left + w)] = img
        boxes = boxes.copy()
        boxes[:, :2] = (boxes[:, :2] * [w, h] + [int(left), int(top)]) / [w * ratio, h * ratio]
        boxes[:, 2:] = (boxes[:, 2:] * [w, h] + [int(left), int(top)]) / [w * ratio, h * ratio]
        return expand_img, boxes

    def _random_crop(self, image, boxes, labels):
        height, width, _ = image.shape
        for _ in range(50):
            mode = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
            if mode is None: return image, boxes, labels

            w = np.random.uniform(0.3 * width, width)
            h = np.random.uniform(0.3 * height, height)
            if h / w < 0.5 or h / w > 2: continue

            left = np.random.uniform(0, width - w)
            top = np.random.uniform(0, height - h)
            rect = np.array([int(left), int(top), int(left + w), int(top + h)])

            overlap = jaccard_numpy(boxes * [width, height, width, height], rect)
            if overlap.min() < mode: continue

            current_image = image[rect[1]:rect[3], rect[0]:rect[2], :]
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0 * [width, height]
            m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
            m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
            mask = m1 * m2
            if not mask.any(): continue

            current_boxes = boxes[mask].copy()
            current_labels = labels[mask]
            current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2] / [width, height])
            current_boxes[:, :2] = (current_boxes[:, :2] * [width, height] - rect[:2]) / [w, h]
            current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:] / [width, height])
            current_boxes[:, 2:] = (current_boxes[:, 2:] * [width, height] - rect[:2]) / [w, h]

            return current_image, current_boxes, current_labels
        return image, boxes, labels