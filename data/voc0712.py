import os
import torch
import torch.utils.data as data
import cv2
import xml.etree.ElementTree as ET
import numpy as np

class VOCDetection(data.Dataset):
    def __init__(self, root: str, image_sets: list = [('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=None):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index: int):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index: int):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        res = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)

            label_idx = VOC_CLASSES.index(name)
            bndbox.append(label_idx)
            res += [bndbox]

        if self.transform is not None:
            target = np.array(res)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = torch.from_numpy(img).permute(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, target, height, width

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)