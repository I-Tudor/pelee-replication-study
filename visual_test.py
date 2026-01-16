import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import random
import xml.etree.ElementTree as ET
from models.peleenet import build_peleenet
from models.pelee_ssd import PeleeSSD
from data.config import voc
from data.voc0712 import VOCDetection, VOC_CLASSES


def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, overlap=0.45, top_k=200):
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    _, idx = scores.sort(0)
    idx = idx[-top_k:]

    keep = []
    while idx.numel() > 0:
        i = idx[-1].item()
        keep.append(i)
        if idx.size(0) == 1:
            break
        idx = idx[:-1]

        xx1 = x1[idx].clamp(min=x1[i])
        yy1 = y1[idx].clamp(min=y1[i])
        xx2 = x2[idx].clamp(max=x2[i])
        yy2 = y2[idx].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0.0)
        h = (yy2 - yy1).clamp(min=0.0)
        inter = w * h

        rem_areas = area[idx]
        union = (rem_areas - inter) + area[i]
        iou = inter / union
        idx = idx[iou.le(overlap)]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def visualize():
    parser = argparse.ArgumentParser(description='Pelee SSD Visual Test')
    parser.add_argument('--weights', default='weights/pelee_ssd_iteration.pth', type=str)
    parser.add_argument('--root', default='VOCdevkit', type=str)
    parser.add_argument('--thresh', default=0.1, type=float)
    parser.add_argument('--top_k', default=5, type=int)
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    backbone = build_peleenet(num_classes=1000)
    net = PeleeSSD(backbone, voc['num_classes'], voc)

    state_dict = torch.load(args.weights, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device).eval()

    dataset = VOCDetection(root=args.root, image_sets=[('2007', 'test')], transform=None)

    idx = random.randint(0, len(dataset) - 1)
    img, gt, orig_h, orig_w = dataset.pull_item(idx)
    display_img = img.copy()

    if isinstance(gt, ET.Element) or hasattr(gt, 'findall'):
        for obj in gt.findall('object'):
            bbox = obj.find('bndbox')
            gx1 = int(bbox.find('xmin').text)
            gy1 = int(bbox.find('ymin').text)
            gx2 = int(bbox.find('xmax').text)
            gy2 = int(bbox.find('ymax').text)
            cv2.rectangle(display_img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
    else:
        for box_data in gt:
            gx1, gy1, gx2, gy2 = int(box_data[0] * orig_w), int(box_data[1] * orig_h), \
                int(box_data[2] * orig_w), int(box_data[3] * orig_h)
            cv2.rectangle(display_img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)

    img_prep = cv2.resize(img, (304, 304)).astype(np.float32)
    img_prep -= np.array((104, 117, 123))
    x = torch.from_numpy(img_prep).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        loc, conf, priors = net(x)

    priors = priors.to(device)
    decoded_boxes = decode(loc[0], priors, voc['variance'])
    conf_scores = F.softmax(conf[0], dim=-1).transpose(1, 0)

    all_scores, all_indices = conf_scores[1:].max(dim=0)
    top_scores, top_idx_in_priors = all_scores.topk(args.top_k)

    print(f"Top {args.top_k} confidences: {top_scores.cpu().numpy()}")

    for i in range(args.top_k):
        score = top_scores[i].item()
        prior_idx = top_idx_in_priors[i].item()
        class_idx = all_indices[prior_idx].item() + 1

        box = decoded_boxes[prior_idx].cpu().numpy()

        box[0::2] *= orig_w
        box[1::2] *= orig_h
        x1, y1, x2, y2 = box.astype(np.int32)

        color = (0, 255, 0) if score > args.thresh else (0, 255, 255)

        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
        label = f"{VOC_CLASSES[class_idx - 1]} {score:.2f}"
        cv2.putText(display_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.imwrite("visual_check.jpg", display_img)
    print(f"Saved visual_check.jpg. Red=GT, Green/Yellow=Model.")


if __name__ == '__main__':
    visualize()