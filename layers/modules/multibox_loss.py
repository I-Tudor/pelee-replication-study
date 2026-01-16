import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes: int, overlap_thresh: float, neg_pos: int, variance: list):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = variance

    def forward(self, predictions: tuple, targets: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        num_priors = priors.size(0)
        device = loc_data.device

        loc_t = torch.zeros(num, num_priors, 4).to(device)
        conf_t = torch.zeros(num, num_priors).to(device).long()

        for idx in range(num):
            truths = targets[idx][:, :-1].to(device)
            labels = targets[idx][:, -1].long().to(device)
            self._match(self.threshold, truths, priors.to(device), self.variance, labels, loc_t, conf_t, idx)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        pos_idx = pos.unsqueeze(2).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = F.cross_entropy(batch_conf, conf_t.view(-1), reduction='none')
        loss_c = loss_c.view(num, -1)

        loss_c[pos] = 0
        _, idx = loss_c.sort(1, descending=True)
        _, rank = idx.sort(1)

        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = rank < num_neg.expand_as(rank)

        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        N = num_pos.data.sum().float()
        if N == 0:
            return torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

        return loss_l / N, loss_c / N

    def _match(self, threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
        overlaps = self._intersect(truths, self._point_form(priors))
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_prior_idx.squeeze_(1)
        best_prior_overlap.squeeze_(1)
        best_truth_overlap.index_fill_(0, best_prior_idx, 2)
        for j in range(best_prior_idx.size(0)):
            best_truth_idx[best_prior_idx[j]] = j
        matches = truths[best_truth_idx]
        conf = labels[best_truth_idx] + 1
        conf[best_truth_overlap < threshold] = 0
        loc = self._encode(matches, priors, variances)
        loc_t[idx] = loc
        conf_t[idx] = conf

    def _intersect(self, box_a, box_b):
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def _point_form(self, boxes):
        return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)

    def _encode(self, matched, priors, variances):
        g_cxcy = (matched[:, :2] + matched[:, 2:4]) / 2 - priors[:, :2]
        g_cxcy /= (variances[0] * priors[:, 2:])
        g_wh = (matched[:, 2:4] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh + 1e-10) / variances[1]
        return torch.cat([g_cxcy, g_wh], 1)