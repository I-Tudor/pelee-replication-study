import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import time
from typing import Tuple, List

from models.peleenet import build_peleenet
from models.pelee_ssd import PeleeSSD
from data.config import voc
from data.voc0712 import VOCDetection
from layers.modules.multibox_loss import MultiBoxLoss
from utils.augmentations import Augmentation


def xavier_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def train():
    parser = argparse.ArgumentParser(description='Pelee SSD Training')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=4e-3, type=float)
    parser.add_argument('--dataset_root', default='VOCdevkit', type=str)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("==> Hardware Found: Apple Silicon (MPS). Optimizing memory buffers...")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print("==> Hardware Found: NVIDIA (CUDA). Enabling CuDNN benchmarking...")
    else:
        device = torch.device("cpu")
        print("==> Hardware Found: CPU. Training will be slow.")

    print("==> Building Pelee SSD...")
    backbone = build_peleenet(num_classes=1000)
    model = PeleeSSD(backbone, voc['num_classes'], voc)
    model.apply(xavier_init)
    model.to(device)

    print("==> Loading Dataset...")
    dataset = VOCDetection(root=args.dataset_root, transform=Augmentation(voc['min_dim']))

    # Auto-optimize Data Loading
    # num_workers=0 avoids the macOS multiprocessing hang
    data_loader = data.DataLoader(
        dataset,
        args.batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=detection_collate,
        pin_memory=False,
        persistent_workers=True,
        multiprocessing_context="forkserver"
    )

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(voc['num_classes'], 0.5, 3, voc['variance'])

    # Cosine Annealing scheduler as per Pelee paper
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=voc['max_iter'])

    if not os.path.exists('weights'):
        os.mkdir('weights')

    print(f"==> Setup Complete. Starting Training on {len(dataset)} images...")
    model.train()

    iteration = 0
    while iteration < voc['max_iter']:
        for images, targets in data_loader:
            if iteration >= voc['max_iter']:
                break

            t0 = time.time()

            images = images.to(device)
            targets = [ann.to(device) for ann in targets]

            # Forward and Backward
            optimizer.zero_grad()
            out = model(images)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            scheduler.step()

            t1 = time.time()

            if iteration % 10 == 0:
                print(f'Iter {iteration:5d} || Loss: {loss.item():.4f} || ' +
                      f'Conf: {loss_c.item():.4f} Loc: {loss_l.item():.4f} || ' +
                      f'Time: {t1 - t0:.4f}s || LR: {scheduler.get_last_lr()[0]:.6f}')

            if iteration != 0 and iteration % 5000 == 0:
                print(f'==> Saving Checkpoint at iteration {iteration}')
                torch.save(model.state_dict(), f'weights/pelee_ssd_{iteration}.pth')

            iteration += 1


if __name__ == "__main__":
    train()