import os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, torch.utils.data as data, argparse, \
    time
from datetime import timedelta
from models.peleenet import build_peleenet
from models.pelee_ssd import PeleeSSD
from data.config import voc
from data.voc0712 import VOCDetection
from layers.modules.multibox_loss import MultiBoxLoss
from utils.augmentations import Augmentation


def detection_collate(batch):
    targets, imgs = [], []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def kaiming_init(m):
    if hasattr(m, 'is_backbone'): return
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--dataset_root', default='VOCdevkit')
    args = parser.parse_args()

    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    backbone = build_peleenet()
    model = PeleeSSD(backbone, voc['num_classes'], voc)
    model.apply(kaiming_init).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(voc['num_classes'], 0.5, 3, voc['variance'])
    ds_criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=voc['max_iter'])

    start_iter = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iter = checkpoint['iteration']

    dataset = VOCDetection(root=args.dataset_root, transform=Augmentation(voc['min_dim']))
    loader = data.DataLoader(dataset, args.batch_size, num_workers=4, shuffle=True, collate_fn=detection_collate)

    model.train()
    iteration = start_iter
    total_start_time = time.time()

    while iteration < voc['max_iter']:
        for images, targets in loader:
            if iteration >= voc['max_iter']: break

            if iteration < 500:
                lr = (args.lr - 1e-6) * (iteration / 500) + 1e-6
                for pg in optimizer.param_groups: pg['lr'] = lr

            images = images.to(device)
            targets = [ann.to(device) for ann in targets]

            optimizer.zero_grad()
            loc, conf, p, ds1, ds2 = model(images)
            loss_l, loss_c = criterion((loc, conf, p), targets)

            ds_targets = torch.zeros(images.size(0), voc['num_classes'] - 1).to(device)
            for i, t in enumerate(targets):
                if t.size(0) > 0:
                    labels = t[:, -1].long()
                    ds_targets[i, labels] = 1.0

            loss_ds = ds_criterion(F.adaptive_avg_pool2d(ds1, 1).view(images.size(0), -1)[:, 1:], ds_targets) + \
                      ds_criterion(F.adaptive_avg_pool2d(ds2, 1).view(images.size(0), -1)[:, 1:], ds_targets)

            loss = loss_l + loss_c + 0.1 * loss_ds
            loss.backward()
            optimizer.step()

            if iteration >= 500: scheduler.step()

            if iteration % 10 == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - total_start_time
                avg_time = elapsed / (iteration - start_iter + 1)
                etc = timedelta(seconds=int(avg_time * (voc['max_iter'] - iteration)))

                print(f'Iter {iteration:6d} || Loss: {loss.item():.4f} || '
                      f'L: {loss_l.item():.4f} C: {loss_c.item():.4f} DS: {loss_ds.item():.4f} || '
                      f'LR: {curr_lr:.6f} || ETC: {etc}')

            if iteration != 0 and iteration % 5000 == 0:
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, f'weights/pelee_checkpoint_{iteration}.pth')
            iteration += 1

if __name__ == "__main__": train()