import numpy as np
import torch
import torch.nn.functional as F


def mIoU(pred, mask, epoch, smooth=1e-10, n_classes=4):
    with torch.no_grad():
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1).contiguous().view(-1)

        mask = torch.argmax(mask, dim=1).contiguous().view(-1)

        # if epoch == 0:
        #     np.savetxt("pred.txt", pred.cpu().numpy(), fmt='%d')
        #     np.savetxt("mask.txt", mask.cpu().numpy(), fmt='%d')

        iou_per_class = []
        for clas in range(n_classes):
            true_class = pred == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:
                iou_per_class.append(float('nan'))
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def pixel_accuracy(pred, mask) -> float:
    with torch.no_grad():
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        mask = torch.argmax(mask, dim=1)
        correct = torch.eq(pred, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy