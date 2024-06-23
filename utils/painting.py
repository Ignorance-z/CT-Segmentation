import random
import re
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt


# 显示数据集
def show_pics(image, mask, nums, epoch=None, net_cat=None):  # TODO 这里需要考虑进一步完善
    mask_num = mask.shape[-1] if mask is not None else 0
    fig, ax = plt.subplots(5, nums)
    for i in range(nums):
        ax_img = ax[0, i]
        ax_img.imshow(image[i], cmap='gray')
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        if mask is not None:
            for j in range(mask_num):
                ax_mask = mask[i, :, :, j]
                ax[j + 1, i].imshow(ax_mask)
                ax[j + 1, i].set_xticks([])
                ax[j + 1, i].set_yticks([])

    print(epoch)
    print(epoch is not None)
    plt.tight_layout()
    if epoch is not None:
        plt.savefig(f'../pic/epoch_{epoch}_{net_cat}.png')
    else:
        plt.show()


# # 计算图像数据集的平均值和方差，便于进行后续的图像正则化操作
# def calculate_mean_std(*args) -> tuple[float, float]:   # TODO 这个函数目前不使用，后续需要进行完善
#     data = np.concatenate(args, axis=0)
#     mean = torch.tensor(data).mean(dim=[0, 1, 2])
#     std = torch.tensor(data).std(dim=[0, 1, 2])
#     return mean.item(), std.item()

def plot_loss(train_losses: List[float], val_losses: List[float]):
    plt.plot(train_losses, label='train', marker='o')
    plt.plot( val_losses, label='val', marker='o')
    plt.title('Train and Val loss per epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()


def plot_acc(train_acc: List[float], val_acc: List[float]):
    plt.plot(train_acc, label='train', marker='o')
    plt.plot( val_acc, label='val', marker='o')
    plt.title('Train and Val acc per epoch')
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()


def plot_miou(train_iou: List[float], val_iou: List[float]):
    plt.plot(train_iou, label='train', marker='o')
    plt.plot( val_iou, label='val', marker='o')
    plt.title('Train and Val iou per epochs')
    plt.ylabel('iou')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()


def show_results(train_losses, val_losses, train_iou, val_iou, train_acc, val_acc):
    plot_loss(train_losses, val_losses)
    plot_acc(train_acc, val_acc)
    plot_miou(train_iou, val_iou)