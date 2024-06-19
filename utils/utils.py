import random
import re
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt


# 显示数据集
def show_pics(image, mask, nums):  # TODO 这里需要考虑进一步完善
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

    plt.tight_layout()
    plt.show()


# # 计算图像数据集的平均值和方差，便于进行后续的图像正则化操作
# def calculate_mean_std(*args) -> tuple[float, float]:   # TODO 这个函数目前不使用，后续需要进行完善
#     data = np.concatenate(args, axis=0)
#     mean = torch.tensor(data).mean(dim=[0, 1, 2])
#     std = torch.tensor(data).std(dim=[0, 1, 2])
#     return mean.item(), std.item()


# def