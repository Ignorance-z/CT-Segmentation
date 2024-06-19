import random
import re
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt


# 进行数据集加载和训练验证集划分
def data_preprocessing(paths: List[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 使用正则表达式对文件名进行匹配
    pattern_images = re.compile(r'.*images.*\.npy')
    pattern_masks = re.compile(r'.*masks.*\.npy')

    # 加载数据
    images = [np.load(filename).astype(np.float32) for filename in paths if pattern_images.match(filename)]
    masks = [np.load(filename).astype(np.int8) for filename in paths if pattern_masks.match(filename)]
    images = np.concatenate(images, axis=0)
    masks = np.concatenate(masks, axis=0)

    index = list(range(len(images)))
    random.shuffle(index)

    train_images, val_images = images[index[:int(0.8 * len(images))]], images[index[int(0.8 * len(images)):]]
    train_masks, val_masks = masks[index[:int(0.8 * len(images))]], masks[index[int(0.8 * len(images)):]]

    return train_images, train_masks, val_images, val_masks


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


