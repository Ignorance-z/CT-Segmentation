import random
import re
from typing import List

import numpy as np
from torch.utils.data import Dataset, DataLoader


from utils.utils import show_pics


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


class CTdataSet(Dataset):
    def __init__(self, images, masks, augmentations=None):
        self.images = images
        self.masks = masks
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.augmentations:
            sample = self.augmentations(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask


# if __name__ == '__main__':
#     train_images, train_masks, val_images, val_masks = data_preprocessing(
#         ['./data/images_medseg.npy', './data/masks_medseg.npy', './data/images_radiopedia.npy',
#          './data/masks_radiopedia.npy']
#     )
#     train_dataset = CTdataSet(train_images, train_masks, augmentations=train_augs)
#     val_dataset = CTdataSet(val_images, val_masks, augmentations=val_augs)
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
#
#     for images, masks in train_loader:
#         print(images.shape)
#         show_pics(images, masks, 4)
#         break