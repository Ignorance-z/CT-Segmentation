import albumentations
import cv2
from torch.utils.data import Dataset, DataLoader


from utils.utils import show_pics, data_preprocessing

# albumentations用于处理np.ndarray类型的数据，在这里经过处理之后大小缩小至256
SOURCE_SIZE = 512
TARGET_SIZE = 256

train_augs = albumentations.Compose([
    albumentations.Rotate(limit=360, p=0.9, border_mode=cv2.BORDER_REPLICATE),
    albumentations.RandomSizedCrop((int(SOURCE_SIZE * 0.75), SOURCE_SIZE),
                                   TARGET_SIZE,
                                   TARGET_SIZE,
                                   interpolation=cv2.INTER_NEAREST),
    albumentations.HorizontalFlip(p=0.5),

])

val_augs = albumentations.Compose([
    albumentations.Resize(TARGET_SIZE, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
])


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


if __name__ == '__main__':
    train_images, train_masks, val_images, val_masks = data_preprocessing(
        ['./data/images_medseg.npy', './data/masks_medseg.npy', './data/images_radiopedia.npy',
         './data/masks_radiopedia.npy']
    )
    train_dataset = CTdataSet(train_images, train_masks, augmentations=train_augs)
    val_dataset = CTdataSet(val_images, val_masks, augmentations=val_augs)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    for images, masks in train_loader:
        print(images.shape)
        show_pics(images, masks, 4)
        break