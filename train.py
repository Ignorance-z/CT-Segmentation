import albumentations
import cv2
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from utils.dataset import CTdataSet, data_preprocessing
from utils.models.unet.model import UNet
from utils.utils import show_pics

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


def train(model, device, train_loader, val_loader, model_path, batch_size, epochs, lr):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        for image, label in train_loader:
            opt.zero_grad()
            image = image.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
            label = label.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)

            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()
            opt.step()
            print('epoch: {:d}， loss: {:f}'.format(epoch, loss.item()))

            # 这两步实际上是获取每层每个像素点对应的类别（0或者1）
            # pred = torch.sigmoid(pred)
            # output = (pred > 0.5).int()

            break


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_images, train_masks, val_images, val_masks = data_preprocessing(
        ['./data/images_medseg.npy', './data/masks_medseg.npy', './data/images_radiopedia.npy',
         './data/masks_radiopedia.npy']
    )
    train_dataset = CTdataSet(train_images, train_masks, augmentations=train_augs)
    val_dataset = CTdataSet(val_images, val_masks, augmentations=val_augs)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    unet = UNet(n_channels=1, n_classes=4, bilinear=False)
    train(unet, device, train_loader, val_loader, model_path='unet.pth', epochs=2, batch_size=1, lr=0.0001)