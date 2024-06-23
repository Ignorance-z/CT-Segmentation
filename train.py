from typing import List

import albumentations
import cv2
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from utils.dataset import CTdataSet, data_preprocessing
from utils.matrix import mIoU, pixel_accuracy
from utils.models.unet.model import UNet
from utils.painting import show_pics, show_results

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


def train(model, device, train_loader, val_loader, model_path, epochs, lr, net_cat=None) -> (
        tuple)[List[float], List[float], List[float], List[float], List[float], List[float]]:
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_iou, val_iou = [], []
    train_acc, val_acc = [], []
    best_loss, best_acc, best_iou = 1e6, 0, 0

    for epoch in range(epochs):
        # 控制图像输出
        flag = 1
        running_loss = 0
        iou_score = 0
        acc = 0
        model.train()
        for idx, (image, mask) in enumerate(train_loader):
            opt.zero_grad()
            image = image.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
            mask = mask.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)

            pred = model(image)
            loss = criterion(pred, mask)

            iou_score += mIoU(pred, mask, epoch)
            acc += pixel_accuracy(pred, mask)

            loss.backward()
            opt.step()

            running_loss += loss.item()

        else:
            model.eval()
            running_val_loss = 0
            running_val_acc = 0
            running_val_iou_score = 0
            with torch.no_grad():
                for idx, (image, mask) in enumerate(val_loader):
                    image = image.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
                    mask = mask.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
                    output = model(image)
                    running_val_iou_score += mIoU(output, mask, epoch)
                    running_val_acc += pixel_accuracy(output, mask)
                    loss = criterion(output, mask)
                    running_val_loss += loss.item()

                    if not idx and flag:
                        flag = 0
                        show_pics(
                            image=image.permute(0, 2, 3, 1).cpu().numpy(),
                            mask=output.permute(0, 2, 3, 1).detach().cpu().numpy(),
                            nums=8,
                            epoch=epoch,
                            net_cat=net_cat
                        )

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(running_val_loss / len(val_loader))
        val_iou.append(running_val_iou_score / len(val_loader))
        train_iou.append(iou_score / len(train_loader))
        train_acc.append(acc / len(train_loader))
        val_acc.append(running_val_acc / len(val_loader))

        print(
            f'[{epoch}/{epochs}] '
            f'train_loss: {running_loss / len(train_loader):.4f} '
            f'val_loss: {running_val_loss / len(val_loader):.4f} '
            f'train_iou: {iou_score / len(train_loader):.4f} '
            f'val_iou: {running_val_iou_score / len(val_loader):.4f} '
            f'train_acc: {acc / len(train_loader):.4f} '
            f'val_acc: {running_val_acc / len(val_loader):.4f}'
        )

        if (running_val_loss / len(val_loader) < best_loss and
                running_val_iou_score / len(val_loader) > best_iou and
                running_val_acc / len(val_loader) > best_acc):
            torch.save(model.state_dict(), model_path)

    return train_losses, val_losses, train_iou, val_iou, train_acc, val_acc


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
    train_losses, val_losses, train_iou, val_iou, train_acc, val_acc = (
        train(
            unet, device, train_loader, val_loader,
            model_path='./result/unet_state_dict.pth',
            epochs=1,
            lr=0.001,
            net_cat='UNet'
        )
    )
    show_results(train_losses, val_losses, train_iou, val_iou, train_acc, val_acc)