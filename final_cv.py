# deep_document_unwarping.py

import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import timm

DATASET_PATH = "renders/synthetic_data_pitch_sweep"

IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-4

OUTPUT_DIR = "outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_rgb(path):

    img = Image.open(path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    img = np.array(img).astype(np.float32) / 255.0

    img = torch.tensor(img).permute(2, 0, 1)

    return img


def load_uv(path):

    uv = Image.open(path).convert("RGB")
    uv = uv.resize((IMAGE_SIZE, IMAGE_SIZE))

    uv = np.array(uv).astype(np.float32) / 255.0

    uv = uv[:, :, :2]

    uv = torch.tensor(uv).permute(2, 0, 1)

    return uv


def load_mask(path):

    mask = Image.open(path).convert("L")
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE))

    mask = np.array(mask).astype(np.float32) / 255.0

    mask = torch.tensor(mask).unsqueeze(0)

    return mask


class DocDataset(Dataset):

    def __init__(self, root):

        self.rgb_paths = sorted(
            glob.glob(os.path.join(root, "rgb", "*.jpg"))
        )

        self.gt_paths = sorted(
            glob.glob(os.path.join(root, "ground_truth", "*.png"))
        )

        self.uv_paths = sorted(
            glob.glob(os.path.join(root, "uv", "*.png"))
        )

        self.mask_paths = sorted(
            glob.glob(os.path.join(root, "border", "*.png"))
        )

    def __len__(self):

        return len(self.rgb_paths)

    def __getitem__(self, idx):

        rgb = load_rgb(self.rgb_paths[idx])

        gt = load_rgb(self.gt_paths[idx])

        uv = load_uv(self.uv_paths[idx])

        mask = load_mask(self.mask_paths[idx])

        return rgb, gt, uv, mask


class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class UpBlock(nn.Module):

    def __init__(self, in_c, skip_c, out_c):

        super().__init__()

        self.conv = ConvBlock(in_c + skip_c, out_c)

    def forward(self, x, skip):

        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class UVDocNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = timm.create_model(
            "resnet34",
            pretrained=True,
            features_only=True
        )

        channels = self.encoder.feature_info.channels()

        self.up1 = UpBlock(channels[4], channels[3], 256)

        self.up2 = UpBlock(256, channels[2], 128)

        self.up3 = UpBlock(128, channels[1], 64)

        self.up4 = UpBlock(64, channels[0], 32)

        self.final = nn.Sequential(

            ConvBlock(32, 32),

            nn.Conv2d(32, 2, kernel_size=1),

            nn.Sigmoid()
        )

    def forward(self, x):

        feats = self.encoder(x)

        f1, f2, f3, f4, f5 = feats

        x = self.up1(f5, f4)

        x = self.up2(x, f3)

        x = self.up3(x, f2)

        x = self.up4(x, f1)

        uv = self.final(x)

        return uv


def dewarp_image(rgb, uv):

    grid = uv.permute(0, 2, 3, 1)

    grid = (grid * 2.0) - 1.0

    rectified = F.grid_sample(
        rgb,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return rectified


def masked_l1(pred, target, mask):

    # Resize pred to match target size if needed
    if pred.shape != target.shape:
        pred = F.interpolate(
            pred,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=False
        )
    
    loss = torch.abs(pred - target)
    
    # Expand mask to match pred/target dimensions if needed
    if mask.shape[1] == 1 and loss.shape[1] > 1:
        mask = mask.expand_as(loss)

    loss = loss * mask

    return loss.mean()


def smoothness_loss(flow):

    dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])

    dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])

    return dx.mean() + dy.mean()


dataset = DocDataset(DATASET_PATH)
print(dataset)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = UVDocNet().to(DEVICE)

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

best_loss = 999999


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    loop = tqdm(loader)

    for rgb, gt, uv_gt, mask in loop:

        rgb = rgb.to(DEVICE)

        gt = gt.to(DEVICE)

        uv_gt = uv_gt.to(DEVICE)

        mask = mask.to(DEVICE)

        pred_uv = model(rgb)

        rectified = dewarp_image(
            rgb,
            pred_uv
        )

        uv_loss = masked_l1(
            pred_uv,
            uv_gt,
            mask
        )

        img_loss = masked_l1(
            rectified,
            gt,
            mask
        )

        smooth_loss = smoothness_loss(pred_uv)

        loss = (
            uv_loss
            + 0.5 * img_loss
            + 0.05 * smooth_loss
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        loop.set_postfix(
            loss=loss.item()
        )

    avg_loss = running_loss / len(loader)

    print(f"Epoch {epoch+1}: {avg_loss:.4f}")

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )


model.eval()

rgb_paths = sorted(
    glob.glob(
        os.path.join(DATASET_PATH, "rgb", "*.jpg")
    )
)

with torch.no_grad():

    for i, path in enumerate(rgb_paths[:10]):

        rgb = load_rgb(path).unsqueeze(0).to(DEVICE)

        pred_uv = model(rgb)

        rectified = dewarp_image(
            rgb,
            pred_uv
        )

        rgb_np = rgb[0].permute(1, 2, 0).cpu().numpy()

        uv_np = pred_uv[0].permute(1, 2, 0).cpu().numpy()

        rect_np = rectified[0].permute(1, 2, 0).cpu().numpy()

        uv_vis = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

        uv_vis[:, :, :2] = uv_np

        comparison = np.concatenate(
            [rgb_np, rect_np],
            axis=1
        )

        cv2.imwrite(
            f"{OUTPUT_DIR}/predicted_uv_{i}.png",
            (uv_vis * 255).astype(np.uint8)
        )

        cv2.imwrite(
            f"{OUTPUT_DIR}/rectified_{i}.png",
            (rect_np * 255).astype(np.uint8)
        )

        cv2.imwrite(
            f"{OUTPUT_DIR}/comparison_{i}.png",
            (comparison * 255).astype(np.uint8)
        )