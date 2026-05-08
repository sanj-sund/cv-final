import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from model import UVDocNet, dewarp_image


DATASET_PATH = "renders/synthetic_data_pitch_sweep"

IMAGE_SIZE = 256
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


def masked_l1(pred, target, mask):

    loss = torch.abs(pred - target)

    loss = loss * mask

    return loss.mean()


def smoothness_loss(flow):

    dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])

    dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])

    return dx.mean() + dy.mean()


dataset = DocDataset(DATASET_PATH)

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