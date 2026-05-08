import os
import glob
import cv2
import numpy as np
from PIL import Image

import torch

from skimage.metrics import structural_similarity as ssim

from model import UVDocNet, dewarp_image


DATASET_PATH = "renders/synthetic_data_pitch_sweep"

IMAGE_SIZE = 256

OUTPUT_DIR = "outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_rgb(path):

    img = Image.open(path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    img = np.array(img).astype(np.float32) / 255.0

    img = torch.tensor(img).permute(2, 0, 1)

    return img


model = UVDocNet().to(DEVICE)

model.load_state_dict(
    torch.load("best_model.pth")
)

model.eval()

rgb_paths = sorted(
    glob.glob(
        os.path.join(DATASET_PATH, "rgb", "*.jpg")
    )
)

gt_paths = sorted(
    glob.glob(
        os.path.join(DATASET_PATH, "ground_truth", "*.png")
    )
)

ssim_scores = []


with torch.no_grad():

    for i, path in enumerate(rgb_paths[:10]):

        rgb = load_rgb(path).unsqueeze(0).to(DEVICE)

        gt = load_rgb(gt_paths[i]).unsqueeze(0).to(DEVICE)

        pred_uv = model(rgb)

        rectified = dewarp_image(
            rgb,
            pred_uv
        )

        rgb_np = rgb[0].permute(1, 2, 0).cpu().numpy()

        uv_np = pred_uv[0].permute(1, 2, 0).cpu().numpy()

        rect_np = rectified[0].permute(1, 2, 0).cpu().numpy()

        gt_np = gt[0].permute(1, 2, 0).cpu().numpy()

        uv_vis = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

        uv_vis[:, :, :2] = uv_np

        comparison = np.concatenate(
            [rgb_np, rect_np, gt_np],
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

        rect_gray = cv2.cvtColor(
            (rect_np * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )

        gt_gray = cv2.cvtColor(
            (gt_np * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )

        score = ssim(
            gt_gray,
            rect_gray,
            data_range=255
        )

        ssim_scores.append(score)

        print(f"Image {i} SSIM: {score:.4f}")

avg_ssim = np.mean(ssim_scores)

print(f"\nAverage SSIM: {avg_ssim:.4f}")