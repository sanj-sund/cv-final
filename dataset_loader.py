"""
Document Reconstruction Dataset Loader
=======================================

This script provides a PyTorch dataset loader for the synthetic document dataset.
Focus on implementing the reconstruction model and training loop.

IMPORTANT: FOCUS ON GEOMETRIC RECONSTRUCTION, NOT PHOTOMETRIC MATCHING!
-----------------------------------------------------------------------
The rendered images include realistic lighting effects (shadows, shading, specular
highlights). Even with perfect geometric dewarping, the lighting will NOT match
the flat ground truth. This is expected and acceptable!

Your goal: Learn the geometric transformation (UV mapping / flow field)
NOT your goal: Match pixel intensities exactly (lighting is different!)

Recommended approach:
- Use SSIMLoss (focuses on structure, robust to lighting)
- Predict flow fields with grid_sample (explicit geometric reasoning)
- Evaluate with SSIM (structure) not just PSNR (pixels)
- Use border masks to focus on document geometry

Dataset Structure:
    renders/synthetic_data_pitch_sweep/
        ├── rgb/           # Input images (warped documents with backgrounds)
        ├── ground_truth/  # Target images (flat paper - DIFFERENT LIGHTING!)
        ├── depth/         # Depth maps (optional, for advanced methods)
        ├── uv/           # UV maps (optional, for advanced methods)
        └── border/       # Border masks (optional, for advanced methods)

Usage Example:
    from dataset_loader import DocumentDataset, get_dataloaders

    # Quick start - get train and val dataloaders
    train_loader, val_loader = get_dataloaders(
        data_dir='renders/synthetic_data_pitch_sweep',
        batch_size=8,
        train_split=0.8
    )

    # Or create custom dataset
    dataset = DocumentDataset(
        data_dir='renders/synthetic_data_pitch_sweep',
        use_depth=True,
        use_uv=False,
        transform=None
    )
"""

import os
import glob
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np


class DocumentDataset(Dataset):
    """
    PyTorch Dataset for document reconstruction.

    Loads RGB images of warped documents and their corresponding ground truth
    flat textures. Optionally loads depth maps, UV maps, and border masks.

    Args:
        data_dir: Root directory containing the dataset
        use_depth: Whether to load depth maps
        use_uv: Whether to load UV maps
        use_border: Whether to load border masks
        transform: Optional transform to apply to images
        img_size: Tuple of (height, width) to resize images to
    """

    def __init__(
        self,
        data_dir: str,
        use_depth: bool = False,
        use_uv: bool = False,
        use_border: bool = False,
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = (512, 512)
    ):
        self.data_dir = Path(data_dir)
        self.use_depth = use_depth
        self.use_uv = use_uv
        self.use_border = use_border
        self.transform = transform
        self.img_size = img_size

        # Find all RGB images
        self.rgb_dir = self.data_dir / 'rgb'
        self.gt_dir = self.data_dir / 'ground_truth'
        self.depth_dir = self.data_dir / 'depth'
        self.uv_dir = self.data_dir / 'uv'
        self.border_dir = self.data_dir / 'border'

        if not self.rgb_dir.exists():
            raise ValueError(f"RGB directory not found: {self.rgb_dir}")
        if not self.gt_dir.exists():
            raise ValueError(f"Ground truth directory not found: {self.gt_dir}")

        # Get list of all samples (based on RGB images)
        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.rgb_dir}")

        print(f"Found {len(self.samples)} samples in {self.data_dir}")

        # Define default transforms if none provided
        if self.transform is None:
            self.transform = self._get_default_transform()

    def _find_samples(self) -> List[str]:
        """Find all valid samples (those with both RGB and ground truth)."""
        samples = []

        # Find all RGB images
        rgb_files = sorted(glob.glob(str(self.rgb_dir / "*.jpg")))

        for rgb_path in rgb_files:
            # Extract base filename (without extension)
            base_name = Path(rgb_path).stem

            # Check if ground truth exists
            gt_path = self.gt_dir / f"{base_name}.png"
            if gt_path.exists():
                samples.append(base_name)

        return samples

    def _get_default_transform(self):
        """Get default image transforms."""
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_rgb(self, base_name: str) -> Image.Image:
        """Load RGB image."""
        rgb_path = self.rgb_dir / f"{base_name}.jpg"
        return Image.open(rgb_path).convert('RGB')

    def _load_ground_truth(self, base_name: str) -> Image.Image:
        """Load ground truth image."""
        gt_path = self.gt_dir / f"{base_name}.png"
        return Image.open(gt_path).convert('RGB')

    def _load_depth(self, base_name: str) -> Optional[np.ndarray]:
        """Load depth map (EXR format)."""
        if not self.use_depth:
            return None

        depth_path = self.depth_dir / f"{base_name}.exr"
        if not depth_path.exists():
            return None

        try:
            import OpenEXR
            import Imath

            exr_file = OpenEXR.InputFile(str(depth_path))
            dw = exr_file.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

            depth_str = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
            depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])

            return depth
        except ImportError:
            print("Warning: OpenEXR not installed. Install with: pip install OpenEXR")
            return None

    def _load_uv(self, base_name: str) -> Optional[Image.Image]:
        """Load UV map."""
        if not self.use_uv:
            return None

        uv_path = self.uv_dir / f"{base_name}.png"
        if not uv_path.exists():
            return None

        return Image.open(uv_path).convert('RGB')

    def _load_border(self, base_name: str) -> Optional[Image.Image]:
        """Load border mask."""
        if not self.use_border:
            return None

        border_path = self.border_dir / f"{base_name}.png"
        if not border_path.exists():
            return None

        return Image.open(border_path).convert('L')  # Grayscale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - 'rgb': Input warped document image [3, H, W] (ImageNet-normalized)
                - 'ground_truth': Target flat texture [3, H, W] (ImageNet-normalized)
                - 'depth': Depth map [1, H, W] (if use_depth=True)
                - 'uv': UV map [2, H, W] in [0, 1] — channel 0 = U, 1 = V
                        (raw geometric coords, NOT ImageNet-normalized) (if use_uv=True)
                - 'uv_mask': Foreground mask [1, H, W] auto-derived from the
                             UV map (background = uniform gray) (if use_uv=True)
                - 'border': Border mask [1, H, W] (if use_border=True)
                - 'filename': Base filename (string)
        """
        base_name = self.samples[idx]

        # Load RGB and ground truth (required)
        rgb = self._load_rgb(base_name)
        ground_truth = self._load_ground_truth(base_name)

        # Apply transforms
        rgb_tensor = self.transform(rgb)
        gt_tensor = self.transform(ground_truth)

        # Build output dictionary
        sample = {
            'rgb': rgb_tensor,
            'ground_truth': gt_tensor,
            'filename': base_name
        }

        # Load optional modalities
        if self.use_depth:
            depth = self._load_depth(base_name)
            if depth is not None:
                # Normalize depth and convert to tensor
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth = torch.from_numpy(depth).unsqueeze(0).float()
                # Resize to match img_size
                depth = transforms.Resize(self.img_size)(depth)
                sample['depth'] = depth

        if self.use_uv:
            uv = self._load_uv(base_name)
            if uv is not None:
                # UV is geometric data, not pixel intensities — load as raw
                # [0, 1] floats (Resize + ToTensor only, NO ImageNet normalize).
                uv_full = transforms.Compose([
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                ])(uv)  # [3, H, W] in [0, 1]: (U, V, 0)

                # Foreground mask is free here: background is uniform gray
                # (R == G == B) from the renderer's world clear color, paper
                # has distinct U/V values so its channels differ.
                uv_mask = ~((uv_full[0] == uv_full[1]) & (uv_full[1] == uv_full[2]))

                sample['uv'] = uv_full[:2]                   # [2, H, W]: (U, V)
                sample['uv_mask'] = uv_mask.unsqueeze(0).float()  # [1, H, W]

        if self.use_border:
            border = self._load_border(base_name)
            if border is not None:
                border_tensor = transforms.Compose([
                    transforms.Resize(self.img_size),
                    transforms.ToTensor()
                ])(border)
                sample['border'] = border_tensor

        return sample


def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    train_split: float = 0.8,
    use_depth: bool = False,
    use_uv: bool = False,
    use_border: bool = False,
    img_size: Tuple[int, int] = (512, 512),
    num_workers: int = 4,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Root directory containing the dataset
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training (0.0 to 1.0)
        use_depth: Whether to load depth maps
        use_uv: Whether to load UV maps
        use_border: Whether to load border masks
        img_size: Tuple of (height, width) to resize images to
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle training data
        random_seed: Random seed for reproducible splits

    Returns:
        (train_loader, val_loader): Tuple of DataLoader objects
    """
    # Create dataset
    dataset = DocumentDataset(
        data_dir=data_dir,
        use_depth=use_depth,
        use_uv=use_uv,
        use_border=use_border,
        img_size=img_size
    )

    # Split into train and validation
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    # Use random_split with generator for reproducibility
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def visualize_batch(batch: Dict[str, torch.Tensor], num_samples: int = 4):
    """
    Visualize a batch of samples.

    Args:
        batch: Dictionary containing batch data
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt

    num_samples = min(num_samples, batch['rgb'].shape[0])

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Denormalize images for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        rgb = batch['rgb'][i] * std + mean
        gt = batch['ground_truth'][i] * std + mean

        # Convert to numpy and transpose to HWC
        rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
        gt_np = gt.permute(1, 2, 0).cpu().numpy()

        # Clip values to [0, 1]
        rgb_np = np.clip(rgb_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)

        # Plot
        axes[i, 0].imshow(rgb_np)
        axes[i, 0].set_title(f"Input RGB - {batch['filename'][i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_np)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_sample(
    sample: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
):
    """
    Render one dataset sample with every available modality side-by-side,
    each panel labelled with its name. Useful for adding a single
    illustration to a writeup / project doc.

    Picks up whichever of {rgb, ground_truth, uv, uv_mask, border, depth}
    are present in the sample dict. RGB and ground_truth are denormalised
    out of ImageNet stats so they look natural. UV is shown as a colour map
    (R=U, G=V); masks and depth are shown grayscale.

    Args:
        sample:    Either a single sample dict from DocumentDataset[idx]
                   (tensors shaped [C, H, W]) or one selected from a
                   dataloader batch (tensors shaped [B, C, H, W] — index 0
                   will be used).
        save_path: If given, save the figure to this path (PNG/PDF).
        show:      If True, also open an interactive window.
        title:     Optional overall figure title (e.g. the filename).
    """
    import matplotlib.pyplot as plt

    def _take_first(t: torch.Tensor) -> torch.Tensor:
        return t[0] if t.dim() == 4 else t

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    panels: List[Tuple[np.ndarray, str, Optional[str]]] = []  # (img, label, cmap)

    if 'rgb' in sample:
        rgb = _take_first(sample['rgb']).cpu()
        rgb = (rgb * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        panels.append((rgb, "Input RGB (warped)", None))

    if 'ground_truth' in sample:
        gt = _take_first(sample['ground_truth']).cpu()
        gt = (gt * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        panels.append((gt, "Ground truth (flat)", None))

    if 'uv' in sample:
        uv = _take_first(sample['uv']).cpu()  # [2, H, W] in [0, 1]
        H, W = uv.shape[-2:]
        uv_rgb = torch.zeros(3, H, W)
        uv_rgb[:2] = uv
        uv_rgb = uv_rgb.clamp(0, 1).permute(1, 2, 0).numpy()
        panels.append((uv_rgb, "UV map (R=U, G=V)", None))

    if 'uv_mask' in sample:
        m = _take_first(sample['uv_mask']).cpu().squeeze(0).numpy()
        panels.append((m, "UV mask (auto)", 'gray'))

    if 'border' in sample:
        b = _take_first(sample['border']).cpu().squeeze(0).numpy()
        panels.append((b, "Border mask", 'gray'))

    if 'depth' in sample:
        d = _take_first(sample['depth']).cpu().squeeze(0).numpy()
        panels.append((d, "Depth", 'viridis'))

    if not panels:
        raise ValueError("Sample dict has no recognised modalities to plot.")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, (img, label, cmap) in zip(axes, panels):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(label, fontsize=14)
        ax.axis('off')

    if title is None and 'filename' in sample:
        fname = sample['filename']
        if isinstance(fname, list):  # batch case
            fname = fname[0]
        title = str(fname)
    if title:
        fig.suptitle(title, fontsize=11, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f"Saved sample visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================================
# STARTER CODE FOR DOCUMENT RECONSTRUCTION MODEL
# ============================================================================

class DocumentReconstructionModel(nn.Module):
    """
    Starter model for document dewarping (geometric correction).

    IMPORTANT: The goal is GEOMETRIC RECONSTRUCTION, not photometric matching!
    - The rendered images have lighting/shading effects
    - Your model should focus on learning the geometric transformation (UV/flow field)
    - Don't worry about exact pixel intensities - focus on structure

    TODO: Implement your own architecture here.
    This is a simple U-Net-style baseline to get started.

    Suggestions for improvement:
    - Use a pretrained encoder from HuggingFace (e.g., ResNet, EfficientNet)
    - Add attention mechanisms
    - Use depth/UV information if available
    - Experiment with different loss functions (SSIM is recommended!)
    - Add skip connections
    - Try different decoder architectures

    IMPORTANT HINT: Consider using torch.nn.functional.grid_sample for differentiable warping!

    One powerful approach for document reconstruction is to:
    1. Predict a deformation/flow field (mapping from distorted space to flat space)
    2. Use grid_sample to warp the input image according to this field
    3. This allows the network to learn geometric transformations explicitly

    Example usage of grid_sample:
        # Predict a flow field [B, 2, H, W] representing (x, y) offsets
        flow = self.flow_predictor(features)

        # Create base grid and add flow to get sampling coordinates
        grid = create_base_grid(B, H, W) + flow

        # Sample from input image using the predicted grid
        warped = torch.nn.functional.grid_sample(
            input_image,
            grid.permute(0, 2, 3, 1),  # [B, H, W, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # TODO: Replace this simple architecture with your own design
        # Consider using HuggingFace transformers or timm models as backbone

        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Reconstructed image [B, 3, H, W]
        """
        # TODO: Implement your forward pass
        # Consider predicting a flow field and using grid_sample for warping!
        features = self.encoder(x)
        output = self.decoder(features)
        return output


def create_base_grid(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Helper function to create a base sampling grid for grid_sample.

    Creates a normalized coordinate grid in the range [-1, 1] as expected by grid_sample.

    Args:
        batch_size: Batch size
        height: Image height
        width: Image width
        device: Device to create tensor on

    Returns:
        Grid tensor of shape [B, H, W, 2] with normalized coordinates

    Usage example:
        # Create base grid
        grid = create_base_grid(batch_size, H, W, device)

        # Predict flow field [B, 2, H, W]
        flow = model.predict_flow(features)

        # Add flow to grid (need to permute flow to [B, H, W, 2])
        sampling_grid = grid + flow.permute(0, 2, 3, 1)

        # Warp image
        warped = F.grid_sample(input, sampling_grid, align_corners=True)
    """
    # Create coordinate grids
    y_coords = torch.linspace(-1, 1, height, device=device)
    x_coords = torch.linspace(-1, 1, width, device=device)

    # Create meshgrid
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Stack to create [H, W, 2] grid
    grid = torch.stack([xx, yy], dim=-1)

    # Expand to batch size [B, H, W, 2]
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return grid


# ============================================================================
# LOSS FUNCTIONS FOR DOCUMENT RECONSTRUCTION
# ============================================================================
#
# This section provides loss functions specifically designed for document
# reconstruction. The key insight is that backgrounds should be ignored during
# training, allowing the network to focus on the document surface.
#
# Available loss functions:
# 1. MaskedL1Loss - L1 loss with optional document masking
# 2. MaskedMSELoss - MSE loss with optional document masking
# 3. UVReconstructionLoss - Combined loss for UV-based reconstruction
#
# Usage:
#   criterion = MaskedL1Loss(use_mask=True)
#   loss = criterion(prediction, ground_truth, mask)
#
# For UV-based models that predict flow fields:
#   criterion = UVReconstructionLoss(
#       reconstruction_weight=1.0,
#       uv_weight=0.5,
#       smoothness_weight=0.01,
#       use_mask=True
#   )
# ============================================================================

class MaskedL1Loss(nn.Module):
    """
    L1 Loss with optional masking to focus on document regions.

    If a border mask is provided, this loss will only compute the error
    on pixels where the document exists, ignoring the background.

    This is useful because:
    - The background is not part of the reconstruction task
    - Focusing on document pixels improves convergence
    - Prevents the model from "cheating" by predicting background

    Args:
        use_mask: Whether to apply masking (requires 'border' in batch)
    """

    def __init__(self, use_mask: bool = False):
        super().__init__()
        self.use_mask = use_mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute masked L1 loss.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            mask: Optional mask [B, 1, H, W] where 1=document, 0=background

        Returns:
            Scalar loss value
        """
        # Compute element-wise L1 distance
        l1_loss = torch.abs(pred - target)

        if self.use_mask and mask is not None:
            # Apply mask (broadcast across channels)
            l1_loss = l1_loss * mask

            # Average only over masked pixels
            # This prevents background from contributing to loss
            num_pixels = mask.sum() * pred.shape[1]  # Total masked pixels across channels
            if num_pixels > 0:
                return l1_loss.sum() / num_pixels
            else:
                return l1_loss.mean()  # Fallback if mask is empty
        else:
            # Standard L1 loss
            return l1_loss.mean()


class MaskedMSELoss(nn.Module):
    """
    MSE Loss with optional masking to focus on document regions.

    Similar to MaskedL1Loss but uses squared error (L2).

    Args:
        use_mask: Whether to apply masking (requires 'border' in batch)
    """

    def __init__(self, use_mask: bool = False):
        super().__init__()
        self.use_mask = use_mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute masked MSE loss.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            mask: Optional mask [B, 1, H, W] where 1=document, 0=background

        Returns:
            Scalar loss value
        """
        # Compute element-wise squared error
        mse_loss = (pred - target) ** 2

        if self.use_mask and mask is not None:
            # Apply mask
            mse_loss = mse_loss * mask

            # Average only over masked pixels
            num_pixels = mask.sum() * pred.shape[1]
            if num_pixels > 0:
                return mse_loss.sum() / num_pixels
            else:
                return mse_loss.mean()
        else:
            # Standard MSE loss
            return mse_loss.mean()


class SSIMLoss(nn.Module):
    """
    SSIM (Structural Similarity) Loss - RECOMMENDED for geometric reconstruction.

    SSIM measures structural similarity rather than pixel-wise differences, making it
    ideal for this task where lighting effects differ between input and ground truth.

    Benefits:
    - Focuses on structure (edges, patterns) not pixel intensities
    - Robust to lighting/shading differences
    - More perceptually aligned than L1/L2

    Args:
        data_range: Expected range of input values (1.0 for normalized images)
        channel: Number of channels (3 for RGB)

    Requires: pip install pytorch-msssim
    """

    def __init__(self, data_range: float = 1.0, channel: int = 3):
        super().__init__()
        try:
            from pytorch_msssim import ssim
            self.ssim_func = ssim
        except ImportError:
            raise ImportError(
                "pytorch-msssim not installed. Install with: pip install pytorch-msssim"
            )
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]

        Returns:
            Scalar loss value (lower is better, range 0-1)
        """
        # SSIM returns value in range [0, 1] where 1 is perfect
        # Convert to loss by: loss = 1 - SSIM
        ssim_val = self.ssim_func(pred, target, data_range=self.data_range)
        return 1 - ssim_val


class UVReconstructionLoss(nn.Module):
    """
    Combined loss for UV-based document reconstruction.

    This loss combines:
    1. Reconstruction loss (L1, MSE, or SSIM) on the final image
    2. Optional UV map supervision (if your model predicts UV explicitly)
    3. Optional flow smoothness regularization

    IMPORTANT — UV channel layout:
        target_uv coming from DocumentDataset is shape [B, 2, H, W] (channels
        are U and V; the redundant zero B channel is dropped during loading).
        Your model's pred_uv must match: predict 2 channels, not 3.

    IMPORTANT — using SSIM with UV supervision:
        SSIMLoss is image-oriented and assumes >=3 channels. With loss_type
        ='ssim', the same SSIMLoss instance is reused for the UV branch, so
        passing 2-channel UV tensors will fail. Options:
            (a) Use loss_type='l1' or 'mse' for the UV branch (recommended,
                since UV is geometric coordinates, not perceptual imagery), or
            (b) Replicate one channel before computing SSIM:
                    pred_uv3   = torch.cat([pred_uv,   pred_uv[:,  :1]], 1)
                    target_uv3 = torch.cat([target_uv, target_uv[:, :1]], 1)
        The image branch (pred_image / target_image) is unaffected — those
        are 3-channel and work with SSIM normally.

    Args:
        reconstruction_weight: Weight for image reconstruction loss
        uv_weight: Weight for UV map loss (set to 0 if not predicting UV)
        smoothness_weight: Weight for flow smoothness regularization
        use_mask: Whether to use masking
        loss_type: Type of loss ('l1', 'mse', or 'ssim')
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        uv_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        use_mask: bool = False,
        loss_type: str = 'ssim'  # 'l1', 'mse', or 'ssim' (recommended!)
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.uv_weight = uv_weight
        self.smoothness_weight = smoothness_weight

        # Choose base loss function
        if loss_type == 'ssim':
            self.recon_loss = SSIMLoss()
            self.use_ssim = True
        elif loss_type == 'l1':
            self.recon_loss = MaskedL1Loss(use_mask=use_mask)
            self.use_ssim = False
        else:  # mse
            self.recon_loss = MaskedMSELoss(use_mask=use_mask)
            self.use_ssim = False

    def forward(
        self,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
        pred_uv: Optional[torch.Tensor] = None,
        target_uv: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred_image: Predicted reconstructed image [B, 3, H, W]
            target_image: Ground truth image [B, 3, H, W]
            pred_uv: Predicted UV map [B, 2, H, W] (optional)
            target_uv: Ground truth UV map [B, 2, H, W] (optional)
            flow: Predicted flow field [B, 2, H, W] (optional, for smoothness)
            mask: Document mask [B, 1, H, W] (optional)

        Returns:
            Dictionary with 'total' loss and individual loss components
        """
        losses = {}

        # 1. Reconstruction loss
        if self.use_ssim:
            # SSIM doesn't use mask (operates on full image structure)
            losses['reconstruction'] = self.recon_loss(pred_image, target_image)
        else:
            losses['reconstruction'] = self.recon_loss(pred_image, target_image, mask)
        total_loss = self.reconstruction_weight * losses['reconstruction']

        # 2. UV supervision loss (if applicable)
        # Note: target_uv from DocumentDataset is [B, 2, H, W]. If
        # loss_type='ssim', SSIMLoss expects >=3 channels and will fail here
        # — use 'l1' or 'mse' for the UV branch (see class docstring).
        if self.uv_weight > 0 and pred_uv is not None and target_uv is not None:
            if self.use_ssim:
                losses['uv'] = self.recon_loss(pred_uv, target_uv)
            else:
                losses['uv'] = self.recon_loss(pred_uv, target_uv, mask)
            total_loss += self.uv_weight * losses['uv']

        # 3. Flow smoothness regularization (Total Variation)
        if self.smoothness_weight > 0 and flow is not None:
            # Compute gradients
            dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
            dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
            losses['smoothness'] = (dx.abs().mean() + dy.abs().mean())
            total_loss += self.smoothness_weight * losses['smoothness']

        losses['total'] = total_loss
        return losses


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """
    Train for one epoch.

    TODO: Modify this to add:
    - Additional metrics (PSNR, SSIM)
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training
    - Logging to tensorboard/wandb
    """
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        rgb = batch['rgb'].to(device)
        ground_truth = batch['ground_truth'].to(device)

        # Optional: Load mask if using masked loss
        mask = batch.get('border', None)
        if mask is not None:
            mask = mask.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(rgb)

        # Compute loss (handles both standard and masked losses)
        if isinstance(criterion, (MaskedL1Loss, MaskedMSELoss)):
            loss = criterion(output, ground_truth, mask)
        elif isinstance(criterion, SSIMLoss):
            loss = criterion(output, ground_truth)
        elif isinstance(criterion, UVReconstructionLoss):
            # For advanced UV-based losses, extract additional outputs if available
            # This assumes your model returns (image, uv, flow) - adapt as needed
            losses = criterion(pred_image=output, target_image=ground_truth, mask=mask)
            loss = losses['total']
        else:
            # Standard loss (MSE, L1, etc.)
            loss = criterion(output, ground_truth)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate the model.

    TODO: Modify this to add more metrics here (PSNR, SSIM, etc.)
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch['rgb'].to(device)
            ground_truth = batch['ground_truth'].to(device)

            # Optional: Load mask if using masked loss
            mask = batch.get('border', None)
            if mask is not None:
                mask = mask.to(device)

            output = model(rgb)

            # Compute loss (handles both standard and masked losses)
            if isinstance(criterion, (MaskedL1Loss, MaskedMSELoss)):
                loss = criterion(output, ground_truth, mask)
            elif isinstance(criterion, SSIMLoss):
                loss = criterion(output, ground_truth)
            elif isinstance(criterion, UVReconstructionLoss):
                losses = criterion(pred_image=output, target_image=ground_truth, mask=mask)
                loss = losses['total']
            else:
                loss = criterion(output, ground_truth)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    """
    Main training loop - STARTER CODE

    TODO: Modify this to customize for your experiments:
    1. Implement a better model architecture
    2. Try different loss functions
    3. Add learning rate scheduling
    4. Implement early stopping
    5. Add visualization and logging
    6. Experiment with data augmentation
    7. Use pretrained models from HuggingFace
    """

    # Configuration
    DATA_DIR = 'renders/synthetic_data_pitch_sweep'
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    IMG_SIZE = (512, 512)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    # IMPORTANT: Set use_border=True to enable masked losses!
    train_loader, val_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        use_depth=False,  # TODO: Set to True if you want to use depth information
        use_uv=False,     # TODO: Set to True if you want to use UV maps
        use_border=False  # TODO: Set to True if you want to use border masks for better training
    )

    # Visualize a batch (optional)
    sample_batch = next(iter(train_loader))
    print(f"Batch RGB shape: {sample_batch['rgb'].shape}")
    print(f"Batch GT shape: {sample_batch['ground_truth'].shape}")
    if 'border' in sample_batch:
        print(f"Batch Border mask shape: {sample_batch['border'].shape}")
    # visualize_batch(sample_batch)  # Uncomment to visualize

    # Create model
    model = DocumentReconstructionModel().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # TODO: Try different loss functions
    # Option 1: Simple losses (baseline, not recommended)
    criterion = nn.MSELoss()  # Simple L2 loss - sensitive to lighting!
    # criterion = nn.L1Loss()  # Try L1 loss - also sensitive to lighting

    # Option 2: SSIM Loss (RECOMMENDED - focuses on structure, not lighting!)
    # Uncomment this line (requires: pip install pytorch-msssim)
    # criterion = SSIMLoss()

    # Option 3: Masked losses (focuses on document pixels)
    # Uncomment these lines and set use_border=True above
    # criterion = MaskedL1Loss(use_mask=True)
    # criterion = MaskedMSELoss(use_mask=True)

    # Option 4: Combined loss with UV supervision (ADVANCED)
    # Uncomment and set use_uv=True, use_border=True above
    # criterion = UVReconstructionLoss(
    #     reconstruction_weight=1.0,
    #     uv_weight=0.5,
    #     smoothness_weight=0.01,
    #     use_mask=True,
    #     loss_type='ssim'  # Use SSIM for geometric reconstruction!
    # )

    # TODO: Try different optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*50}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with val loss: {val_loss:.4f}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    # Example: Just load and visualize data
    print("Document Reconstruction Dataset Loader")
    print("="*50)

    # Quick test
    try:
        train_loader, val_loader = get_dataloaders(
            data_dir='renders/synthetic_data_pitch_sweep',
            batch_size=4,
            img_size=(512, 512)
        )

        print("\nDataset loaded successfully!")

        # Visualize a sample batch
        print("\nVisualizing a sample batch...")
        sample_batch = next(iter(train_loader))
        print(f"Batch shape - RGB: {sample_batch['rgb'].shape}, Ground Truth: {sample_batch['ground_truth'].shape}")
        visualize_batch(sample_batch, num_samples=min(4, sample_batch['rgb'].shape[0]))

        print("\nTo start training, uncomment the main() function call below")
        # main()  # Uncomment this to start training

    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("Please check that the data directory exists and contains the required files.")
