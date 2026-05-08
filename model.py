import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


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