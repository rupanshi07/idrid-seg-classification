import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MultiTaskUNet(nn.Module):
    def __init__(self, in_ch=3, n_seg_classes=5, n_cls_classes=5, base=32):
        super().__init__()
        # Encoder
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.enc4 = conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(base * 8, base * 16)

        # Segmentation decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.seg_head = nn.Conv2d(base, n_seg_classes, 1)

        # Classification head (designed separately, branches off bottleneck)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 16, base * 4),
            nn.BatchNorm1d(base * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base * 4, base),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base, n_cls_classes),
        )

    def forward(self, x, task="both"):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        seg_out, cls_out = None, None

        if task in ("seg", "both"):
            d4 = self.up4(b)
            d4 = self.dec4(torch.cat([d4, e4], dim=1))
            d3 = self.up3(d4)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            seg_out = self.seg_head(d1)

        if task in ("cls", "both"):
            cls_out = self.cls_head(b)

        return seg_out, cls_out
