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


class AttentionGate(nn.Module):
    """Gates skip-connection features using the decoder's coarser signal."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attn = self.relu(g + s)
        attn = self.psi(attn)
        return skip * attn


class MultiTaskUNet(nn.Module):
    def __init__(self, in_ch=3, n_seg_classes=5, n_cls_classes=5, base=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.enc4 = conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.att4 = AttentionGate(base * 8, base * 8, base * 4)
        self.dec4 = conv_block(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.dec3 = conv_block(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.dec2 = conv_block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate(base, base, base // 2)
        self.dec1 = conv_block(base * 2, base)

        self.seg_head = nn.Conv2d(base, n_seg_classes, 1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 16, base * 4),
            nn.LayerNorm(base * 4),
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
            e4_gated = self.att4(d4, e4)
            d4 = self.dec4(torch.cat([d4, e4_gated], dim=1))

            d3 = self.up3(d4)
            e3_gated = self.att3(d3, e3)
            d3 = self.dec3(torch.cat([d3, e3_gated], dim=1))

            d2 = self.up2(d3)
            e2_gated = self.att2(d2, e2)
            d2 = self.dec2(torch.cat([d2, e2_gated], dim=1))

            d1 = self.up1(d2)
            e1_gated = self.att1(d1, e1)
            d1 = self.dec1(torch.cat([d1, e1_gated], dim=1))

            seg_out = self.seg_head(d1)

        if task in ("cls", "both"):
            cls_out = self.cls_head(b)

        return seg_out, cls_out
