import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), probs.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)
        intersection = (probs * targets).sum(-1)
        union = probs.sum(-1) + targets.sum(-1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight  # tensor [C] or None

    def forward(self, logits, targets):
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        pw = pw.view(1, -1, 1, 1) if pw is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if pw is not None:
            weight_map = torch.where(targets == 1, pw, torch.ones_like(targets))
            bce = bce * weight_map
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class SegComboLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, pos_weight=None):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(pos_weight=pos_weight)
        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + self.fw * self.focal(logits, targets)


def mean_dice_per_lesion(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    dims = (0, 2, 3)
    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice
