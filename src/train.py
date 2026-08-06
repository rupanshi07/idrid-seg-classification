import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from itertools import cycle

from src.datasets.idrid_dataset import SegDataset, ClsDataset
from src.models.unet_multitask import MultiTaskUNet
from src.losses.losses import SegComboLoss, mean_dice_per_lesion

LESIONS = ["Microaneurysms", "Haemorrhages", "Hard_Exudates", "Soft_Exudates", "Optic_Disc"]

# sqrt-scaled pos_weight from measured foreground ratios (MA, HE, EX, SE, OD)
POS_WEIGHT = torch.tensor([28.8, 9.9, 10.7, 22.7, 7.4])


def evaluate(model, seg_loader, cls_loader, device):
    model.eval()
    dice_sum = torch.zeros(5)
    n_batches = 0
    correct, total = 0, 0

    with torch.no_grad():
        for batch in seg_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                seg_logits, _ = model(images, task="seg")
            dice = mean_dice_per_lesion(seg_logits.float(), masks)
            dice_sum += dice.cpu()
            n_batches += 1

        for batch in cls_loader:
            images = batch["image"].to(device)
            grades = batch["grade"].to(device)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                _, cls_logits = model(images, task="cls")
            preds = cls_logits.argmax(dim=1)
            correct += (preds == grades).sum().item()
            total += grades.size(0)

    mean_dice = dice_sum / max(n_batches, 1)
    acc = correct / max(total, 1)
    return mean_dice, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=r"D:\IDRID2\DATASET")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_seg = SegDataset(args.data_root, split="Training", img_size=args.img_size)
    test_seg = SegDataset(args.data_root, split="Testing", img_size=args.img_size)
    train_cls = ClsDataset(args.data_root, split="Training", img_size=args.img_size)
    test_cls = ClsDataset(args.data_root, split="Testing", img_size=args.img_size)

    train_seg_loader = DataLoader(train_seg, batch_size=args.batch_size, shuffle=True, num_workers=2)
    train_cls_loader = DataLoader(train_cls, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_seg_loader = DataLoader(test_seg, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_cls_loader = DataLoader(test_cls, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MultiTaskUNet().to(device)
    seg_loss_fn = SegComboLoss(pos_weight=POS_WEIGHT)
    cls_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_mean_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        cls_iter = cycle(train_cls_loader)
        running_seg_loss, running_cls_loss = 0.0, 0.0
        n_steps = 0
        optimizer.zero_grad()

        for step, seg_batch in enumerate(train_seg_loader):
            cls_batch = next(cls_iter)

            images_seg = seg_batch["image"].to(device)
            masks = seg_batch["mask"].to(device)
            images_cls = cls_batch["image"].to(device)
            grades = cls_batch["grade"].to(device)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                seg_logits, _ = model(images_seg, task="seg")
                seg_loss = seg_loss_fn(seg_logits, masks)

                _, cls_logits = model(images_cls, task="cls")
                cls_loss = cls_loss_fn(cls_logits, grades)

                loss = (seg_loss + cls_loss) / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_seg_loss += seg_loss.item()
            running_cls_loss += cls_loss.item()
            n_steps += 1

        mean_dice, acc = evaluate(model, test_seg_loader, test_cls_loader, device)
        overall_dice = mean_dice.mean().item()
        scheduler.step(overall_dice)

        print(f"Epoch {epoch}/{args.epochs} | seg_loss={running_seg_loss/n_steps:.4f} "
              f"cls_loss={running_cls_loss/n_steps:.4f} | mean_dice={overall_dice:.4f} | cls_acc={acc:.4f}")
        for lesion, d in zip(LESIONS, mean_dice.tolist()):
            print(f"    {lesion}: {d:.4f}")

        if overall_dice > best_mean_dice:
            best_mean_dice = overall_dice
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best_model.pth"))
            print(f"    -> New best mean dice {best_mean_dice:.4f}, checkpoint saved.")

    print("Training complete. Best mean dice:", best_mean_dice)


if __name__ == "__main__":
    main()
