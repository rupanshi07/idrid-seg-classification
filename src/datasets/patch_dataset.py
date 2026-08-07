import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LESIONS = ["Microaneurysms", "Haemorrhages", "Hard_Exudates", "Soft_Exudates", "Optic_Disc"]


class PatchSegDataset(Dataset):
    """
    Loads full training images/masks once at img_size resolution, caches in memory,
    and serves random patches biased toward lesion-containing regions.
    """
    def __init__(self, root, split="Training", img_size=1024, patch_size=256,
                 patches_per_image=6, lesion_bias=0.75):
        set_name = "a_Training_Set" if split == "Training" else "b_Testing_Set"
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.lesion_bias = lesion_bias

        img_dir = os.path.join(root, "A_Segmentation", "Original_Images", set_name)
        mask_dirs = {
            lesion: os.path.join(root, "A_Segmentation", "All_Segmentation_Groundtruths", set_name, lesion)
            for lesion in LESIONS
        }
        image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

        self.images = []
        self.masks = []
        self.fg_coords = []  # list of (row, col) arrays per image, union across all lesions

        for img_path in image_paths:
            patient_id = os.path.splitext(os.path.basename(img_path))[0]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_size, img_size))

            masks = []
            for lesion in LESIONS:
                matches = glob.glob(os.path.join(mask_dirs[lesion], f"{patient_id}_*"))
                if matches:
                    m = cv2.imread(matches[0], cv2.IMREAD_GRAYSCALE)
                    m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                    m = (m > 0).astype(np.uint8)
                else:
                    m = np.zeros((img_size, img_size), dtype=np.uint8)
                masks.append(m)
            mask_stack = np.stack(masks, axis=0)  # [5, H, W]

            union_fg = (mask_stack.sum(axis=0) > 0)
            coords = np.argwhere(union_fg)  # [[row, col], ...]

            self.images.append(image)
            self.masks.append(mask_stack)
            self.fg_coords.append(coords)

        print(f"PatchSegDataset: cached {len(self.images)} images at {img_size}px, "
              f"{patches_per_image} patches/image/epoch")

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx % len(self.images)
        image = self.images[img_idx]
        mask = self.masks[img_idx]
        coords = self.fg_coords[img_idx]
        H, W = self.img_size, self.img_size
        ps = self.patch_size
        half = ps // 2

        use_lesion_center = (len(coords) > 0) and (np.random.rand() < self.lesion_bias)
        if use_lesion_center:
            cy, cx = coords[np.random.randint(len(coords))]
        else:
            cy, cx = np.random.randint(0, H), np.random.randint(0, W)

        y0 = np.clip(cy - half, 0, H - ps)
        x0 = np.clip(cx - half, 0, W - ps)

        img_patch = image[y0:y0 + ps, x0:x0 + ps, :]
        mask_patch = mask[:, y0:y0 + ps, x0:x0 + ps]

        img_t = torch.from_numpy(img_patch.transpose(2, 0, 1).copy()).float() / 255.0
        mask_t = torch.from_numpy(mask_patch.copy()).float()

        return {"image": img_t, "mask": mask_t}
