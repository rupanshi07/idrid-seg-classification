import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LESIONS = ["Microaneurysms", "Haemorrhages", "Hard_Exudates", "Soft_Exudates", "Optic_Disc"]


class SegDataset(Dataset):
    """A_Segmentation subset: image + 5 lesion masks. No grade labels."""
    def __init__(self, root, split="Training", img_size=512, transform=None):
        set_name = "a_Training_Set" if split == "Training" else "b_Testing_Set"
        self.img_size = img_size
        self.transform = transform
        self.img_dir = os.path.join(root, "A_Segmentation", "Original_Images", set_name)
        self.mask_dirs = {
            lesion: os.path.join(root, "A_Segmentation", "All_Segmentation_Groundtruths", set_name, lesion)
            for lesion in LESIONS
        }
        self.image_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        if len(self.image_paths) == 0:
            self.image_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))

    def __len__(self):
        return len(self.image_paths)

    def _find_mask(self, patient_id, lesion):
        matches = glob.glob(os.path.join(self.mask_dirs[lesion], f"{patient_id}_*"))
        return matches[0] if matches else None

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        patient_id = os.path.splitext(os.path.basename(img_path))[0]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        masks = []
        for lesion in LESIONS:
            mpath = self._find_mask(patient_id, lesion)
            if mpath is None:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.img_size, self.img_size))
                mask = (mask > 0).astype(np.uint8)
            masks.append(mask)
        mask_stack = np.stack(masks, axis=0).astype(np.float32)  # [5, H, W]

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask_stack)

        return {"image": image_t, "mask": mask_t, "patient_id": patient_id}


class ClsDataset(Dataset):
    """B_DiseaseGrading subset: image + DR grade (0-4). No masks."""
    def __init__(self, root, split="Training", img_size=512, transform=None):
        set_name = "a_Training_Set" if split == "Training" else "b_Testing_Set"
        self.img_size = img_size
        self.transform = transform
        self.img_dir = os.path.join(root, "B_DiseaseGrading", "Original_Images", set_name)

        grading_dir = os.path.join(root, "B_DiseaseGrading", "Groundtruths")
        csvs = glob.glob(os.path.join(grading_dir, f"{split}_Labels.csv"))
        df = pd.read_csv(csvs[0])
        df.columns = [c.strip() for c in df.columns]
        df = df[["Image name", "Retinopathy grade"]].dropna()
        df["Image name"] = df["Image name"].astype(str).str.strip()
        self.labels = dict(zip(df["Image name"], df["Retinopathy grade"].astype(int)))

        all_imgs = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        self.image_paths = [p for p in all_imgs if os.path.splitext(os.path.basename(p))[0] in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        patient_id = os.path.splitext(os.path.basename(img_path))[0]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        grade_t = torch.tensor(self.labels[patient_id], dtype=torch.long)
        return {"image": image_t, "grade": grade_t, "patient_id": patient_id}
