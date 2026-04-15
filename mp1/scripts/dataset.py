import json
import os
import numpy as np
import cv2
import threading as thr
import torch
from torch.utils.data import Dataset, random_split


class CaptureDataset(Dataset):
    def __init__(self, src: str, resize=(640, 384)) -> None:
        self._src = src
        self._image_dir = os.path.join(src, "images")
        self._pose_dir = os.path.join(src, "poses")
        self._mask_dir = os.path.join(src, "masks")
        self._save_id = 0
        self._resize = resize

        os.makedirs(self._src, exist_ok=True)
        os.makedirs(self._image_dir, exist_ok=True)
        os.makedirs(self._pose_dir, exist_ok=True)
        os.makedirs(self._mask_dir, exist_ok=True)
        
        self._save_id = len(os.listdir(self._pose_dir))
        
    def capture(self, image: np.ndarray, pose: dict) -> None:
        cv2.imwrite(os.path.join(self._image_dir, f"{self._save_id}.png"), image)
        with open(os.path.join(self._pose_dir, f"{self._save_id}.json"), "w") as f:
            json.dump(pose, f)
        self._save_id += 1
    
    def write_mask(self, mask: np.ndarray, idx: int) -> None:
        cv2.imwrite(os.path.join(self._mask_dir, f"{idx}.png"), mask)

    def __len__(self) -> int:
        return self._save_id
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        image = cv2.imread(os.path.join(self._image_dir, f"{idx}.png"))
        mask = cv2.imread(os.path.join(self._mask_dir, f"{idx}.png"), cv2.IMREAD_GRAYSCALE)
        if self._resize:
            image = cv2.resize(image, self._resize)
            mask = cv2.resize(mask, self._resize, interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = torch.from_numpy(image).float()[None, ...] / 255.0
        mask = (torch.from_numpy(mask) / 255).long()
        return image, mask

    def read(self, idx: int) -> tuple[np.ndarray, dict]:
        image = cv2.imread(os.path.join(self._image_dir, f"{idx}.png"))
        with open(os.path.join(self._pose_dir, f"{idx}.json")) as f:
            pose = json.load(f)
        return image, pose

    def split(self, s: float) -> tuple[Dataset, Dataset]:
        s = min(max(s, 0), 1)
        n_train = int(len(self) * s)
        n_val = len(self) - n_train
        return random_split(self, [n_train, n_val])