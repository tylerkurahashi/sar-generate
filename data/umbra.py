from glob import glob
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


def get_img_df(data_dir: str, fold_cfg_path: str, output_dir: str):
    fold = OmegaConf.load(fold_cfg_path).fold
    df = pd.DataFrame()
    df["img"] = sorted(glob(f"{data_dir}/*.tif"))
    df["fold"] = df["img"].apply(lambda x: int(fold[x.split("/")[-1].split("_")[0]]))

    df.to_csv(f"{output_dir}/img_paths.csv")

    return df


class UmbraDataset(torch.utils.data.Dataset):
    def __init__(self, df, fold, transforms, phase) -> None:
        super().__init__()
        if phase == "train":
            self.df = df[df["fold"] != fold]
        else:
            self.df = df[df["fold"] == fold]

        self.df.reset_index(inplace=True, drop=True)
        self.transform = transforms

    def load_img(self, path):
        img = np.array(tifffile.imread(path), dtype=np.float32)
        img = img[..., np.newaxis]

        return img

    def __getitem__(self, index: int):
        path = self.df["img"].iloc[index]
        img = self.load_img(path)

        transformed = self.transform(image=img)

        return transformed["image"]

    def __len__(self) -> int:
        return len(self.df["img"])


class UmbraDataModule(LightningDataModule):
    def __init__(self, cfg_dm: DictConfig) -> None:
        super().__init__()

        OmegaConf.resolve(cfg_dm)

        self.seed = cfg_dm.seed
        self.batch_size = cfg_dm.batch_size
        self.fold_cfg_path = cfg_dm.fold_cfg_path
        self.data_dir = cfg_dm.data_dir
        self.output_dir = cfg_dm.output_dir
        self.image_resolution = cfg_dm.image_resolution

        self.train_aug = A.Compose(
            [
                A.Resize(
                    self.image_resolution,
                    self.image_resolution,
                    interpolation=cv2.INTER_LINEAR,
                ),
                A.Flip(p=0.5),
                A.Transpose(p=0.5),
                ToTensorV2(),
            ]
        )
        self.val_aug = A.Compose(
            [
                A.Resize(
                    self.image_resolution,
                    self.image_resolution,
                    interpolation=cv2.INTER_LINEAR,
                ),
                ToTensorV2(),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        df = get_img_df(self.data_dir, self.fold_cfg_path, self.output_dir)
        self.train_dataset = UmbraDataset(df, 0, self.train_aug, phase="train")
        self.val_dataset = UmbraDataset(df, 0, self.val_aug, phase="val")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
