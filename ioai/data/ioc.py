#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from .base import IoaiDataset, IoaiDetectionDataset, IoaiClassificationDataset, IoaiSegmentationDataset
from .base import BaseDataset
from typing import Dict

import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch

from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip,
    ShiftScaleRotate,
    VerticalFlip,
    Normalize,
    Flip,
    Compose,
    GaussNoise,
    Resize,
)

import zlib
import base64


class IoaiDatasetClassification(BaseDataset):
    def __init__(self, phase="train", ioc_conf=None):
        self.label_type = "classification"
        self.phase = phase

        self.image_dir = ioc_conf.get("image_dir")
        _csvfile = ioc_conf.get("cls_csv")

        self.labels = pd.read_csv(_csvfile)
        self.images = self.labels["image_id"].unique()

        self.transforms = self.get_transforms(self.phase)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        image_name = image_id + ".jpg"
        image_path = os.path.join(self.image_dir, image_name)

        arr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        arr_image = cv2.cvtColor(arr_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        arr_image /= 255.0

        df_labels = self.labels[self.labels["image_id"] == image_id]

        target = 1.0 if df_labels[["anomaly"]].values else 0.0

        if self.transforms:
            image = self.transforms(image=arr_image)["image"]

        return image, target

    def get_transforms(self, phase):
        list_transforms = []

        if phase == "train":
            list_transforms.extend(
                [
                    # RandomHorizontalFlip(),
                    Flip(p=0.5),
                ]
            )

        list_transforms.extend(
            [
                Resize(128, 128),
                ToTensorV2(),
            ]
        )

        return Compose(list_transforms)

    def summary(self):
        super().summary()
        print(f"label type: {self.label_type}")
        print(self.labels["anomaly"].value_counts())
        print(self.labels["direction"].value_counts())
        print("==============================================")


class IoaiDatasetDetection(BaseDataset):
    def __init__(self, phase="train", ioc_conf=None):
        self.label_type = "detection"
        self.phase = phase

        self.image_dir = ioc_conf.get("image_dir")
        _csvfile = ioc_conf.get("det_csv")

        self.labels = pd.read_csv(_csvfile)
        self.images = self.labels["image_id"].unique()

        self.transforms = self.get_transforms(self.phase)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        image_name = image_id + ".jpg"
        image_path = os.path.join(self.image_dir, image_name)

        arr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        arr_image = cv2.cvtColor(arr_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        arr_image /= 255.0

        df_labels = self.labels[self.labels["image_id"] == image_id]
        boxes = df_labels[["x", "y", "w", "h"]].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        _class = torch.ones((df_labels.shape[0],), dtype=torch.int64)
        iscrowd = torch.zeros((df_labels.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = _class
        target["image_id"] = torch.tensor(idx)
        target["area"] = area
        target["iscrowd"] = iscrowd

        sample = {
            "image": arr_image,
            "bboxes": target["boxes"],
            "labels": target["labels"],
        }

        if self.transforms:
            sample = self.transforms(**sample)

        image = sample["image"]
        target["boxes"] = torch.stack(
            tuple(map(torch.tensor, zip(*sample["bboxes"])))
        ).permute(1, 0)

        return image, target

    def get_transforms(self, phase):
        list_transforms = []

        if phase == "train":
            list_transforms.extend(
                [
                    Flip(p=0.5),
                ]
            )

        list_transforms.extend(
            [
                Resize(240, 240),
                ToTensorV2(),
            ]
        )

        return Compose(
            list_transforms,
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )

    def summary(self):
        super().summary()
        print(f"label type: {self.label_type}")
        print(self.labels["type"].value_counts())
        print("==============================================")


class IoaiDatasetSegmentation(BaseDataset):
    def __init__(self, phase="train", ioc_conf=None):
        self.label_type = "segmentation"
        self.phase = phase

        self.image_dir = ioc_conf.get("image_dir")
        _csvfile = ioc_conf.get("seg_csv")

        self.labels = pd.read_csv(_csvfile)
        self.images = self.labels["image_id"].unique()

        self.transforms = self.get_transforms(self.phase)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        image_name = image_id + ".jpg"
        image_path = os.path.join(self.image_dir, image_name)

        arr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        arr_image = cv2.cvtColor(arr_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        arr_image /= 255.0

        imh, imw, _ = arr_image.shape

        df_labels = self.labels[self.labels["image_id"] == image_id]
        mask_origin = df_labels[["origin_x", "origin_y"]].values[0]
        mask_b64 = df_labels[["mask"]].values[0][0]

        local_mask = self.base64_2_mask(mask_b64)
        img_mask = np.zeros([imh, imw], dtype=np.uint8)
        img_zero = np.zeros([imh, imw], dtype=np.uint8)

        local_shape = local_mask.shape

        img_mask[
            mask_origin[1] : mask_origin[1] + local_shape[0],
            mask_origin[0] : mask_origin[0] + local_shape[1],
        ] += local_mask

        sample = {
            "image": arr_image,
            "mask": img_mask,
        }

        if self.transforms:
            sample = self.transforms(**sample)

        image = sample["image"]
        target = sample["mask"]

        return image, target

    def get_transforms(self, phase):
        list_transforms = []

        if phase == "train":
            list_transforms.extend(
                [
                    # RandomHorizontalFlip(),
                    Flip(p=0.5),
                ]
            )

        list_transforms.extend(
            [
                Resize(80, 80),
                ToTensorV2(),
            ]
        )

        return Compose(list_transforms)

    def base64_2_mask(self, s):
        z = zlib.decompress(base64.b64decode(s))
        n = np.frombuffer(z, np.uint8)
        mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(np.uint8)
        return mask

    def summary(self):
        super().summary()
        print(f"label type: {self.label_type}")
        print(self.labels["type"].value_counts())
        print("==============================================")
