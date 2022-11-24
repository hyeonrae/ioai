#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader, Dataset

import json
import pandas as pd


ALLOWED_DATASETS = [
    'ioc_samples',
    'ioc_reorgan',
    'ioc_qray',
]


class BaseDataset(Dataset):
    def __init__(self, phase='train'):
        print('IOAI base data')
        images = []
        labels = []

        self.phase = phase  # [train, test]
        self._type = None
        self.transforms = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        raise NotImplementedError()

    def summary(self):
        print("== Data Summary ==============================")
        print(pd.Series(self.images))
        print(self.labels)
        print()

        print(f"nImages: {len(self.images)}")
        print(f"nLabels: {len(self.labels)}")
        print()
        print("==============================================")


class DatasetBag:
    def __init__(self):
        self.name = None
        self.label_type = None
        self.phase = None

    def load(self, name=None, label_type='classification', phase='train'):
        """
        name: dataset name
        label_type: ['classification', 'detection', 'segmentation']
        phase: ['train, 'test']

        """

        allowed_list = ALLOWED_DATASETS
        assert name in allowed_list, f"[{name}] is not supported"

        self.name = name
        self.label_type = label_type
        self.phase = phase

        # conf file must exist.
        with open(f'conf/{self.name}.json') as f:
            data_conf = json.load(f)

        return self._load_ioc_series(
            name=self.label_type,
            phase=self.phase,
            ioc_conf=data_conf
        )


    def _load_ioc_series(self, name, phase, ioc_conf):
        _module_name = 'ioai.data.ioc'
        mod = __import__(_module_name, fromlist=['ioai'])

        class_name = 'IoaiDataset' + name.capitalize()
        class_args = (phase, ioc_conf)
        return getattr(mod, class_name)(*class_args)


