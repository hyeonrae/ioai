#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch.utils.data import DataLoader, random_split


DEVICE_DEFAULT = 'cpu'


class TrainerBase:
    def __init__(
        self,
        model,
        print_callback=None
    ):
        self.model = model
        self.dataset = None
        self.print = print_callback 
        self.device = DEVICE_DEFAULT

        if not self.print:
            self.print = print

    def run(self):
        raise NotImplementedError()

    def to(self, device='cpu'):
        self.device = device
        return self

    def save(self, path):
        pass


class TrainerForDetection(TrainerBase):
    def __init__(
        self,
        model,
        dataset,
        batch_size=1,
        print_callback=None
    ):
        super().__init__(
            model=model,
            print_callback=print_callback
        )

        self.dataset = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainerForDetection.collate_fn
        )

    def run(self, n_epoch):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        self.model.to(self.device)

        total_train_loss = []
        for epoch in range(n_epoch):
            print(f'Epoch :{epoch + 1}')
            train_loss = []
            for images, targets in self.dataset:

                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                train_loss.append(losses.item())        
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                self.print(loss_dict)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class TrainerForClassification(TrainerBase):
    def __init__(
        self,
        model,
        dataset,
        batch_size=1,
        print_callback=None
    ):
        super().__init__(
            model=model,
            print_callback=print_callback
        )

        self.dataset = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    def run(self, n_epoch):
        #params = [p for p in self.model.parameters() if p.requires_grad]
        #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        self.model.to(self.device)

        total_train_loss = []
        for epoch in range(n_epoch):
            print(f'Epoch :{epoch + 1}')
            train_loss = []
            for images, targets in self.dataset:
                images = images.to(self.device)
                targets = targets.to(self.device)

                out = self.model(images)
                self.model.train()

                self.print(out)
                #loss = loss_fn(out, targets)
                #loss.backward()
                #optimizer.step()
                #optimizer.zero_grad()
                #self.print(loss)


class TrainerForSegmentation(TrainerBase):
    def __init__(
        self,
        model,
        dataset,
        batch_size=1,
        print_callback=None
    ):
        super().__init__(
            model=model,
            print_callback=print_callback
        )

        self.dataset = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    def run(self, n_epoch):
        #params = [p for p in self.model.parameters() if p.requires_grad]
        #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        self.model.to(self.device)

        total_train_loss = []
        for epoch in range(n_epoch):
            print(f'Epoch :{epoch + 1}')
            train_loss = []
            for images, targets in self.dataset:
                images = images.to(self.device)
                targets = targets.to(self.device)

                out = self.model(images)
                self.model.train()

                self.print(out)
                #loss = loss_fn(out, targets)
                #loss.backward()
                #optimizer.step()
                #optimizer.zero_grad()
                #self.print(loss)

