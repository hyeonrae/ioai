#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import ioai

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def image_convert(image):
    image = image.clone().cpu().numpy()

    if len(image.shape) == 3:
        image = image.transpose((1,2,0))
    else: # == 2
        image = image.transpose((0,1))

    image = (image * 255).astype(np.uint8)
    return image


def plot_det(data, idx):
    out = data.__getitem__(idx)
    image = image_convert(out[0])
    image = np.ascontiguousarray(image)
    bb = out[1]['boxes'].numpy()
    for i in bb:
        cv2.rectangle(image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,255,0), thickness=2)
    plt.imshow(image)
    plt.show()

def plot_cls(data, idx):
    out = data.__getitem__(idx)
    #out = data[idx]
    image = image_convert(out[0])
    image = np.ascontiguousarray(image)
    print(out[1])
    plt.imshow(image)
    plt.show()


def plot_seg(data, idx):
    out = data.__getitem__(idx)
    #out = data[idx]
    image = image_convert(out[0])
    image = np.ascontiguousarray(image)

    mask = image_convert(out[1])
    mask = np.ascontiguousarray(mask)

    f, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[1].imshow(mask)
    plt.show()



# Trainer(model, data, callback_fn, 'detection')

def cbfn(*args):
    print('[cb] ', args)


def main():

    # init
    with open('data_conf.json') as f:
        data_conf = json.load(f)

    #filelist = data_conf.get('fileList'))

    # run
    print('\n>>> Program start <<<\n')

    dataset_param = {
        'c': ('ioc_reorgan', 'classification', 'train'),
        'd': ('ioc_reorgan', 'detection', 'train'),
        's': ('ioc_reorgan', 'segmentation', 'train'),
    }

    trainers = {
        'c': ioai.TrainerForClassification,
        'd': ioai.TrainerForDetection,
        's': ioai.TrainerForSegmentation,
    }

    models = {
        'c': ioai.models.resnet.type_1,
        'd': ioai.models.faster_rcnn.type_2,
        's': ioai.models.unet.type_1,
    }

    cb = {
        'c': None,
        'd': None,
        's': None,
    }

    
    #size = len(data)
    #a_size = int(0.8 * size)
    #b_size = int(0.1 * size)
    #c_size = size - a_size - b_size
    #print(size, a_size, b_size, c_size)
    #A, B, C = random_split(data, [a_size, b_size, c_size])

    #plot_img(data, 9)
    #plot_seg(data, 0) # nomal
    #plot_seg(data, 9) # caries
    #print(type(data[0][0]))
    #plt.imshow(data[0][0])
    #plt.show()


    # c: classification - resnet
    # d: detection - faster_rcnn
    # s: segmentation - unet
    TASKTYPE = 'c'
    #TASKTYPE = 'd'
    #TASKTYPE = 's'

    # === DATA ===
    data = ioai.data.DatasetBag().load(*dataset_param[TASKTYPE])

    # === MODEL ===
    model = models[TASKTYPE]()

    # === Training ===
    num_epochs = 5
    train_loss_min = 0.9

    T = trainers[TASKTYPE](
        model=model,
        dataset=data,
        batch_size=1,
        #print_callback=cbfn
    ).to('cpu')
    #.to(device)
    T.run(n_epoch=num_epochs)
    T.save()



    #ioai.models.Model()
    #model = ioai.models.FasterRCNN()
    #_ = model.fit(data.train, data,labels)


    # terminate
    print('\n>>> Program end <<<\n')



if __name__ == "__main__":
    main()
