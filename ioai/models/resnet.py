#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

def type_1(n_classes=2):
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, n_classes)
    return model


if __name__ == "__main__":
    from torchinfo import summary
    model = type_1()
    model.eval()
    print(model)

    try:
        summary(model, (1, 3, 256, 256))
    except RuntimeError:
        print('FAILED')
 
