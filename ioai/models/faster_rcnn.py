#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def type_1() -> FasterRCNN:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        #box_score_thresh=0.9
        box_fg_iou_thresh=0.2,
        box_bg_iou_thresh=0.2,
    )

    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def type_2() -> FasterRCNN:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        #box_score_thresh=0.9,
        box_fg_iou_thresh=0.2,
        box_bg_iou_thresh=0.2,
    )

    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def type_3() -> FasterRCNN:
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    
    # resnet18, resnet34, resnet50, resnet101, resnet152, ...
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.DEFAULT, trainable_layers=3)
    
    anchor_generator = AnchorGenerator(
        sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0)
    )
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        #featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )
    
    model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_fg_iou_thresh=0.2,
        box_bg_iou_thresh=0.2,
    )


if __name__ == "__main__":
    from torchinfo import summary

    model = type_2()
    model.eval()

    print(model)
    summary(model, (1, 3, 100, 100))
