#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import data
from . import models

from .train import TrainerForClassification
from .train import TrainerForDetection
from .train import TrainerForSegmentation

__version__ = "0.0.1"


print("== Intraoral AI (ioai) ==")
print("data: 2020-11-08")
print("email: hyeonrae.cho@gmail.com")


#class Device(Enum):
#    CPU = "CPU"
#    GPU = "GPU"
#    TPU = "TPU"
