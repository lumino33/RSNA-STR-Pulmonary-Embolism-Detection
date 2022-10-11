from turtle import forward
import torch
import torch.nn as nn
import torchvision
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()
        self.model = efficientnet_b0(weights = EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier = nn.Sequential(nn.Linear(1280, 512), 
                                 nn.ReLU(True),
                                 nn.Dropout(0.3),
                                 nn.Linear(512, 64),
                                 nn.ReLU(True),
                                 nn.Dropout(0.3),
                                 nn.Linear(64,1),
                                 nn.Sigmoid()
                                 )
    def forward(self, x):
        x = self.model(x)
        return x
        