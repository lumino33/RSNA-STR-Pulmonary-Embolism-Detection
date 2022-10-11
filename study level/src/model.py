import torch.nn as nn
import torch
import timm

class EfficientNetB0(nn.Module):
    def __init__(self, backbone, pretrained=True, channel = 1, num_classes = 9):
        super(EfficientNetB0, self).__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.channel = channel
        self.model = timm.create_model(self.backbone, 
                                       pretrained=self.pretrained, 
                                       in_chans=self.channel, 
                                       num_classes=self.num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

class SEResNeXt(nn.Module):
    def __init__(self, backbone, pretrained=True, channel = 1, num_classes = 9):
        super(SEResNeXt, self).__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.channel = channel
        self.model = timm.create_model(self.backbone, 
                                       pretrained=self.pretrained, 
                                       in_chans=self.channel, 
                                       num_classes=self.num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)
