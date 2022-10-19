import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm


class EfficientNet(nn.Module):
    def __init__(self, backbone, pretrained=False, num_classes=1, channel=3, in_features=1280):
        super(EfficientNet, self).__init__()
        self.base =  timm.create_model(backbone, pretrained=pretrained,
                                       num_classes = 0,
                                       in_chans=channel)
        self.fc1= nn.Linear(in_features, 2048, bias=True)
        self.fc2= nn.Linear(2048, 1024, bias=True)
        self.last_linear= nn.Linear(1024, num_classes, bias=True)
        # self.sigmoid = nn.Sigmoid()
        
    @autocast()
    def forward(self, x):
        x = self.base(x)
        x= self.fc1(x)
        x= F.relu(x)
        x= F.dropout(x, p=0.6)
        x= self.fc2(x)
        x= F.relu(x)
        x= F.dropout(x, p=0.6)
        x= self.last_linear(x)
        # x = self.sigmoid(x)
        return x

    @autocast()
    def extract_features(self, x):
        x= self.base(x)
        x= self.fc1(x)
        x= F.relu(x)
        emb= self.fc2(x)
        out= F.relu(emb)
        out= self.last_linear(out)
        # out = self.sigmoid(out)
        return emb, out