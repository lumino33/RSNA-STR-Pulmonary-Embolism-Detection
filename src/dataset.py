import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import cv2 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2


class PEDataset(Dataset):
    def __init__(self, image_dirs, labels, mode = "train"):
        super(PEDataset, self).__init__()
        self.image_dirs = image_dirs
        self.labels = labels
        self.mode = mode
        if self.mode == "train":
            self.transform = albu.Compose([
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                ])
        else:
            self.transform = albu.Compose([
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_dir = self.image_dirs[index]
        image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        label = torch.Tensor([self.labels[index]]).float()
        return image, label
    
            