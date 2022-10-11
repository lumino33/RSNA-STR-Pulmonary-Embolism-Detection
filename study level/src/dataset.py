import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob



class ExamPEDataset(Dataset):
    def __init__(self, image_dirs, labels, exam_data):
        super(ExamPEDataset, self).__init__()
        self.image_dirs = image_dirs
        self.labels = labels
        self.exam_data = exam_data
            
    def __len__(self):
        return len(self.image_dirs)
    
    def __getitem__(self, index):
        image_dir = self.image_dirs[index]
        image = self.exam_data[image_dir].unsqueeze(dim=0)
        label = torch.Tensor([self.labels[index]]).float()
        return image, label
    
            