import torch
from torch.utils.data import Dataset

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, \
    ToTensor, Normalize, RandomRotation, RandomAutocontrast, RandAugment, CenterCrop, Resize
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PEDataset(Dataset):
    def __init__(self, image_dirs, labels, mode = "train"):
        super(PEDataset, self).__init__()
        self.image_dirs = image_dirs
        self.labels = labels
        self.mode = mode
        if self.mode == "train":
            # self.transform = albu.Compose([
            #     # albu.HorizontalFlip(p=0.5),
            #     # albu.VerticalFlip(p=0.5),
            #     # albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=0, value=0, p=0.25),               
            #     albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #     ToTensorV2()
            #     ])
            self.transform = Compose([
                CenterCrop((190,190)),
                Resize((256,256)),
                RandomVerticalFlip(p=0.5),
                RandomHorizontalFlip(p=0.5),
                RandAugment(num_ops=3),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            # self.transform = albu.Compose([
            #     albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #     ToTensorV2(),
            # ])
            self.transform = Compose([
                CenterCrop((190,190)),
                Resize((256,256)),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_dir = self.image_dirs[index]
        # image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
        image = Image.open(image_dir).convert("RGB")
        # image = np.asarray(image)
        # image = self.transform(image=image)['image']
        # try:
        image = self.transform(image)
        # except:
        #     image = ToTensor()(image)
        label = torch.Tensor([self.labels[index]])
        return image, label
    
            