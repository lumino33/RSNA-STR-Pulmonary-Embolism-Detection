import os
import glob
import torch
import numpy as np
import pandas as pd
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from tqdm import tqdm

from dataset import PEDataset
from model import EfficientNetB0

# Hyper-parameter
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH_MODEL = "/home/hungld11/Documents/RSNA Competition/saved model/best.pth" 
SAVE_PATH_FILE = "/home/hungld11/Documents/RSNA Competition/RSNA-STR-Pulmonary-Embolism-Detection/study level/exam image/test_exam.pt"

df_train = pd.read_csv("/home/hungld11/Documents/RSNA Competition/RSNA-STR-Pulmonary-Embolism-Detection/study level/prepare/exam_level_test.csv")
folder_paths = list(('/home/hungld11/Documents/RSNA Competition/init train/'+df_train.StudyInstanceUID+'/'+ df_train.SeriesInstanceUID+'/'))

model = EfficientNetB0()
model.load_state_dict(torch.load(SAVE_PATH_MODEL))
model.eval()

# f = open(SAVE_PATH_FILE, "wb")
exam_dict = {}
for folder_path in tqdm(folder_paths, total=len(folder_paths)):
    image_paths = sorted(glob.glob(os.path.join(folder_path,'*.jpg')))
    embedding_features = torch.tensor([])
    transform = albu.Compose([albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2()])
    for image_path in image_paths:
        #read image
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img = transform(image=img)['image']
        img = torch.unsqueeze(img, dim=0)
        #get embedding vector
        with torch.no_grad():
            _, embedding_feature = model(img)
        embedding_features = torch.cat([embedding_features, embedding_feature], axis = 0)
    exam_dict[folder_path] = embedding_features.squeeze()
        #concat

torch.save(exam_dict, SAVE_PATH_FILE) #save to file 

# f.close()

