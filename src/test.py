import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time

from dataset import PEDataset
from model import EfficientNetB0

#Load Data
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    SAVE_PATH = "./saved model/best.pth" 
    
    df_test = pd.read_csv('RSNA-STR-Pulmonary-Embolism-Detection/prepare/test.csv') 
    test_image_dirs = df_test.image_path.tolist()
    test_labels = df_test.pe_present_on_image.tolist()

    test_datagen = PEDataset(image_dirs=test_image_dirs, labels=test_labels, mode="test")
    testloader = DataLoader(test_datagen, batch_size = batch_size, num_workers=2, shuffle=False)

    model = EfficientNetB0()
    model.load_state_dict(torch.load(SAVE_PATH))
    model.to(device)
    model.eval()
    
    criterion = nn.BCELoss()
    
    accuracy = {}
    accuracy['test'] = []

    loss_record = {}
    loss_record['test'] = []
    
    correct = 0
    losses = torch.tensor([])
    for i, data in enumerate(testloader, 0):
        # forward pass
        X, y = data
        X = X.to(torch.float)
        X, y = X.to(device), y.to(device)
        
        with torch.no_grad():
            output = model(X)
        # get number of accurate predictions
        predicted = torch.where(output > 0.5, 1, 0).squeeze()
        correct += (predicted == y).sum()
        loss = criterion(output, y)
        losses = torch.cat((losses, torch.tensor([loss.detach()])), 0)
    # calculate mean accuracy and print
    meanacc = float(correct) / (len(testloader) * batch_size)
    meanloss = float(losses.mean())
    print('Loss:', meanloss, 'Accuracy:', meanacc)
    accuracy['test'].append(meanacc)
    loss_record['test'].append(meanloss)
    
    # putting this in to keep the console clean
    time.sleep(0.5)
    
if __name__ == "__main__":
    main()