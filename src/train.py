import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from tqdm import tqdm
from model import EfficientNetB0
from dataset import PEDataset
from statistics import mean
import time

def main():
    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyper-parameter
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 30
    SAVE_PATH = "./saved model/best.pth" 
    
    #iterator for training 
    df_train = pd.read_csv('RSNA-STR-Pulmonary-Embolism-Detection/prepare/train.csv')
    image_dirs = df_train.image_path.tolist()
    labels = df_train.pe_present_on_image.tolist()
    train_datagen = PEDataset(image_dirs=image_dirs, labels=labels, mode="train")
    trainloader = torch.utils.data.DataLoader(train_datagen, batch_size = batch_size, num_workers=2, shuffle=True, pin_memory=True)
    
    df_val = pd.read_csv('RSNA-STR-Pulmonary-Embolism-Detection/prepare/val.csv')
    val_image_dirs = df_val.image_path.tolist()
    val_labels = df_val.pe_present_on_image.tolist()
    val_datagen = PEDataset(image_dirs=val_image_dirs, labels=val_labels, mode="val")
    valloader = DataLoader(val_datagen, batch_size = batch_size, num_workers=2, shuffle=False)
    
    # build model 
    model = EfficientNetB0().to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    #training
    accuracy = {}
    accuracy['train'] = []
    accuracy['valid'] = []

    loss_record = {}
    loss_record['train'] = []
    loss_record['valid'] = []
    
    best_loss = 9999

    for epoch in range(num_epochs):
        #train
        correct = 0
        losses = torch.tensor([])
        bar = tqdm(enumerate(trainloader, 0), total=len(trainloader))
        for i, data in bar:
            X, y = data
            X = X.to(torch.float)
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad() #zero gradients
            output = model(X)
            loss = criterion(output, y) #caculate losses
            loss.backward() #backward pass
            
            predicted = torch.where(output > 0.5, 1, 0).squeeze()
            correct += (predicted == y).sum()
            losses = torch.cat((losses, torch.tensor([loss.detach()])), 0)
            optimizer.step() #update parameters
            
            bar.set_postfix(loss=loss.item())
        
        #caculate mean accuracy and mean loss
        meanacc = float(correct) / (len(trainloader) * batch_size)
        meanloss = float(losses.mean())
        print('Epoch:', epoch, 'Loss:', meanloss, 'Accuracy:', meanacc)
        accuracy['train'].append(meanacc)
        loss_record['train'].append(meanloss)
        # putting this in to keep the console clean
        time.sleep(0.5)
        
        
        # validation
        correct = 0
        losses = torch.tensor([])
        for i, data in enumerate(valloader, 0):
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
        meanacc = float(correct) / (len(valloader) * batch_size)
        meanloss = float(losses.mean())
        print('Validation epoch:', epoch, 'Loss:', meanloss, 'Accuracy:', meanacc)
        accuracy['valid'].append(meanacc)
        loss_record['valid'].append(meanloss)
        if meanloss < best_loss:
            torch.save(model.state_dict(), SAVE_PATH) 
            best_loss = meanloss 
        # putting this in to keep the console clean
        time.sleep(0.5)
if __name__ == "__main__":
    main()