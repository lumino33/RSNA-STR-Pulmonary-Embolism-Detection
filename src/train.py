import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from statistics import mean
import time

from model import EfficientNet
from dataset import PEDataset
from focalloss import FocalLoss

def main():
    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyper-parameter
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 15
    folds = [0, 1, 2, 3, 4]
    df = pd.read_csv("/home/hungld11/Documents/RSNA Competition/RSNA-STR-Pulmonary-Embolism-Detection/prepare/image level.csv")
    w = 0.07361963 #Image-level log loss weight
    
    for fold in folds:
    #iterator for training 
        train_df = df.loc[df["fold"] != fold]
        val_df = df.loc[df["fold"] == fold]
        
        q_val = len(val_df[val_df.pe_present_on_image == 1])/len(val_df)
        
        print("q_val =", q_val)
    
        image_dirs = train_df.image_path.tolist()
        labels = train_df.pe_present_on_image.tolist()
        train_datagen = PEDataset(image_dirs=image_dirs, labels=labels, mode="train")
        trainloader = torch.utils.data.DataLoader(train_datagen, batch_size = batch_size, num_workers=2, shuffle=True, pin_memory=True)
    
        val_image_dirs = val_df.image_path.tolist()
        val_labels = val_df.pe_present_on_image.tolist()
        val_datagen = PEDataset(image_dirs=val_image_dirs, labels=val_labels, mode="val")
        valloader = DataLoader(val_datagen, batch_size = batch_size, num_workers=2, shuffle=False)
    
    # build model 
        model = EfficientNet("tf_efficientnet_b1_ns", pretrained=True, num_classes=1, channel=3, in_features=1280).to(device)
        
        criterion1 = nn.BCEWithLogitsLoss().to(device)
        criterion2 = FocalLoss(gamma=2.0, device=device).to(device)
        val_criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(q_val*w))
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=1)
    
    #training
        print("-----------------------")
        print('FOLD: {} | TRAIN: {} | VALID: {}'.format(fold, len(trainloader.dataset), len(valloader.dataset)))
        print("-----------------------")

    
        best_loss = 9999
        SAVE_PATH = "/home/hungld11/Documents/RSNA Competition/saved model/image_level_best_"+str(fold)+".pth"

        for epoch in range(num_epochs):
            #train
            # if epoch < 3:
            #     for param in model.base.parameters():
            #         param.requires_grad = False
            # else:
            #     for param in model.parameters():
            #         param.requires_grad = True
            losses = torch.tensor([])
            bar = tqdm(enumerate(trainloader, 0), total=len(trainloader))
            for _, data in bar:
                X, y = data
                X = X.to(torch.float)
                X, y = X.to(device), y.to(device)
                
                optimizer.zero_grad() #zero gradients
                output = model(X)
                
                loss = criterion1(output,y) #+criterion2(output,y)#caculate losses
                loss.backward() #backward pass
                
                losses = torch.cat((losses, torch.tensor([loss.detach()])), 0)
                optimizer.step() #update parameters
                
                bar.set_postfix(loss=loss.item())
            #caculate mean accuracy and mean loss
            meanloss = float(losses.mean())
            print('Train Epoch:', epoch, 'Loss:', meanloss)
            # putting this in to keep the console clean
            time.sleep(0.5)
            
            # validation
            losses = torch.tensor([])
            val_bar = tqdm(enumerate(trainloader, 0), total=len(trainloader))
            for _, data in val_bar:
                # forward pass
                X, y = data
                X = X.to(torch.float)
                X, y = X.to(device), y.to(device)
                
                with torch.no_grad():
                    model.eval()
                    output = model(X)
                # get number of accurate predictions
                loss = val_criterion(output, y)
                losses = torch.cat((losses, torch.tensor([loss.detach()])), 0)
                bar.set_postfix(loss=loss.item())
            # calculate mean accuracy and print
            meanloss = float(losses.mean())
            print('Validation epoch:', epoch, 'Loss:', meanloss)
            
            if meanloss < best_loss:
                torch.save(model.state_dict(), SAVE_PATH) 
                best_loss = meanloss 
            # putting this in to keep the console clean
            scheduler.step(meanloss)
            print(f"End of epoch {epoch}, lr =", optimizer.param_groups[0]['lr'])
            print(optimizer.param_groups[0]['lr'])
            print("-------------------")
            time.sleep(0.5)
            
if __name__ == "__main__":
    main()