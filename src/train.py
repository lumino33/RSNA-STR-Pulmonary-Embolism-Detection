import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast

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
    batch_size = 32
    num_epochs = 10
    folds = [1, 0, 2, 3, 4]
    df = pd.read_csv("/home/hungld11/Documents/RSNA Competition/RSNA-STR-Pulmonary-Embolism-Detection/prepare/second image level.csv")
    
    for fold in folds:
    #iterator for training 
        learning_rate = 1e-3
        print(learning_rate)
        
        train_df = df.loc[df["fold"] != fold]
        val_df = df.loc[df["fold"] == fold]
    
        image_dirs = train_df.image_path.tolist()
        labels = train_df.pe_present_on_image.tolist()
        train_datagen = PEDataset(image_dirs=image_dirs, labels=labels, mode="train")
        trainloader = torch.utils.data.DataLoader(train_datagen, batch_size = batch_size, num_workers=1, shuffle=True, pin_memory=True)
    
        val_image_dirs = val_df.image_path.tolist()
        val_labels = val_df.pe_present_on_image.tolist()
        val_datagen = PEDataset(image_dirs=val_image_dirs, labels=val_labels, mode="val")
        valloader = DataLoader(val_datagen, batch_size = batch_size, num_workers=1, shuffle=False)
    
    # build model 
        model = EfficientNet("tf_efficientnet_b3_ns", pretrained=True, num_classes=1, channel=3, in_features=1536).to(device)
        
        criterion1 = nn.BCEWithLogitsLoss().to(device)
        # criterion2 = FocalLoss(gamma=2.0, device=device).to(device)
        val_criterion = nn.BCEWithLogitsLoss() #(pos_weight = torch.tensor(q_val*w))
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, eta_min=1e-5, last_epoch=-1)
        iters = len(trainloader)
    #training
        print("-------------------------------")
        print("-------------------------------")
        print('FOLD: {} | TRAIN: {} | VALID: {}'.format(fold, len(trainloader.dataset), len(valloader.dataset)))
        print("-------------------------------")
        print("-------------------------------")

    
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
            for i, data in bar:
                X, y = data
                X, y = X.to(torch.float).to(device), y.to(device)
                
                optimizer.zero_grad() #zero gradients
                with autocast():
                    output = model(X)
                    loss = criterion1(output,y) #+criterion2(output,y)#caculate losses
                loss.backward() #backward pass
                
                losses = torch.cat((losses, torch.tensor([loss.detach()])), 0)
                optimizer.step() #update parameters
                scheduler.step(epoch + i / iters)
                bar.set_postfix(loss=loss.item())
                #caculate mean accuracy and mean loss
                if (i % 20000 == 0)&(i > 0):
                    meanloss = float(losses.mean())
                    print('Train Epoch:', epoch, 'Iteration:', i, 'Loss:', meanloss)
                    print("-----------------------")
            
                    # validation
                    model.eval()
                    val_losses = torch.tensor([])
                    val_bar = tqdm(enumerate(valloader, 0), total=len(valloader))
                    for _, data in val_bar:
                        # forward pass
                        X, y = data
                        X = X.to(torch.float)
                        X, y = X.to(device), y.to(device)

                        
                        with torch.no_grad():
                            output = model(X)
                            # get number of accurate predictions
                            val_loss = val_criterion(output,y)
                        val_losses = torch.cat((val_losses, torch.tensor([val_loss.detach()])), 0)
                    # calculate mean accuracy and print
                    val_meanloss = float(val_losses.mean())
                    print('Validation epoch:', epoch,'Iteration:', i, 'Loss:', val_meanloss)
                    
                    if val_meanloss < best_loss:
                        torch.save(model.state_dict(), SAVE_PATH) 
                        best_loss = val_meanloss 
                    # scheduler.step(val_meanloss)
                    # print(f"End of epoch {epoch}, lr =", optimizer.param_groups[0]['lr'])
                    # print(optimizer.param_groups[0]['lr'])
                    model.train()
                    print("-------------------")
                    
            
if __name__ == "__main__":
    main()