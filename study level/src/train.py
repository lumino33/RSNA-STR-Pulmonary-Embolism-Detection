import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import EfficientNetB0, SEResNeXt
from dataset import ExamPEDataset
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyper-parameter
    batch_size = 8
    learning_rate = 1e-3
    num_epochs =100
    folds = [0,1,2,3,4]
    loss_weight_dict = {
                     'negative_exam_for_pe': 0.0736196319,
                     'indeterminate': 0.09202453988,
                     'chronic_pe': 0.1042944785,
                     'acute_and_chronic_pe': 0.1042944785,
                     'central_pe': 0.1877300613,
                     'leftsided_pe': 0.06257668712,
                     'rightsided_pe': 0.06257668712,
                     'rv_lv_ratio_gte_1': 0.2346625767,
                     'rv_lv_ratio_lt_1': 0.0782208589,
                   }
    
    #load data
    df = pd.read_csv("/home/hungld11/Documents/RSNA Competition/RSNA-STR-Pulmonary-Embolism-Detection/study level/prepare/exam_level.csv")
    exam_data = torch.load("/home/hungld11/Documents/RSNA Competition/RSNA-STR-Pulmonary-Embolism-Detection/study level/exam image/data_exam.pt")
    
    #train k-folds
    loss_record_train = {0: [], 1:[], 2:[], 3:[], 4:[]}
    loss_record_val = {0: [], 1:[], 2:[], 3:[], 4:[]}

    for fold in folds:
        #load dataset
        train_df = df.loc[df["fold"] != fold]
        val_df = df.loc[df["fold"] == fold]
        
        train_keys = list(('/home/hungld11/Documents/RSNA Competition/init train/'+train_df.StudyInstanceUID+'/'+ train_df.SeriesInstanceUID+'/'))
        val_keys = list(('/home/hungld11/Documents/RSNA Competition/init train/'+val_df.StudyInstanceUID+'/'+ val_df.SeriesInstanceUID+'/'))
        
        train_labels = train_df[["negative_exam_for_pe","indeterminate","chronic_pe", 
                                 "acute_and_chronic_pe", "central_pe", "leftsided_pe", 
                                 "rightsided_pe", "rv_lv_ratio_gte_1", "rv_lv_ratio_lt_1"]].values.tolist()
        val_labels = val_df[["negative_exam_for_pe","indeterminate","chronic_pe", 
                                 "acute_and_chronic_pe", "central_pe", "leftsided_pe", 
                                 "rightsided_pe", "rv_lv_ratio_gte_1", "rv_lv_ratio_lt_1"]].values.tolist()
        
        train_dataset = ExamPEDataset(train_keys, train_labels, exam_data)
        val_dataset = ExamPEDataset(val_keys, val_labels, exam_data)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

        #model
        model = SEResNeXt("seresnext50_32x4d", pretrained=True, channel=1, num_classes=9).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(list(loss_weight_dict.values())).to(device))
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        #train-val
        print("-----------------------")
        print('FOLD: {} | TRAIN: {} | VALID: {}'.format(fold, len(train_loader.dataset), len(val_loader.dataset)))
        print("-----------------------")
        best_loss = 9999
        SAVE_PATH = "/home/hungld11/Documents/RSNA Competition/saved model/test_exam_best"+str(fold)+".pth"
        #train
        for epoch in range(num_epochs):
            losses = torch.tensor([])
            bar = tqdm(enumerate(train_loader, 0), total=len(train_loader))
            for i, data in bar:
                X, y = data
                X = X.to(torch.float)
                X, y = X.to(device), y.to(device)
                
                optimizer.zero_grad() #zero gradients
                output = model(X)
                loss = criterion(output, y.squeeze(dim=1)) #caculate losses
                loss.backward() #backward pass
                
                losses = torch.cat((losses, torch.tensor([loss.detach()])), 0)
                optimizer.step() #update parameters
                
                bar.set_postfix(loss=loss.item(), )
            scheduler.step()
            #caculate mean loss
            meanloss = float(losses.mean())
            print('Epoch:', epoch, 'Loss:', meanloss)
            loss_record_train[fold].append(meanloss)  
         
            # validation
            losses = torch.tensor([])
            for i, data in enumerate(val_loader, 0):
                # forward pass
                X, y = data
                X = X.to(torch.float)
                X, y = X.to(device), y.to(device)
                
                with torch.no_grad():
                    output = model(X)
                # get number of accurate predictions
                loss = criterion(output, y.squeeze(dim=1))
                losses = torch.cat((losses, torch.tensor([loss.detach()])), 0)
            # calculate mean accuracy and print
            meanloss = float(losses.mean())
            print('Validation epoch:', epoch, 'Loss:', meanloss)
            loss_record_val[fold].append(meanloss)
            if meanloss < best_loss:
                torch.save(model.state_dict(), SAVE_PATH) 
                best_loss = meanloss 
        
if __name__=="__main__":
    main()
        