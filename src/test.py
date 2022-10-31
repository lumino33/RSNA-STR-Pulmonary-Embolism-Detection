import torch
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from tqdm import tqdm

from dataset import PEDataset
from model import EfficientNet

#Load Data
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    folds = [0, 1, 2, 3, 4]
    for fold in folds:
        SAVE_PATH = "/home/hungld11/Documents/RSNA Competition/saved model/y"+str(fold)+".pth" 
        
        df_test = pd.read_csv('/home/hungld11/Documents/RSNA Competition/RSNA-STR-Pulmonary-Embolism-Detection/prepare/third image level.csv') 
        test_image_dirs = df_test[df_test["fold"]==fold].image_path.tolist()
        test_labels = df_test[df_test["fold"]==fold].pe_present_on_image.tolist()

        test_datagen = PEDataset(image_dirs=test_image_dirs, labels=test_labels, mode="test")
        testloader = DataLoader(test_datagen, batch_size = batch_size, num_workers=1, shuffle=False)

        model = EfficientNet("tf_efficientnet_b6_ns", pretrained=False, num_classes=1, channel=3, in_features=2304).to(device)
        model.load_state_dict(torch.load(SAVE_PATH))
        model.eval()
        
        criterion = nn.BCEWithLogitsLoss().to(device)
        

        losses = torch.tensor([])
        for i, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
            # forward pass
            X, y = data
            X = X.to(torch.float)
            X, y = X.to(device), y.to(device)
            
            with torch.no_grad():
                output = model(X)
            loss = criterion(output, y)
            losses = torch.cat((losses, torch.tensor([loss.detach()])), 0)
        # calculate mean accuracy and print
        meanloss = float(losses.mean())
        print("Fold:", fold, 'Loss:', meanloss)
        
        # putting this in to keep the console clean
        time.sleep(0.5)
    
if __name__ == "__main__":
    main()