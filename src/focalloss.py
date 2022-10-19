
import torch, sys, os, pdb
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, gamma = 1.0, device = "cpu"):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, dtype = torch.float32).to(device)
        self.eps = 1e-6

    def forward(self, input, target):
        # input are the probabilities, if not, uncomment the below code prob ...
        # input and target shape: (bs, n_classes)
        # sigmoid
        # probs = torch.sigmoid(input)
        log_probs = -torch.log(input)

        focal_loss = torch.sum(torch.pow(1-input + self.eps, self.gamma).mul(log_probs).mul(target), dim=1)
        # bce_loss = torch.sum(log_probs.mul(target), dim = 1)
        
        return focal_loss.mean() # bce_loss
