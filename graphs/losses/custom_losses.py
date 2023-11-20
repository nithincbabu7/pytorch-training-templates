import torch
import torchvision.transforms as transforms
from torch import nn
from random import randint
import torch.nn.functional as F

class CustomLoss(nn.Module):
    
    def __init__(self, exponent=2):
        super(CustomLoss, self).__init__()
        # init functions
        self.exponent = exponent

    def forward(self, y_pred, y):
        return torch.mean(torch.pow(y_pred - y, self.exponent))     # implements a mean of a custom powered error