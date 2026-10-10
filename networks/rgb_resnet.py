import torch
import torch.nn as nn
from torchvision import models

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        

        # === FREEZE ALL PARAMETERS FIRST ===
        for param in resnet.parameters():
            param.requires_grad = False

        # Replace fc layer
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 1)
        self.model = resnet

    def forward(self, x):
        return self.model(x)
