
import torch
import torch.nn as nn
from torchvision import models

class Resnet18_2CH(nn.Module):
    def __init__(self, in_channels=2, mlp_dim=128):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.load_state_dict(torch.load("checkpoints/pretrained/resnet18.pth"))
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels=in_channels,  # Change from 3 to 2
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )

        # === OPTIONAL: COPY PRETRAINED WEIGHTS TO NEW CONV ===
        with torch.no_grad():
            resnet.conv1.weight[:, :2] = old_conv.weight[:, :2]
        # Freeze conv layers (optional)

        # === FREEZE ALL PARAMETERS FIRST ===
        for param in resnet.parameters():
            param.requires_grad = False

        # === UNFREEZE ONLY FIRST CONV AND FC ===
        for param in resnet.conv1.parameters():
            param.requires_grad = True

        for param in resnet.fc.parameters():
            param.requires_grad = True

        # Replace fc layer
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.ReLU(),
            nn.Linear(mlp_dim//2, 1)
        )
        self.model = resnet

    def forward(self, x):
        return self.model(x)

class Resnet18(nn.Module):
    def __init__(self, mlp_dim=128):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.load_state_dict(torch.load("checkpoints/pretrained/resnet18.pth"))
        # === FREEZE ALL PARAMETERS FIRST ===
        for param in resnet.parameters():
            param.requires_grad = False

        # Replace fc layer
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.ReLU(),
            nn.Linear(mlp_dim//2, 1)
        )

        resnet.fc = nn.Linear(num_ftrs, 1)
        self.model = resnet

    def forward(self, x):
        return self.model(x)


class Resnet18_2CH_unlocked(nn.Module):
    def __init__(self, in_channels=2, mlp_dim=128):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.load_state_dict(torch.load("checkpoints/pretrained/resnet18.pth"))
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels=in_channels,  # Change from 3 to 2
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )

        # === OPTIONAL: COPY PRETRAINED WEIGHTS TO NEW CONV ===
        with torch.no_grad():
            resnet.conv1.weight[:, :2] = old_conv.weight[:, :2]
        # Freeze conv layers (optional)

        # Replace fc layer
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.ReLU(),
            nn.Linear(mlp_dim//2, 1)
        )
        self.model = resnet

    def forward(self, x):
        return self.model(x)

class Resnet18_unlocked(nn.Module):
    def __init__(self, mlp_dim=128):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.load_state_dict(torch.load("checkpoints/pretrained/resnet18.pth"))

        # Replace fc layer
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.ReLU(),
            nn.Linear(mlp_dim//2, 1)
        )

        resnet.fc = nn.Linear(num_ftrs, 1)
        self.model = resnet

    def forward(self, x):
        return self.model(x)

