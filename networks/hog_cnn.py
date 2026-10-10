import torch
import math
import timm
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F

class TrainableFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Start from Sobel filters (edge detectors)
        sobel_x = torch.tensor(
            [[[-1., 0., 1.],
              [-2., 0., 2.],
              [-1., 0., 1.]]]
        )
        sobel_y = torch.tensor(
            [[[-1., -2., -1.],
              [ 0.,  0.,  0.],
              [ 1.,  2.,  1.]]]
        )
        # Add a tiny random noise for learnability
        self.Gx = nn.Parameter(sobel_x.unsqueeze(0) + 0.01 * torch.randn(1, 1, 3, 3))
        self.Gy = nn.Parameter(sobel_y.unsqueeze(0) + 0.01 * torch.randn(1, 1, 3, 3))
        # More stable than BatchNorm2d for small channels
        self.norm = nn.GroupNorm(1, 2)
    def forward(self, x):
        x_f = F.conv2d(x, self.Gx, padding=1)
        y_f = F.conv2d(x, self.Gy, padding=1)
        # Compute magnitude and phase safely
        mag = torch.sqrt(x_f.square() + y_f.square() + 1e-6)
        phase = torch.atan2(y_f, x_f + 1e-8)
        phase = torch.nan_to_num(phase)
        combined = torch.cat((mag, phase), dim=1)
        combined = self.norm(combined)
        if torch.isnan(combined).any():
            print("Warning: NaN in TrainableFeatureExtractor output!")
        return combined

class SimpleCNN_anomally(nn.Module):
    def __init__(self, in_channels=2, embedding_dim=128):
        super().__init__()  # fixed: was super(SimpleCNN_Proto, self) which raises NameError
        
        # Block 1: 32 filters (224 -> 112)
        self.block1 = self._make_block(in_channels, 32)
        # Block 2: 64 filters (112 -> 56)
        self.block2 = self._make_block(32, 64)
        # Block 3: 128 filters (56 -> 28)
        self.block3 = self._make_block(64, 128)
        # Block 4: 256 filters (28 -> 14)
        self.block4 = self._make_block(128, 256)
        # Block 5: 512 filters (14 -> 7)
        self.block5 = self._make_block(256, 512)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Prototypical Embedding Head
        self.embedding_head = nn.Linear(512, embedding_dim)

    def _make_block(self, in_f, out_f):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_f),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Input expected: [Batch, 2, 224, 224]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        x = self.gap(x)            # Output: [Batch, 512, 1, 1]
        x = torch.flatten(x, 1)    # Output: [Batch, 512]
        
        x = self.embedding_head(x)
#        
        # L2 Normalization for Prototypical Distance calculation
        return F.normalize(x, p=2, dim=1)
        
class SimpleCNN_pooling(nn.Module):
    def __init__(self, in_channels=2):
        super(SimpleCNN_pooling, self).__init__() # Fixed class name
        
        self.block1 = self._make_block(in_channels, 32)
        self.block2 = self._make_block(32, 64)
        self.block3 = self._make_block(64, 128)
        self.block4 = self._make_block(128, 256)
        self.block5 = self._make_block(256, 512)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Define this so DualCNN can read it
        self.flatten_dim = 512 

    def _make_block(self, in_f, out_f):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_f),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # Note: Removing normalization here if you want to use the 
        # classifier's ReLU/Linear layers effectively.
        return x

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, img_size=224):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # 224 -> 224
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 112 -> 112
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 56 -> 56
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
        )
        self.flatten_dim = 128 * (img_size//8) * (img_size//8)

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # flatten

class DualCNN_pooling(nn.Module):
    def __init__(self, mlp_dim=512):
        super(DualCNN_pooling, self).__init__()
        self.cnn_branch = SimpleCNN_pooling(in_channels=2)

        # This now correctly evaluates to 512
        combined_dim = self.cnn_branch.flatten_dim 
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_dim // 2, 1)
        )

    def forward(self, x):
        # x: [Batch, 2, 224, 224]
        features = self.cnn_branch(x)
        out = self.classifier(features)
        return out



class DualCNN(nn.Module):
    def __init__(self, num_classes=1, mlp_dim=512):
        super(DualCNN, self).__init__()
        # Two CNN branches
        self.cnn_branch = SimpleCNN(in_channels=2)

        # Combined fully connected layers
        combined_dim = self.cnn_branch.flatten_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.ReLU(),
            nn.Linear(mlp_dim//2, 1)
        )

    def forward(self, x):
        combined = self.cnn_branch(x)
        out = self.classifier(combined)
        return out
        
class DualCNN_128_trainFilter(nn.Module):
    def __init__(self, num_classes=1, mlp_dim=128):
        super().__init__()
        self.feature_extractor = TrainableFeatureExtractor()
        self.cnn_branch = SimpleCNN(in_channels=2)
        combined_dim = self.cnn_branch.flatten_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.ReLU(),
            nn.Linear(mlp_dim//2, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        combined = self.cnn_branch(x)
        out = self.classifier(combined)
        return out

class DualCNN_128(nn.Module):
    """
    The proposed Grad + CNN model from the paper.
    SimpleCNN backbone (3 blocks, 32 -> 64 -> 128 filters) followed by
    the shared MLP head with hidden layers of 128 and 64 units.

    in_channels controls the input representation:
        2 = grayscale gradient (mag + dir), the paper's default
        6 = per channel RGB gradient (mag_R/G/B + dir_R/G/B), color ablation
    """
    def __init__(self, num_classes=1, mlp_dim=128, in_channels=2, img_size=224):
        super().__init__()
        self.cnn_branch = SimpleCNN(in_channels=in_channels, img_size=img_size)

        # Combined fully connected layers
        combined_dim = self.cnn_branch.flatten_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.ReLU(),
            nn.Linear(mlp_dim//2, 1)
        )

    def forward(self, x):
        combined = self.cnn_branch(x)
        out = self.classifier(combined)
        return out

class DualCNN_128_5ch(nn.Module):
    def __init__(self, num_classes=1, mlp_dim=128, img_size=224):
        super().__init__()
        # Branch 1: RGB (3 channels)
        self.rgb_branch = SimpleCNN(in_channels=3, img_size=img_size)
        
        # Branch 2: HOG (2 channels)
        self.hog_branch = SimpleCNN(in_channels=2, img_size=img_size)

        combined_dim = self.rgb_branch.flatten_dim + self.hog_branch.flatten_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1) # Output 1 for binary/regression
        )

    def forward(self, x):
        # x shape: [Batch, 5, H, W]
        
        # Slicing the tensor:
        # x[:, 0:3, :, :] -> Channels 0, 1, 2 (RGB)
        # x[:, 3:5, :, :] -> Channels 3, 4 (Mag, Dir)
        rgb_data = x[:, :3, :, :]
        hog_data = x[:, 3:, :, :]
        
        # Process branches
        rgb_feat = self.rgb_branch(rgb_data)
        hog_feat = self.hog_branch(hog_data)
        
        # Concatenate and classify
        combined = torch.cat((rgb_feat, hog_feat), dim=1)
        return self.classifier(combined)
        
        
class DualCNN_anomally(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()  # fixed: was super(DualCNN_Prototypical, self) which raises NameError
        self.cnn_branch = SimpleCNN_pooling(in_channels=2)

        backbone_dim = self.cnn_branch.flatten_dim 
        
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        features = self.cnn_branch(x)
        embeddings = self.projection_head(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
        

class RGB_anomally(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()  # fixed: was super(RGBCNN_Prototypical, self) which raises NameError
        self.cnn_branch = SimpleCNN_pooling(in_channels=3)
        backbone_dim = self.cnn_branch.flatten_dim 
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        features = self.cnn_branch(x)
        embeddings = self.projection_head(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class DualCNN_Prototypical(nn.Module):
    def __init__(self, embedding_dim=128, img_size=224):
        super().__init__()
        # Branch 1: RGB
        self.rgb_branch = SimpleCNN(in_channels=3, img_size=img_size)
        # Branch 2: HOG (Mag + Dir)
        self.hog_branch = SimpleCNN(in_channels=2, img_size=img_size)

        combined_dim = self.rgb_branch.flatten_dim + self.hog_branch.flatten_dim
        
        # Instead of 1 output, we output a high-dimensional vector
        self.embedder = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim) 
        )

    def forward(self, x):
        rgb_data = x[:, :3, :, :]
        hog_data = x[:, 3:, :, :]
        
        rgb_feat = self.rgb_branch(rgb_data)
        hog_feat = self.hog_branch(hog_data)
        
        combined = torch.cat((rgb_feat, hog_feat), dim=1)
        embedding = self.embedder(combined)
        
        # Normalize the embedding to a hypersphere (standard in metric learning)
        return F.normalize(embedding, p=2, dim=1)
        
        
class RealCenterLoss(nn.Module):
    def __init__(self, embedding_dim=128, margin=0.5):
        super().__init__()
        # This is the "Mean Vector" the model tries to learn
        self.center = nn.Parameter(torch.randn(1, embedding_dim))
        self.margin = margin # How far fakes must be pushed away

    def forward(self, embeddings, labels):
        """
        labels: 1 for Real, 0 for Fake
        """
        distances = torch.norm(embeddings - self.center, p=2, dim=1)
        
        # Loss for Real: Minimize distance to center
        loss_real = distances[labels == 1].pow(2).mean()
        
        # Loss for Fake: Maximize distance (up to a margin)
        # We use Hinge Loss: only penalize if the fake is too close to the Real Center
        loss_fake = F.relu(self.margin - distances[labels == 0]).pow(2).mean()
        
        # Handle cases where a batch might not have both classes
        loss = 0
        if not torch.isnan(loss_real): loss += loss_real
        if not torch.isnan(loss_fake): loss += loss_fake
            
        return loss

class HogCNN(nn.Module):
    def __init__(self, num_bins=9, mlp_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_bins, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 224  112

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 112  56

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 56  28
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)  # binary
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x