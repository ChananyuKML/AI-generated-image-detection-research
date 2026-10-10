import torch
import torch.nn as nn
import timm

class ViTRGB(nn.Module):
    def __init__(self, mlp_dim=128):
        super().__init__()

        vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        
        state_dict = torch.load("checkpoints/pretrained/vit_base_patch16_224.pth", weights_only=True)
        vit.load_state_dict(state_dict)

        for param in vit.parameters():
            param.requires_grad = False

        num_ftrs = vit.head.in_features
        
        vit.head = nn.Sequential(
            nn.Linear(num_ftrs, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_dim // 2, 1)
        )
        
        self.model = vit

    def forward(self, x):
        return self.model(x)
        

class ViT2ch(nn.Module):
    def __init__(self, mlp_dim=128):
        super().__init__()
        # 1. Create the base architecture
        vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        
        # 2. Modify patch_embed to 2 channels
        old_proj = vit.patch_embed.proj
        vit.patch_embed.proj = nn.Conv2d(
            in_channels=2,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding
        )

        # 3. Load the checkpoint
        state_dict = torch.load("checkpoints/pretrained/vit_base_patch16_224.pth", weights_only=True)

        # === THE FIX: SLICE PRETRAINED WEIGHTS ===
        # The checkpoint has shape [768, 3, 16, 16]
        # We slice the second dimension to get [768, 2, 16, 16]
        if 'patch_embed.proj.weight' in state_dict:
            state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'][:, :2, :, :]
        # =========================================

        # 4. Load weights (now strict=True can be used if desired)
        vit.load_state_dict(state_dict, strict=False)

        # 5. Freeze parameters
        for param in vit.parameters():
            param.requires_grad = False
        
        # Keep patch_embed and head trainable
        for param in vit.patch_embed.parameters():
            param.requires_grad = True

        # 6. Replace the head
        num_ftrs = vit.head.in_features
        vit.head = nn.Sequential(
            nn.Linear(num_ftrs, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_dim // 2, 1)
        )
        
        self.model = vit

    def forward(self, x):
        return self.model(x)