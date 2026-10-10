import torch
import torch.nn.functional as f
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter
from einops import rearrange
import math

class PatchesSplitting(object):
    def __init__(self, img_size=256, patch_size=32):
        self.patch_num = img_size//patch_size
        self.patch_size = patch_size

    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        patches = []
        for i in range(self.patch_num):
            for j in range(self.patch_num):
                patch = object[:, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size]
                patches.append(patch)
        return torch.stack(patches)

class HogPatch(object):
    def __init__(self):
        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0 ,1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        hist_list = []
        for patch in object:
            x_filtered = f.conv2d(patch, self.Gx, padding='same', groups=1)
            y_filtered = f.conv2d(patch, self.Gy, padding='same', groups=1)
            dir = torch.atan2(y_filtered, x_filtered)
            dir_hist = torch.histc(dir, max=torch.max(dir), min=torch.min(dir), bins=8)
            hist_list.append(dir_hist)
        return torch.stack(hist_list)

class GradTransform(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0 ,1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def normalize(self, x):
        denominator = torch.max(x) - torch.min(x)
        return (x - torch.min(x)) / (denominator + 1e-8)
    
    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        stacked = torch.cat((self.normalize(x_filtered), self.normalize(y_filtered)), dim=0)
        return stacked
    
class HogTransformScaled(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0 ,1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def normalize(self, x):
        denominator = torch.max(x) - torch.min(x)
        return (x - torch.min(x)) / (denominator + 1e-8)
    
    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        mag, dir = (x_filtered.square() + y_filtered.square()).sqrt(), torch.atan2(y_filtered, x_filtered)
        stacked = torch.cat((self.normalize(mag), self.normalize(dir)), dim=0)
        return stacked
        
class RawGradMagDir(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [0, 0, 0],
            [0, -1, 1],
            [0, 0, 0]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [0, -1, 0],
            [0,  1, 0],
            [0,  0, 0]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def normalize(self, x):
        denominator = torch.max(x) - torch.min(x)
        return (x - torch.min(x)) / (denominator + 1e-8)
    
    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        mag, dir = (x_filtered.square() + y_filtered.square()).sqrt(), torch.atan2(y_filtered, x_filtered)
        stacked = torch.cat((self.normalize(mag), self.normalize(dir)), dim=0)
        return stacked

class RGBGradMagDir(object):
    """
    Per channel FFD gradient transform for the color ablation.

    Applies the same Forward Finite Difference kernels as RawGradMagDir,
    but independently to each RGB channel via grouped convolution.
    Output is a 6 channel tensor:
        [mag_R, mag_G, mag_B, dir_R, dir_G, dir_B]
    Each channel is min-max normalized independently so that no single
    color channel dominates the input scale.
    """
    def __init__(self):
        kernel_x = torch.tensor([
            [0, 0, 0],
            [0, -1, 1],
            [0, 0, 0]
        ], dtype=torch.float32)

        kernel_y = torch.tensor([
            [0, -1, 0],
            [0,  1, 0],
            [0,  0, 0]
        ], dtype=torch.float32)

        # Shape [3, 1, 3, 3]: one kernel per input channel, groups=3
        self.Gx = kernel_x.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.Gy = kernel_y.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def normalize_per_channel(self, x):
        # x: [C, H, W]
        c_min = x.amin(dim=(1, 2), keepdim=True)
        c_max = x.amax(dim=(1, 2), keepdim=True)
        return (x - c_min) / (c_max - c_min + 1e-8)

    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        assert object.dim() == 3 and object.size(0) == 3, \
            "RGBGradMagDir expects an RGB tensor of shape [3, H, W]"

        x = object.unsqueeze(0)  # [1, 3, H, W]
        x_filtered = f.conv2d(x, self.Gx, padding='same', groups=3)
        y_filtered = f.conv2d(x, self.Gy, padding='same', groups=3)

        mag = (x_filtered.square() + y_filtered.square()).sqrt().squeeze(0)  # [3, H, W]
        dir = torch.atan2(y_filtered, x_filtered).squeeze(0)                 # [3, H, W]

        stacked = torch.cat((
            self.normalize_per_channel(mag),
            self.normalize_per_channel(dir)
        ), dim=0)  # [6, H, W]
        return stacked

class PrewittGrad(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -1, -1],
            [0,  0, 0],
            [1,  1, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def normalize(self, x):
        denominator = torch.max(x) - torch.min(x)
        return (x - torch.min(x)) / (denominator + 1e-8)
    
    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        mag, dir = (x_filtered.square() + y_filtered.square()).sqrt(), torch.atan2(y_filtered, x_filtered)
        stacked = torch.cat((self.normalize(mag), self.normalize(dir)), dim=0)
        return stacked
        
class ScharrGrad(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-3, -10, -3],
            [0,  0, 0],
            [3, 10, 3]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def normalize(self, x):
        denominator = torch.max(x) - torch.min(x)
        return (x - torch.min(x)) / (denominator + 1e-8)
    
    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        mag, dir = (x_filtered.square() + y_filtered.square()).sqrt(), torch.atan2(y_filtered, x_filtered)
        stacked = torch.cat((self.normalize(mag), self.normalize(dir)), dim=0)
        return stacked
        
class HogTransformScaled_5ch(object):
    def __init__(self):
        # We define the kernels for a single channel
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # Reshape to [Out_Channels, In_Channels, H, W]
        # We use 1 in_channel here and apply it to a grayscale version internally 
        # to get a single Magnitude/Direction map.
        self.Gx = kernel_x.reshape(1, 1, 3, 3)
        self.Gy = kernel_y.reshape(1, 1, 3, 3)

    def normalize(self, x):
        denominator = torch.max(x) - torch.min(x)
        return (x - torch.min(x)) / (denominator + 1e-8)

    def __call__(self, img_tensor):
        # img_tensor is [3, 224, 224] (RGB)
        
        # 1. Create a grayscale version for gradient calculation [1, 224, 224]
        # This ensures we get 1 Mag and 1 Dir channel regardless of RGB input
        gray = 0.2989 * img_tensor[0] + 0.5870 * img_tensor[1] + 0.1140 * img_tensor[2]
        gray = gray.unsqueeze(0).unsqueeze(0) # Shape: [1, 1, 224, 224]

        # 2. Apply Sobel Filters
        x_filtered = f.conv2d(gray, self.Gx, padding='same')
        y_filtered = f.conv2d(gray, self.Gy, padding='same')

        # 3. Calculate Mag and Dir
        mag = (x_filtered.square() + y_filtered.square()).sqrt().squeeze(0) # [1, 224, 224]
        direction = torch.atan2(y_filtered, x_filtered).squeeze(0)          # [1, 224, 224]

        # 4. Concatenate: RGB (3) + Mag (1) + Dir (1) = 5 Channels
        # img_tensor: [3, 224, 224]
        # mag: [1, 224, 224]
        # direction: [1, 224, 224]
        stacked = torch.cat((img_tensor, self.normalize(mag), self.normalize(direction)), dim=0)
        
        return stacked
        
    
class HogTransform(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0 ,1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        mag, dir = (x_filtered.square() + y_filtered.square()).sqrt(), torch.atan2(y_filtered, x_filtered)
        stacked = torch.stack((mag, dir), dim=1)
        return stacked.squeeze(0)
    
class HogFFT(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0 ,1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def fft_map(self, input_tensor):
        f = torch.fft.fft2(input_tensor)
        fshift = torch.fft.fftshift(f)
        magnitude = torch.abs(fshift)
        spectrum = torch.log1p(magnitude)
        return (spectrum/(torch.max(spectrum) + 1e-8))

    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        mag, dir = (x_filtered.square() + y_filtered.square()).sqrt(), torch.atan2(y_filtered, x_filtered)
        mag_fft = self.fft_map(mag)
        dir_fft = self.fft_map(dir)
        stacked = torch.stack((mag_fft, dir_fft), dim=1)
        return stacked.squeeze(0)

    
        
class Hog2Hist(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0 ,1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def normalize(self, x):
        return (x-torch.min(x))/(torch.max(x)-torch.min(x))
    
    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        mag, dir = (x_filtered.square() + y_filtered.square()).sqrt(), torch.atan2(y_filtered, x_filtered)
        mag_norm, dir_norm = self.normalize(mag), self.normalize(dir)
        mag_norm_hist, dir_norm_hist = torch.histc(mag_norm*255, bins=256, min=0, max=255), torch.histc(dir_norm*255, bins=256, min=0, max=255)
        # stacked = torch.stack((mag_norm_hist, dir_norm_hist), dim=1)
        # return stacked.unsqueeze(0)
    
        concatenated = torch.concatenate((mag_norm_hist, dir_norm_hist), dim=0)
        return concatenated.unsqueeze(0)


class HOGTransform(object):
    def __init__(self, num_bins=9, eps=1e-6):
        self.num_bins = num_bins
        self.eps = eps
        self.bin_width = math.pi / num_bins  # unsigned gradients [0, ?)
        self.registered = False

    def _init_kernels(self, device, dtype):
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            device=device, dtype=dtype
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            device=device, dtype=dtype
        ).view(1, 1, 3, 3)

        self.sobel_x = sobel_x
        self.sobel_y = sobel_y
        self.registered = True

    def __call__(self, x):
        """
        x: torch.Tensor [1, H, W] (grayscale)
        return: HOG feature map [num_bins, H, W]
        """

        if x.dim() != 3 or x.size(0) != 1:
            raise ValueError("HOGTransform expects input of shape [1, H, W]")

        # Add batch dimension
        x = x.unsqueeze(0)  # [1, 1, H, W]

        if not self.registered:
            self._init_kernels(x.device, x.dtype)

        # Sobel gradients
        gx = f.conv2d(x, self.sobel_x, padding=1)
        gy = f.conv2d(x, self.sobel_y, padding=1)

        # Magnitude and orientation
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)
        orientation = torch.atan2(gy, gx) % math.pi  # [0, ?)

        _, _, H, W = magnitude.shape

        # Initialize orientation bins
        hog = torch.zeros(
            1, self.num_bins, H, W,
            device=x.device, dtype=x.dtype
        )

        # Hard binning
        for b in range(self.num_bins):
            angle_start = b * self.bin_width
            angle_end = (b + 1) * self.bin_width

            mask = (orientation >= angle_start) & (orientation < angle_end)
            hog[:, b][mask.squeeze(1)] = magnitude.squeeze(1)[mask.squeeze(1)]

        return hog.squeeze(0)  # [num_bins, H, W]
        
class RGBHOGTransform:
    def __init__(self, image_size=224):
        self.rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.hog_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),           # -> [1, H, W]
            HOGTransform(num_bins=9),         # -> [9, H, W]
        ])

    def __call__(self, img):
        rgb = self.rgb_transform(img)
        hog = self.hog_transform(img)

        return {
            "rgb": rgb,   # [3, 224, 224]
            "hog": hog    # [9, 224, 224]
        }

        
#class HOGTransform(nn.Module):
#    def __init__(self, num_bins=9, cell_size=8):
#        super().__init__()
#        self.num_bins = num_bins
#        self.cell_size = cell_size
#
#        sobel_x = torch.tensor(
#            [[-1, 0, 1],
#             [-2, 0, 2],
#             [-1, 0, 1]], dtype=torch.float32
#        ).view(1, 1, 3, 3)
#
#        sobel_y = sobel_x.transpose(2, 3)
#
#        self.register_buffer("sobel_x", sobel_x)
#        self.register_buffer("sobel_y", sobel_y)
#
#    def forward(self, x_gray):
#    
#        gx = F.conv2d(x_gray, self.sobel_x, padding=1)
#        gy = F.conv2d(x_gray, self.sobel_y, padding=1)
#
#        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
#        orientation = torch.atan2(gy, gx) % math.pi
#
#        bin_width = math.pi / self.num_bins
#        bin_idx = torch.floor(orientation / bin_width).long()
#        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)
#
#        B, _, H, W = magnitude.shape
#        hog = torch.zeros(B, self.num_bins, H, W, device=x_gray.device)
#
#        for b in range(self.num_bins):
#            hog[:, b:b+1] = magnitude * (bin_idx == b)
#
#        hog = F.avg_pool2d(
#            hog,
#            kernel_size=self.cell_size,
#            stride=self.cell_size
#        )
#
#        hog = hog / torch.sqrt(torch.sum(hog**2, dim=1, keepdim=True) + 1e-6)
#
#        return hog

hog_hist = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        Hog2Hist(),
    ])

hog_hist_8D = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        PatchesSplitting(),
        HogPatch()
    ])

hog_256 = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        HogTransform(),
    ])

hog_224 = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        HogTransformScaled(),
    ])

grad_224 = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        HogTransformScaled(),
    ])
    
# Raw grayscale pixels for filters learned inside the model. Match the fixed
# FFD preset's image geometry without applying an image filter here.
gray_center_224 = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

raw_grad_224 = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        RawGradMagDir(),
    ])

rgb_grad_224 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),          # [3, 224, 224], no Grayscale
        RGBGradMagDir(),                # [6, 224, 224]
    ])

scharr_grad_224 = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ScharrGrad(),
    ])
    
prewitt_grad_224 = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        PrewittGrad(),
    ])
    
grad_224_crop = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.RandomCrop(224, pad_if_needed=True, padding_mode='symmetric'),
        HogTransformScaled()
    ])
    
grad_224_5ch = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # Produces [3, 224, 224]
    HogTransformScaled_5ch()
])

dxdy_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        GradTransform()
    ])

hog_224_vit = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        HogTransformScaled(),
        transforms.Normalize([0.5, 0.5], [0.5,0.5])
    ])

hogfft_224 = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        HogFFT()
    ])

hogfft_256 = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        HogFFT()
    ])

gray_224 = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


rgb_224 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
HoG_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        HOGTransform(num_bins=9)
    ])
