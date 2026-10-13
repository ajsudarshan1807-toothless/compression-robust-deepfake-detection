import torch
import torch.nn as nn
import pywt
import numpy as np

def dwt_features(image_tensor):
    """
    Extracts frequency features using 2-level Haar DWT.
    Expects input shape: (B, C, H, W).
    Returns shape: (B, C*3, H//4, W//4) where C*3 represents the stacked high-frequency subbands.
    Note: Since pywt doesn't operate on PyTorch tensors directly in batched GPU memory, 
    we process on CPU and convert back.
    """
    # Move to CPU for pywt
    device = image_tensor.device
    np_img = image_tensor.detach().cpu().numpy()
    b, c, h, w = np_img.shape
    
    features = []
    for i in range(b):
        channel_features = []
        for j in range(c):
            # 2-level DWT
            coeffs = pywt.wavedec2(np_img[i, j], 'haar', level=2)
            # coeffs[0] is LL2 (approximation), coeffs[1] is (LH2, HL2, HH2), coeffs[2] is (LH1, HL1, HH1)
            # We use the level 2 high frequency subbands: LH2, HL2, HH2
            # Their shape is (H//4, W//4)
            lh2, hl2, hh2 = coeffs[1]
            channel_features.extend([lh2, hl2, hh2])
        # Stack channels: 3 channels * 3 subbands = 9 channels
        features.append(np.stack(channel_features, axis=0))
        
    features = np.stack(features, axis=0)
    return torch.tensor(features, dtype=torch.float32, device=device)

class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=9, num_classes=1):
        super(FrequencyBranch, self).__init__()
        
        self.cnn = nn.Sequential(
            # Input: 9 channels (from DWT)
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global average pooling
        )
        
        self.feature_dim = 128
        
        # We append a single-neuron classification head for standalone testing (Model B)
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, return_features=False):
        # x is (B, C, H, W) RGB image tensor. First apply DWT.
        x_dwt = dwt_features(x)
        features = self.cnn(x_dwt)
        features = features.view(features.size(0), -1) # Flatten to 128-d
        
        if return_features:
            return features
        return self.head(features)
