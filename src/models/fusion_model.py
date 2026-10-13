import torch
import torch.nn as nn
from .spatial_branch import SpatialBranch
from .frequency_branch import FrequencyBranch
from .clip_branch import ClipBranch

class FusionModel(nn.Module):
    def __init__(self, pretrained_spatial=True):
        super(FusionModel, self).__init__()
        
        self.spatial = SpatialBranch(pretrained=pretrained_spatial)
        self.frequency = FrequencyBranch()
        
        fusion_dim = self.spatial.feature_dim + self.frequency.feature_dim # 1280 + 128 = 1408
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features from both branches
        spatial_features = self.spatial(x, return_features=True)
        frequency_features = self.frequency(x, return_features=True)
        
        # Concatenate features
        fused = torch.cat((spatial_features, frequency_features), dim=1)
        
        # Classification
        out = self.fusion_head(fused)
        return out

class FusionModelV2(nn.Module):
    """
    V2 uses OpenAI CLIP as the spatial feature extractor, completely eliminating
    the semantic bias (e.g. human=fake) present in the training dataset.
    """
    def __init__(self, pretrained_spatial=True, freeze_clip=True):
        super(FusionModelV2, self).__init__()
        
        self.spatial = ClipBranch(pretrained=pretrained_spatial, freeze=freeze_clip)
        self.frequency = FrequencyBranch()
        
        fusion_dim = self.spatial.feature_dim + self.frequency.feature_dim # 768 + 128 = 896
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_features = self.spatial(x, return_features=True)
        frequency_features = self.frequency(x, return_features=True)
        
        fused = torch.cat((spatial_features, frequency_features), dim=1)
        
        out = self.fusion_head(fused)
        return out
