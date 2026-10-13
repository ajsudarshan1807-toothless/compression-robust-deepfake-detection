import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SpatialBranch(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(SpatialBranch, self).__init__()
        
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)
            
        # EfficientNet-B0 has a classifier layer named `classifier`
        # We replace it with an identity to extract the 1280-d feature vector
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # We optionally add our own classification head for Model A
        self.head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
            nn.Sigmoid()
        )
        
        self.feature_dim = in_features

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        if return_features:
            return features
        return self.head(features)
