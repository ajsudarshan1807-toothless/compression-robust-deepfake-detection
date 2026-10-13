import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class ClipBranch(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(ClipBranch, self).__init__()
        
        # We use openai/clip-vit-base-patch32, which is small and fast but incredibly robust
        if pretrained:
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from transformers import CLIPVisionConfig
            config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPVisionModel(config)
            
        self.feature_dim = 768 # CLIP ViT-B/32 has hidden_size of 768
        
        # Freeze the CLIP model to prevent destroying its pre-trained semantic knowledge
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Full fine-tuning or partial fine-tuning
            # Let's support unfreezing just the last N layers to save memory
            unfreeze_layers = 6
            
            # Freeze everything first
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Unfreeze the last few transformer layers
            for layer in self.model.encoder.layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
            # Unfreeze the post_layernorm
            for param in self.model.post_layernorm.parameters():
                param.requires_grad = True
                
    def forward(self, x, return_features=True):
        # x is (B, C, 224, 224) normalized RGB tensor
        outputs = self.model(pixel_values=x)
        # pooler_output is (B, 768)
        features = outputs.pooler_output
        return features
