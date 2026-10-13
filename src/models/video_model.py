import torch
import torch.nn as nn
from .fusion_model import FusionModelV2

class TemporalFusionModel(nn.Module):
    """
    Temporal Sequence Model for AI Video Deepfake Detection.
    Uses FusionModelV2 (CLIP + DWT Frequency) to extract features per frame,
    then passes the sequence of features through an LSTM to detect temporal 
    inconsistencies (flickering, face-swapping glitches).
    """
    def __init__(self, pretrained_spatial=True, freeze_clip=True, lstm_hidden_size=256, lstm_layers=1):
        super(TemporalFusionModel, self).__init__()
        
        # Base Image Feature Extractor
        self.frame_extractor = FusionModelV2(pretrained_spatial=pretrained_spatial, freeze_clip=freeze_clip)
        
        # Total extracted feature dim = 768 (CLIP) + 128 (Frequency) = 896
        self.feature_dim = self.frame_extractor.spatial.feature_dim + self.frame_extractor.frequency.feature_dim
        
        # LSTM for Temporal Sequence Modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.5 if lstm_layers > 1 else 0
        )
        
        # Final classification head for the video sequence
        self.video_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape expects: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()
        
        # Flatten batch and sequence dimensions to process all frames through the CNN
        x_flat = x.view(-1, c, h, w)
        
        # Extract features (Spatial + Frequency) for each frame
        spatial_features = self.frame_extractor.spatial(x_flat, return_features=True)
        frequency_features = self.frame_extractor.frequency(x_flat, return_features=True)
        frame_features = torch.cat((spatial_features, frequency_features), dim=1) # Shape: (batch * seq_len, 896)
        
        # Reshape back to sequences: (batch_size, sequence_length, feature_dim)
        sequence_features = frame_features.view(batch_size, seq_len, self.feature_dim)
        
        # Pass sequence through LSTM
        lstm_out, (hn, cn) = self.lstm(sequence_features)
        
        # Use the output of the last time step for classification
        last_hidden_state = lstm_out[:, -1, :] # Shape: (batch_size, lstm_hidden_size)
        
        # Classify the video
        out = self.video_head(last_hidden_state)
        return out
