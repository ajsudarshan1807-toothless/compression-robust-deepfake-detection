import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from compression_aug import social_media_aug
from models.spatial_branch import SpatialBranch
from models.frequency_branch import FrequencyBranch
from models.fusion_model import FusionModel

class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, transform=None, use_compression_aug=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_compression_aug = use_compression_aug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        
        image = Image.open(img_name).convert('RGB')
        
        if self.use_compression_aug:
            image = social_media_aug(image)
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor([label], dtype=torch.float32)

def train_model(model, train_loader, val_loader, model_name, epochs=5, lr=1e-4, patience=3):
    device = torch.device("cpu") # Forced to CPU per constraints
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience_counter = 0
    
    save_path = f"checkpoints/{model_name}.pt"
    
    print(f"Starting training for {model_name}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    return model

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if splits exist
    if not os.path.exists('data/splits/train.csv'):
        print("Data splits not found. Please run data_prep.py first.")
        exit(1)
        
    val_dataset = DeepfakeDataset('data/splits/val.csv', transform=transform, use_compression_aug=False)
    train_aug_dataset = DeepfakeDataset('data/splits/train.csv', transform=transform, use_compression_aug=True)
    
    batch_size = 16 # Increased for efficiency
    
    # Using 4 workers and pin_memory for faster CPU data loading
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_aug_loader = DataLoader(train_aug_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    epochs = 3 # Reduced for CPU to just quickly retrain for AI detection
    
    # Model D: Fusion (Augmented)
    print("Training Fusion Model with Compression Augmentation on AI vs Real images...")
    model_d = FusionModel()
    train_model(model_d, train_aug_loader, val_loader, "model_d_fusion_compaug", epochs=epochs)
    
    print("All training completed.")
