import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from compression_aug import jpeg_compress
from models.spatial_branch import SpatialBranch
from models.fusion_model import FusionModel

class EvaluatorDataset(Dataset):
    def __init__(self, csv_file, transform, qf=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.qf = qf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        
        image = Image.open(img_name).convert('RGB')
        
        if self.qf is not None:
            image = jpeg_compress(image, self.qf)
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def evaluate():
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Models
    model_a = SpatialBranch().to(device)
    model_a.load_state_dict(torch.load("checkpoints/model_a_spatial_clean.pt", map_location=device))
    model_a.eval()
    
    model_c = FusionModel().to(device)
    model_c.load_state_dict(torch.load("checkpoints/model_c_fusion_clean.pt", map_location=device))
    model_c.eval()
    
    model_d = FusionModel().to(device)
    model_d.load_state_dict(torch.load("checkpoints/model_d_fusion_compaug.pt", map_location=device))
    model_d.eval()
    
    models = {'Model A': model_a, 'Model C': model_c, 'Model D': model_d}
    compressions = {'Clean': None, 'QF90': 90, 'QF70': 70, 'QF50': 50, 'QF30': 30}
    
    results = []
    plot_data = {'Model A': [], 'Model C': [], 'Model D': []}
    
    for qf_name, qf_val in compressions.items():
        print(f"Evaluating on {qf_name} test set...")
        test_dataset = EvaluatorDataset('data/splits/test.csv', transform=transform, qf=qf_val)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        for model_name, model in models.items():
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    
                    probs = outputs.squeeze().cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    
                    # Handle single item batch edge cases
                    if probs.ndim == 0:
                        probs = [probs]
                        preds = [preds]
                    
                    all_probs.extend(probs)
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, zero_division=0)
            rec = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = 0.5 # Default if only one class is present in test set for some reason
            
            results.append({
                'Model': model_name,
                'Compression': qf_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'AUC-ROC': auc
            })
            
            plot_data[model_name].append(acc)

    df_results = pd.DataFrame(results)
    df_results.to_csv('results/ablation_table.csv', index=False)
    print("Saved results to ablation_table.csv")
    
    # Plotting
    labels = list(compressions.keys())
    x = range(len(labels))
    
    plt.figure(figsize=(10, 6))
    for model_name, accs in plot_data.items():
        plt.plot(x, accs, marker='o', label=model_name)
        
    plt.xticks(x, labels)
    plt.ylabel('Accuracy')
    plt.xlabel('Compression Level')
    plt.title('Accuracy vs Compression Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('results/accuracy_vs_compression.png')
    print("Saved plot to accuracy_vs_compression.png")

if __name__ == "__main__":
    evaluate()
