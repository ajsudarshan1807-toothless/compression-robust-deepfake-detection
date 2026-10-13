import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import cv2

sys.path.append('src')
from models.video_model import TemporalFusionModel

def load_video_frames(path, seq_len=10, transform=None):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total_frames // seq_len)
    
    frames = []
    frame_idx = 0
    while cap.isOpened() and len(frames) < seq_len:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            if transform:
                img = transform(img)
            frames.append(img)
        frame_idx += 1
    cap.release()
    
    while len(frames) < seq_len:
        pad_img = Image.new('RGB', (224, 224), (0, 0, 0))
        if transform:
            pad_img = transform(pad_img)
        frames.append(pad_img)
        
    return torch.stack(frames).unsqueeze(0) # Add batch dimension

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    model = TemporalFusionModel(pretrained_spatial=False, freeze_clip=True, lstm_hidden_size=256)
    
    try:
        model.load_state_dict(torch.load('checkpoints/video_temporal_model.pt', map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
        
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    real_video = None
    fake_video = None
    
    real_dir = 'data/videos/real'
    fake_dir = 'data/videos/fake'
    
    if os.path.exists(real_dir) and os.listdir(real_dir):
        real_video = os.path.join(real_dir, os.listdir(real_dir)[0])
    if os.path.exists(fake_dir) and os.listdir(fake_dir):
        fake_video = os.path.join(fake_dir, os.listdir(fake_dir)[0])
        
    with torch.no_grad():
        if real_video:
            print(f"\nTesting Real Video: {real_video}")
            real_tensor = load_video_frames(real_video, transform=transform).to(device)
            real_pred = model(real_tensor).item()
            print(f"Real Video Prediction: {real_pred:.4f} (Closer to 0 is Real)")
            
        if fake_video:
            print(f"\nTesting Fake Video: {fake_video}")
            fake_tensor = load_video_frames(fake_video, transform=transform).to(device)
            fake_pred = model(fake_tensor).item()
            print(f"Fake Video Prediction: {fake_pred:.4f} (Closer to 1 is Fake)")

if __name__ == '__main__':
    main()
