import os
import pandas as pd
import numpy as np
from PIL import Image
import io
import requests
import pyarrow.parquet as pq
from tqdm import tqdm
import uuid

def setup_directories():
    dirs = [
        'data/raw', 
        'data/processed/real', 
        'data/processed/fake', 
        'data/splits', 
        'src/models', 
        'checkpoints', 
        'results'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def download_file(url, path):
    if not os.path.exists(path):
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

def process_and_save_dataset():
    print("Downloading Parquet files...")
    
    url_real = "https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets/resolve/main/data/train-00000-of-00006-336b26d54a26e17a.parquet"
    url_fake = "https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets/resolve/main/data/train-00005-of-00006-4bd152a5ab76dba7.parquet"
    
    path_real = "data/raw/train_chunk_0.parquet"
    path_fake = "data/raw/train_chunk_5.parquet"
    
    download_file(url_real, path_real)
    download_file(url_fake, path_fake)
    
    print("Extracting full images (no face cropping for AI Generation detection)...")
    
    real_count = 0
    fake_count = 0
    max_per_class = 5000
    
    real_records = []
    fake_records = []
    
    # Process Real
    print("Processing Real Images...")
    table_real = pq.read_table(path_real)
    df_real = table_real.to_pandas()
    for i, row in tqdm(df_real.iterrows(), total=len(df_real)):
        if real_count >= max_per_class: break
        if row['label'] != 0: continue
        try:
            img = Image.open(io.BytesIO(row['image']['bytes']))
            if img.mode != 'RGB': img = img.convert('RGB')
            resized = img.resize((224, 224))
            filename = f"{uuid.uuid4().hex}.jpg"
            save_path = os.path.join('data/processed/real', filename)
            resized.save(save_path, quality=100)
            real_records.append({'filename': save_path, 'label': 0})
            real_count += 1
        except: continue
            
    # Process Fake
    print("Processing Fake Images...")
    table_fake = pq.read_table(path_fake)
    df_fake = table_fake.to_pandas()
    for i, row in tqdm(df_fake.iterrows(), total=len(df_fake)):
        if fake_count >= max_per_class: break
        if row['label'] != 1: continue
        try:
            img = Image.open(io.BytesIO(row['image']['bytes']))
            if img.mode != 'RGB': img = img.convert('RGB')
            resized = img.resize((224, 224))
            filename = f"{uuid.uuid4().hex}.jpg"
            save_path = os.path.join('data/processed/fake', filename)
            resized.save(save_path, quality=100)
            fake_records.append({'filename': save_path, 'label': 1})
            fake_count += 1
        except: continue
    
    print(f"Successfully processed {real_count} real images and {fake_count} fake images.")
    
    # Save stats
    with open('results/dataset_stats.txt', 'w') as f:
        f.write(f"Real images: {real_count}\n")
        f.write(f"Fake images: {fake_count}\n")
        f.write(f"Total: {real_count + fake_count}\n")
        
    # Combine and shuffle
    all_records = real_records + fake_records
    final_df = pd.DataFrame(all_records)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split 70/15/15
    n = len(final_df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = final_df.iloc[:train_end]
    val_df = final_df.iloc[train_end:val_end]
    test_df = final_df.iloc[val_end:]
    
    train_df.to_csv('data/splits/train.csv', index=False)
    val_df.to_csv('data/splits/val.csv', index=False)
    test_df.to_csv('data/splits/test.csv', index=False)
    
    print("Splits saved to data/splits/")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    setup_directories()
    process_and_save_dataset()
