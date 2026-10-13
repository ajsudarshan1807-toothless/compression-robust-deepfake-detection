import random
import io
from PIL import Image, ImageFilter
import numpy as np

def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """
    Compresses a PIL Image using JPEG compression at the specified quality.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    # Ensure it's fully loaded into memory before the buffer closes
    compressed_image.load()
    return compressed_image

def random_compress(image: Image.Image) -> Image.Image:
    """
    Applies JPEG compression with a random quality factor from [30, 50, 70, 90].
    """
    quality = random.choice([30, 50, 70, 90])
    return jpeg_compress(image, quality)

def social_media_aug(image: Image.Image) -> Image.Image:
    """
    Simulates aggressive social media artifacts to prevent domain shift.
    Applies scaling, blur, noise, and compression.
    """
    # 15% chance to do absolutely nothing (pass pristine image)
    if random.random() < 0.15:
        return image
        
    img = image.copy()
    
    # 1. Random Downscale & Upscale (Simulate resizing in WhatsApp)
    if random.random() < 0.5:
        w, h = img.size
        scale = random.uniform(0.3, 0.8)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        img = img.resize((w, h), Image.BICUBIC)
        
    # 2. Random Gaussian Blur (Simulate poor lens or denoising)
    if random.random() < 0.4:
        radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius))
        
    # 3. Random Gaussian Noise (Simulate sensor noise)
    if random.random() < 0.3:
        img_np = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(5, 25), img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
    # 4. Aggressive JPEG Compression
    if random.random() < 0.8:
        quality = random.randint(15, 80)
        img = jpeg_compress(img, quality)
        
    return img

def generate_sanity_check(image_paths, save_path="results/compression_sanity_check.png"):
    """
    Generates a grid of images showing different compression levels for a visual sanity check.
    """
    import matplotlib.pyplot as plt
    
    qualities = [90, 70, 50, 30]
    fig, axes = plt.subplots(len(image_paths), len(qualities) + 1, figsize=(15, 4 * len(image_paths)))
    
    for i, path in enumerate(image_paths):
        img = Image.open(path).convert('RGB')
        
        # Original
        if len(image_paths) == 1:
            ax = axes[0]
        else:
            ax = axes[i, 0]
        ax.imshow(img)
        ax.set_title("Original (Clean)")
        ax.axis('off')
        
        # Compressed
        for j, q in enumerate(qualities):
            comp_img = jpeg_compress(img, q)
            if len(image_paths) == 1:
                ax = axes[j+1]
            else:
                ax = axes[i, j+1]
            ax.imshow(comp_img)
            ax.set_title(f"QF: {q}")
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Sanity check saved to {save_path}")

if __name__ == "__main__":
    import os
    import glob
    # Test on random processed images if available
    imgs = glob.glob("data/processed/*/*.jpg")
    if len(imgs) > 0:
        sample = random.sample(imgs, min(3, len(imgs)))
        generate_sanity_check(sample)
    else:
        print("No images found in data/processed/ for sanity check.")
