import os
from PIL import Image

# Ensure the output directory exists
os.makedirs('poc_inventory', exist_ok=True)

print("Establishing stream to image dataset...")
try:
    from datasets import load_dataset
    dataset = load_dataset('zh-plus/tiny-imagenet', split='train', streaming=True)
    
    print("Extracting 10,000 images...")
    for i, example in enumerate(dataset.take(10000)):
        img = example['image'].convert("RGB")
        filename = f"poc_inventory/NAT_{i:05d}.jpg"
        img.save(filename)
        
        if (i + 1) % 1000 == 0:
            print(f"Successfully processed {i + 1} images...")
            
except Exception as e:
    print(f"Attempting alternative stream... ({e})")
    import tensorflow_datasets as tfds
    dataset = tfds.load('cats_vs_dogs', split='train')
    for i, example in enumerate(dataset.take(10000)):
        image_tensor = example['image']
        img = Image.fromarray(image_tensor.numpy())
        filename = f"poc_inventory/NAT_{i:05d}.jpg"
        img.save(filename)
        if (i + 1) % 1000 == 0:
            print(f"Successfully processed {i + 1} images...")

print("Data extraction complete. 10,000 images are ready in ./poc_inventory")

