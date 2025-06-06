from google.colab import drive
drive.mount('/content/drive')
!cp "/content/drive/MyDrive/path/to/your/DATASET.zip" "/content/"


from google.colab import files
uploaded = files.upload()
import zipfile
import os

def extract_dataset(zip_path='DATASET.zip', extract_path='/content/dataset'):
    """Extract the dataset and print its structure"""
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset structure:")
    for root, dirs, files in os.walk(extract_path):
        level = root.replace(extract_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files[:3]:
            print(f"{subindent}{f}")
        if len(files) > 3:
            print(f"{subindent}...")
        print()

extract_dataset()


def count_images(dataset_path='/content/dataset/DATASET'):
    """Count images in train and test directories"""
    for split in ['train', 'test']:
        print(f"\n{split.upper()} dataset:")
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            for category in os.listdir(split_path):
                category_path = os.path.join(split_path, category)
                if os.path.isdir(category_path):
                    num_images = len([f for f in os.listdir(category_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    print(f"{category}: {num_images} images")
        else:
            print(f"Warning: {split} directory not found!")


count_images()


import matplotlib.pyplot as plt
import random
import cv2
import numpy as np

def show_sample_images(dataset_path='/content/dataset/DATASET/train', num_samples=5):
    """Display sample images from each category"""
    categories = os.listdir(dataset_path)
    plt.figure(figsize=(15, 3*len(categories)))

    for i, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            samples = random.sample(images, min(num_samples, len(images)))

            for j, img_name in enumerate(samples):
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                plt.subplot(len(categories), num_samples, i*num_samples + j + 1)
                plt.imshow(img)
                plt.axis('off')
                if j == 0:
                    plt.title(f'{category}')

    plt.tight_layout()
    plt.show()
    show_sample_images()