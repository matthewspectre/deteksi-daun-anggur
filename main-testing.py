import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
from google.colab import files
import matplotlib.pyplot as plt
from IPython.display import display, Image

def upload_and_save_image():
    """Upload image from local computer and save it"""
    print("Please upload your grape leaf image...")
    uploaded = files.upload()

    if not uploaded:
        print("No file was uploaded!")
        return None

    filename = list(uploaded.keys())[0]
    print(f"Successfully uploaded: {filename}")
    return filename

def show_image(img_path, title="Uploaded Image"):
    """Display an image using matplotlib"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def extract_dataset(zip_path, extract_path):
    """Extract the dataset from zip file"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def load_image(img_path, target_size=(224, 224)):
    """Load and preprocess an image"""
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def extract_features(model, img_path):
    """Extract features from an image using the model"""
    img = load_image(img_path)
    features = model.predict(img, verbose=0)
    features_flatten = features.flatten()
    return features_flatten

def find_similar_images(query_img_path, dataset_path, top_k=5):
    """Find similar images in the dataset"""
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    query_features = extract_features(model, query_img_path)

    dataset_features = []
    image_paths = []

    categories = ["Black_rot", "Esca_(Black_Measles)", "Healthy",
                 "Leaf_blight_(Isariopsis_Leaf_Spot)"]

    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            for img_name in os.listdir(category_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_path, img_name)
                    try:
                        features = extract_features(model, img_path)
                        dataset_features.append(features)
                        image_paths.append((img_path, category))
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")

    dataset_features = np.array(dataset_features)

    similarities = cosine_similarity([query_features], dataset_features)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'path': image_paths[idx][0],
            'category': image_paths[idx][1],
            'similarity': similarities[idx]
        })

    return results

def show_results(query_img_path, similar_images):
    """Display the query image and similar images with their information"""
    n_similar = len(similar_images)
    plt.figure(figsize=(15, 4 * (n_similar + 1)))

    plt.subplot(n_similar + 1, 2, 1)
    query_img = cv2.imread(query_img_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    plt.imshow(query_img)
    plt.title("Your uploaded image")
    plt.axis('off')

    for i, result in enumerate(similar_images):
        plt.subplot(n_similar + 1, 2, 2*i + 3)
        img = cv2.imread(result['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Match {i+1}: {result['category']}\n"
                 f"Similarity: {result['similarity']*100:.2f}%")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    dataset_zip = "dataset.zip"
    dataset_extract_path = "dataset"
    dataset_train_path = "dataset/DATASET/train"

    if not os.path.exists(dataset_train_path):
        print("Extracting dataset...")
        extract_dataset(dataset_zip, dataset_extract_path)

    query_img_path = upload_and_save_image()
    if query_img_path is None:
        return

    print("\nYour uploaded image:")
    show_image(query_img_path)

    print("\nAnalyzing image and finding matches...")
    similar_images = find_similar_images(query_img_path, dataset_train_path)

    show_results(query_img_path, similar_images)

    print("\nDetailed Results:")
    for i, result in enumerate(similar_images, 1):
        similarity_percentage = result['similarity'] * 100
        print(f"\n{i}. Match found in category: {result['category']}")
        print(f"   Similarity: {similarity_percentage:.2f}%")

if __name__ == "__main__":
    main()