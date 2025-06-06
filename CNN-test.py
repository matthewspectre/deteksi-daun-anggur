import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import zipfile
import random


dirtrain = "dataset/DATASET/train"
dirtest = "dataset/DATASET/test"

categories = ["Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)"]


def load_data(directory):
    data = []
    counts = []
    for category in categories:
        path = os.path.join(directory, category)
        if not os.path.exists(path):
            print(f"Warning: Directory {path} not found!")
            counts.append(0)
            continue
        class_num = categories.index(category)
        count = 0
        for img_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img_name))
                img_array = cv2.resize(img_array, (256, 256))
                data.append([img_array, class_num])
                count += 1
            except Exception as e:
                pass
        counts.append(count)
    return data, counts


training_data, count_train = load_data(dirtrain)
testing_data, count_test = load_data(dirtest)

if not training_data:
    print("Training data is empty. Please check your dataset structure.")
if not testing_data:
    print("Testing data is empty. Please check your dataset structure.")

random.shuffle(training_data)
random.shuffle(testing_data)

if training_data:
    x_train = np.array([features for features, _ in training_data]).reshape(-1, 256, 256, 3)
    y_train = np.array([label for _, label in training_data])

if testing_data:
    x_test = np.array([features for features, _ in testing_data]).reshape(-1, 256, 256, 3)
    y_test = np.array([label for _, label in testing_data])

# model CNN
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(8, 8)),

    Conv2D(32, (3, 3), padding='same', activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(8, 8)),

    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(categories), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
if training_data:
    y_train_cat = to_categorical(y_train, num_classes=len(categories))
    model.fit(x_train, y_train_cat, batch_size=32, epochs=10, validation_split=0.15, shuffle=True)
if testing_data:
    y_test_cat = to_categorical(y_test, num_classes=len(categories))
    loss, acc = model.evaluate(x_test, y_test_cat, verbose=2)
    print(f'Model accuracy: {acc * 100:.2f}%')

def predict_image(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = img.reshape(-1, 256, 256, 3)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    return categories[predicted_class]

d = "dataset/single_prediction_image.jpg"
if os.path.exists(d):
    predicted_category = predict_image(model, d)
    print(f'The predicted category is: {predicted_category}')
else:
    print(f"Image {d} not found!")