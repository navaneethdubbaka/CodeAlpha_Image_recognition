#Creating the Model Using Custom Data Set.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import time

# Training data generator
train_datagen = ImageDataGenerator(rescale=1/255)

# Testing data generator
test_datagen = ImageDataGenerator(rescale=1/255)

# Define paths to training and testing datasets
train_data_path = 'C:\\Users\\sushu\\OneDrive\\Desktop\\OTHER COURSES\\Intern\\CodeAlpha\\Machine Learning\\Image reconization\\dataset\\data\\training\\'

test_data_path = 'C:\\Users\\sushu\\OneDrive\\Desktop\\OTHER COURSES\\Intern\\CodeAlpha\\Machine Learning\\Image reconization\\dataset\\data\\testing\\'

# Load training and testing datasets
train_dataset = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

test_dataset = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

# Print class indices for training and testing datasets
train_class_indices = train_dataset.class_indices
test_class_indices = test_dataset.class_indices
print("Training class indices:", train_class_indices)
print("Testing class indices:", test_class_indices)

# Define the CNN model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(200, 200, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# Train the model
model_fit = model.fit(
    train_dataset,
    steps_per_epoch=len(train_dataset),
    epochs=10,
    validation_data=test_dataset,
    validation_steps=len(test_dataset)
)

# Save the trained model
model.save("Navaneeth_image_recognizaton.h5")