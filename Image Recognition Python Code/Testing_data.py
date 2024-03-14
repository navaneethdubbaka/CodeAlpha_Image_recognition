#using a static data for testing the model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import time
class_indices = {
    0: 'airplane',
    1: 'car',
    2: 'cat',
    3: 'dog',
    4: 'flower',
    5: 'fruit',
    6: 'motorbike',
    7: 'person'
}
def classify_image(image_path, model):
    img = image.load_img(image_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_indices[predicted_class_index]
    
    return predicted_class
'''image_path = "C:\\Users\\sushu\\OneDrive\\Desktop\\OTHER COURSES\\Intern\\CodeAlpha\\Machine Learning\\Image reconization\\dataset\\data\\training\\airplane\\airplane_0012.jpg"
#capture_and_save_image(image_path)
loaded_model = tf.keras.models.load_model("Navaneeth_image_recognizaton.h5")

# Recompile the loaded model
loaded_model.compile(loss='categorical_crossentropy',
                     optimizer=RMSprop(learning_rate=0.001),
                     metrics=['accuracy'])
'''
# Define the directory containing the images
folder_path = "C:\\Users\\sushu\\OneDrive\\Desktop\\OTHER COURSES\\Intern\\CodeAlpha\\Machine Learning\\Image reconization\\dataset\\test\\"

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Iterate over each image file and classify
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    predicted_class = classify_image(image_path, loaded_model)
    print("Image:", image_file, "Predicted class:", predicted_class)