#using Web Cam to classify the images

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import time
# Define class indices
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

# Function to classify an image
def classify_image(image_path, model):
    img = image.load_img(image_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_indices[predicted_class_index]
    
    return predicted_class

# Load the pre-trained model
loaded_model = tf.keras.models.load_model("Navaneeth_image_recognizaton.h5")

# Recompile the loaded model if needed
loaded_model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                     metrics=['accuracy'])

def capture_and_save_image(save_path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam")
        return None
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame")
        cap.release()  # Release the camera capture object
        return None
    
    # Extract the directory path and image name from the provided save_path
    directory_path, image_name = os.path.split(save_path)
    
    # Ensure that the directory path exists
    os.makedirs(directory_path, exist_ok=True)
    
    # Save the captured image with the specified name
    cv2.imwrite(os.path.join(directory_path, image_name), frame)
    
    cap.release()  # Release the camera capture object
    #print("Image saved successfully at:", save_path)
    time.sleep(1)  # Add a delay of 1 second
    
    return save_path

save_path = "C:\\Users\\sushu\\OneDrive\\Desktop\\OTHER COURSES\\Intern\\CodeAlpha\\Machine Learning\\Image reconization\\dataset\\data\\captured\\my_captured_image.jpg"

image_path = capture_and_save_image(save_path)
# Classify the specified image
predicted_class = classify_image(image_path, loaded_model)
print("Image:", image_path, "Predicted class:", predicted_class)