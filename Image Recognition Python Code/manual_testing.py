#giving a single image from manual testing
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    try:
        img = image.load_img(image_path, target_size=(200, 200))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_indices[predicted_class_index]

        return predicted_class
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Load the pre-trained model
loaded_model = tf.keras.models.load_model("Navaneeth_image_recognizaton.h5")

# Recompile the loaded model if needed

# Path to the specific image
image_path = r"C:\Users\sushu\OneDrive\Desktop\OTHER COURSES\Intern\CodeAlpha\Machine Learning\Image reconization\dataset\manualtest\n.jpg"


# Classify the image
predicted_class = classify_image(image_path, loaded_model)
if predicted_class is not None:
    print("Predicted class:", predicted_class)
else:
    print("Failed to classify the image.")
#using Matplot to verify the image.
image = plt.imread(image_path)
# Display the image
plt.imshow(image)
plt.axis('off') 
plt.show()
