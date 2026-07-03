import os
import numpy as np
from PIL import Image

def load_images(local_path, broiler_path, size=(64, 64)):

    images = [] # List that stores images
    labels = [] # List that stores labels
    SIZE = (64, 64) # Size of the final image

    class_info = [(local_path, 0), (broiler_path, 1)] # Maps each folder class to its label

    # Loops through each folder
    for folder_path, label in class_info: 
        for filename in os.listdir(folder_path): # Lists all files in the current folder
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')): # Checks whether or not the file is an image
                image_path = os.path.join(folder_path, filename) # Full path of the image file
                img = Image.open(image_path) # Open the image
                img = img.convert('RGB') # Converts to RGB
                img = img.resize(SIZE) # Resizes the image to its desired final size (64, 64)
                img_array = np.array(img) # Converts it to a numpy array
                images.append(img_array) # Adds it in the images list
                labels.append(label) # Add its label in the labels list

    images = np.array(images)
    labels = np.array(labels)

    return images, labels