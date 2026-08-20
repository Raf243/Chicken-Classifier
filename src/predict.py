from pathlib import Path
import numpy as np
from PIL import Image
import pickle

# Get the directory containing predict.py
CURRENT_DIR = Path(__file__).resolve().parent

# Go from src/ to the project root, then into models/
PROJECT_ROOT = CURRENT_DIR.parent
PARAMS_PATH = PROJECT_ROOT / "models" / "parameters.pkl"
STATS_PATH = PROJECT_ROOT / "models" / "train_stats.pkl"

# Use package-style imports and robust paths to models
from utils import forward_prop

def process_image(image_path, mean, std):
    """
    This function takes the path of an image as input, loads the image and applies
    the necessary preprocessing steps.
    """
    img = Image.open(image_path)
    img = img.convert('RGB') # Converts to RGB
    img = img.resize(size = (64, 64)) # Resizes the image to its desired final size (64, 64)
    img_array = np.array(img) # Converts it to a numpy array
    img = img_array.flatten()

    img_flat = img.reshape(-1, 1)
    img_norm = (img_flat-mean) / std

    return img_norm


def predict(image_path):
    """
    This function takes the path of an image as input and returns a prediction about that image.
    """
    with open(PARAMS_PATH, 'rb') as f:
        parameters = pickle.load(f)
    with open(STATS_PATH, 'rb') as f:
        train_stats = pickle.load(f)

    mean = train_stats['mean']
    std = train_stats['std']

    X = process_image(image_path, mean, std)

    AL, _ = forward_prop(X, parameters)
    probability = float(np.asarray(AL).squeeze())

    if probability > 0.5:
        label = 'Broiler'
        confidence = probability * 100
    else:
        label = 'Local'
        confidence = (1 - probability) * 100

    return {
        'Label': label,
        'Probability': round(probability, 4),
        'Confidence': round(confidence, 2)
    }

