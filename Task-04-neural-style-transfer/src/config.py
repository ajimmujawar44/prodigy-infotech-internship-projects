"""Central configuration for neural style transfer."""
import os

# VGG19 layers (Gatys et al.)
CONTENT_LAYERS = ["block5_conv2"]
STYLE_LAYERS = ["block1_conv1", "block2_conv1", "block3_conv1",
                "block4_conv1", "block5_conv1"]

# Optimisation hyper-parameters
MAX_DIM = 512
STEPS = 1000
STYLE_WEIGHT = 1e-2
CONTENT_WEIGHT = 1e4
TV_WEIGHT = 30
LEARNING_RATE = 0.02
BETA_1 = 0.99
EPSILON = 1e-1

# Default sample images (same as the TensorFlow tutorial)
CONTENT_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
STYLE_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg"
HUB_MODEL_URL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
