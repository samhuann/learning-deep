import numpy as np
import struct
import os

# Define the MNIST dataset directory

def load_mnist_images(filename):
    """Loads MNIST test images from the specified file."""
    
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        
        # Normalize images to range [-0.5, 0.5] (improves training stability)
        images = (images / 255) - 0.5
        return images

def load_mnist_labels(filename):
    """Loads MNIST test labels from the specified file."""

    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def load_test_data():
    """Loads MNIST test images and labels."""
    test_images = load_mnist_images("mnist_data/t10k-images.idx3-ubyte")[:1000]
    test_labels = load_mnist_labels("mnist_data/t10k-labels.idx1-ubyte")[:1000]

    return test_images, test_labels
