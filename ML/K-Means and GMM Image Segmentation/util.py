import numpy as np
from skimage import io
# Utility functions for image loading, preprocessing, and reconstruction
def load_image(path):
    image = io.imread(path)
    image = image[:, :, :3]
    return image

def preprocess_image(image):

    h, w, c = image.shape #getting height, width and channels of the image

    pixels = image.reshape(-1, 3).astype(np.float64)

    # Normalize [0,1]
    pixels /= 255.0

    return pixels, h, w

def reconstruct_image(labels, centroids, h, w):
    #reconstructing segmented image from labels and centroids 
    #beacause we need to convert the 1D labels back to the original image shape and assign the corresponding centroid color to each pixel

    segmented = centroids[labels]

    segmented = segmented.reshape(h, w, 3)

    # Convert back from [0,1] to [0,255]
    segmented = segmented * 255.0

    segmented = np.clip(segmented, 0, 255)

    return segmented.astype(np.uint8)