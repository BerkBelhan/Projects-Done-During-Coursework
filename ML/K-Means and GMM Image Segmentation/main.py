import os
import matplotlib.pyplot as plt

from util import *
from kmeans import KMeans
from gmm import GMM

IMAGE_FOLDER = "images"
K_VALUES = [2, 4, 6, 8]#my chosen K values for image segmentation 
#main function for image segmentation using kmeans and gmm 
images = os.listdir(IMAGE_FOLDER)

for image_name in images:

    path = os.path.join(IMAGE_FOLDER, image_name)
    image = load_image(path)
    X, h, w = preprocess_image(image)

    # kmeans strip
    fig, axes = plt.subplots(1, len(K_VALUES) + 1, figsize=(5 * (len(K_VALUES) + 1), 5))
    fig.suptitle(f"{image_name} — KMeans", fontsize=14, fontweight="bold")

    axes[0].imshow(image)#adding original image to the strip for kmeans
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    for ax, K in zip(axes[1:], K_VALUES):
        kmeans = KMeans(K=K)
        labels, centroids = kmeans.fit(X)
        segmented = reconstruct_image(labels, centroids, h, w)

        ax.imshow(segmented)
        ax.set_title(f"K = {K}", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"outputs/{image_name}_kmeans_strip.png", dpi=150, bbox_inches="tight")
    plt.close()

    # GMM strip
    fig, axes = plt.subplots(1, len(K_VALUES) + 1, figsize=(5 * (len(K_VALUES) + 1), 5))
    fig.suptitle(f"{image_name} — GMM", fontsize=14, fontweight="bold")

    axes[0].imshow(image)#adding original image to the strip for gmm
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    for ax, K in zip(axes[1:], K_VALUES):
        gmm = GMM(K=K)
        labels, means = gmm.fit(X)
        segmented = reconstruct_image(labels, means, h, w)

        ax.imshow(segmented)
        ax.set_title(f"K = {K}", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"outputs/{image_name}_gmm_strip.png", dpi=150, bbox_inches="tight")
    plt.close()