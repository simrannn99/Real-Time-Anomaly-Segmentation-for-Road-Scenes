import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import cv2

from PIL import Image, ImageDraw, ImageFont
import os

from argparse import ArgumentParser

import torch

def generate_colormap():
    """ Generate a colormap that gradually goes from blue to white to red """

    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    # Gradually go from red (index=0) to white (index=128)
    for i in range(128):
        ratio = i / 127
        b = int(255 * ratio)
        g = int(255 * ratio)
        r = 255
        colormap[i, 0] = [b, g, r]
    # Then go from white (index=128) to blue (index=255)
    for i in range(128, 256):
        ratio = (i - 128) / 127
        b = 255
        g = int(255 * (1 - ratio))
        r = int(255 * (1 - ratio))
        colormap[i, 0] = [b, g, r]
    return colormap

def save_colored_score_image(image_path, anomaly_score, save_path, file_name):
    """
    Save the image with the anomaly score colored in a new image.
    
    image_path: path to the input image
    anomaly_score: anomaly score for each pixel
    save_path: path to save the colored image
    """
    
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (anomaly_score.shape[1], anomaly_score.shape[0]))
    
    # Normalize the anomaly score
    anomaly_score = (anomaly_score - np.min(anomaly_score)) / (np.max(anomaly_score) - np.min(anomaly_score))
    
    # Apply the colormap
    anomaly_score = cv2.applyColorMap((anomaly_score * 255).astype(np.uint8), generate_colormap())
    
    # Combine the original image and the colored anomaly score
    # combined = cv2.addWeighted(image, 0.5, anomaly_score, 0.5, 0)
    
    # Save the image
    cv2.imwrite(f"{save_path}/{file_name}.png", cv2.cvtColor(anomaly_score, cv2.COLOR_RGB2BGR))


def visualize_mask(mask):
    """
    Visualize the mask as a color image: red for obstacles, blue for background.

    Parameters:
        mask (numpy.ndarray or torch.Tensor): A single-channel ground truth mask with values 0 and 1.

    Returns:
        numpy.ndarray: A 3-channel colorized mask (HxWx3).
    """
    # Ensure the mask is a numpy array
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Initialize an RGB image with all pixels set to blue
    height, width = mask.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    color_image[:, :] = [0, 0, 255]  # Set all pixels to blue (R=0, G=0, B=255)

    # Set pixels corresponding to obstacles (mask == 1) to red
    color_image[mask == 1] = [255, 0, 0]  # R=255, G=0, B=0 (red)

    # Blue for background (mask == 0)
    color_image[mask == 0] = [0, 0, 255]  # R=0, G=0, B=255 (blue)
    # Convert the RGB image to BGR for OpenCV
    color_image = color_image[:, :, ::-1]  # Swap R and B channels (RGB to BGR)

    return color_image

def save_images(input_path, anomaly_score, ood_gts, save_dir_name):
    """
    Save images with colored score overlays, ground truth labels, and the original cropped image.

    Parameters:
        input_path (str): Path to the input image.
        anomaly_score (numpy.ndarray): Anomaly score map for the image.
        ood_gt (numpy.ndarray): Ground truth label mask for the image.
        args: Command-line arguments containing save directory and related options.
    """

    # Extract file name and directory
    file_name = osp.splitext(osp.basename(input_path))[0]
    save_dir = osp.dirname(save_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save the colored anomaly score image
    save_colored_score_image(
        input_path, anomaly_score, save_path=save_dir, file_name=f"{file_name}_score"
    )

    # Save the ground truth mask with a colored overlay
    cv2.imwrite(f"{save_dir}/{file_name}_ground_truth_colored.png", visualize_mask(ood_gts))

    # Save the original cropped image
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (anomaly_score.shape[1], anomaly_score.shape[0]))
    cv2.imwrite(f"{save_dir}/{file_name}_original.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))