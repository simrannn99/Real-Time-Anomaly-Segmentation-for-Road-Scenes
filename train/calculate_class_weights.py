import os
import random
import time
import numpy as np
import torch
from collections import Counter
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor
from dataset import cityscapes  # Make sure this imports the correct dataset
from transform import Relabel, ToLabel  # Assuming these transforms are defined elsewhere

NUM_CLASSES = 20  # Cityscapes has 20 classes

class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc = enc
        self.augment = augment
        self.height = height

    def __call__(self, input, target):
        input = Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if self.augment:
            # Random hflip
            hflip = random.random()
            if hflip < 0.5:
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            # Random translation
            transX = random.randint(-2, 2)
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX, transY, 0, 0), fill=0)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0), fill=255)  # pad label with 255
            input = input.crop((0, 0, input.size[0] - transX, input.size[1] - transY))
            target = target.crop((0, 0, target.size[0] - transX, target.size[1] - transY))

        input = ToTensor()(input)
        if self.enc:
            target = Resize(int(self.height / 8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target

def calculate_class_histogram(dataloader):
    class_counts = Counter()
    total_pixels = 0

    for images, labels in dataloader:
        # Flatten the label tensor (batch_size * height * width)
        flattened_labels = labels.view(-1)

        # Count occurrences of each class label
        class_counts.update(flattened_labels.cpu().numpy())
        total_pixels += flattened_labels.size(0)

    return class_counts, total_pixels

def calculate_class_weights2(class_counts, total_pixels, num_classes):
    class_weights = torch.zeros(num_classes)
    
    for class_id in range(num_classes):
        # Get the frequency of the class (use 1 if class does not exist in the dataset)
        count = class_counts.get(class_id, 0)
        
        # Calculate the weight: inverse of the class frequency (scaled by total number of pixels)
        class_weights[class_id] = total_pixels / (count + 1e-5)  # Add small epsilon to avoid division by zero

    # Normalize the weights so that the sum is 1
    class_weights = class_weights / class_weights.sum()

    return class_weights

def calculate_class_weights(class_counts, total_pixels, num_classes):
    class_weights = torch.zeros(num_classes)
    
    # Calculate frequency per class (number of pixels per class / total number of pixels in images where class is present)
    class_freq = {}
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count > 0:
            class_freq[class_id] = float(count) / total_pixels
        else:
            class_freq[class_id] = 0.0

    # Calculate the median frequency across all classes
    freq_values = list(class_freq.values())
    median_freq = np.median(freq_values)

    # Calculate class weights using the formula: αc = median_freq / freq(c)
    for class_id in range(num_classes):
        if class_freq[class_id] > 0:
            class_weights[class_id] = median_freq / class_freq[class_id]
        else:
            class_weights[class_id] = 0  # If the class is not present, set the weight to 0

    return class_weights

if __name__ == '__main__':
    datadir = "..\\cityscapes"  # Adjust path if needed
    co_transform = MyCoTransform(False, augment=True, height=512)
    co_transform_val = MyCoTransform(False, augment=False, height=512)
    
    dataset_train = cityscapes(datadir, co_transform, 'train')
    dataset_val = cityscapes(datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, num_workers=4, batch_size=12, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=12, shuffle=False)

    class_counts, total_pixels = calculate_class_histogram(loader)
    print(f"Class Counts: {class_counts}, Total Pixels: {total_pixels}")

    class_weights = calculate_class_weights2(class_counts, total_pixels, NUM_CLASSES)
    print(f"Class Weights: {class_weights}")
