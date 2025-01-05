import os
import random
import time
import numpy as np
import torch
from collections import Counter
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad, RandomCrop
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor, ToPILImage
from dataset import cityscapes  # Make sure this imports the correct dataset
from transform import Relabel, ToLabel  # Assuming these transforms are defined elsewhere

NUM_CLASSES = 20  # Cityscapes has 20 classes

class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target


class ENetCoTransform(object):
    def __init__(self, height=512):
        self.height = height
        pass

    def __call__(self, input, target):
        # do something to both images
        input = Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        input = ToTensor()(input)

        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target


class BiSeNetCoTransform(object):
    def __init__(self, height=512, width=1024, scales=(0.75, 1, 1.25, 1.5, 1.75, 2.0), augment=True):
        self.height = height
        self.width = width
        self.scales = scales
        self.augment = augment

    def random_scaling(self, input, target):
        scale_factor = random.choice(self.scales)
        new_size = (int(input.size[1] * scale_factor), int(input.size[0] * scale_factor))  # (height, width)
        input = Resize(new_size, Image.BILINEAR)(input)
        target = Resize(new_size, Image.NEAREST)(target)
        return input, target

    def random_cropping(self, input, target):
        i, j, h, w = RandomCrop.get_params(input, output_size=(self.height, self.width))
        input = TF.crop(input, i, j, h, w)
        target = TF.crop(target, i, j, h, w)
        return input, target

    def __call__(self, input, target):
        # Apply augmentations if enabled
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                input = TF.hflip(input)
                target = TF.hflip(target)

            # Apply random scaling
            input, target = self.random_scaling(input, target)
        
        # Apply random cropping
        input, target = self.random_cropping(input, target)
        
        # Resize to the fixed resolution
        input = Resize((self.height, self.width), Image.BILINEAR)(input)
        target = Resize((self.height, self.width), Image.NEAREST)(target)
        
        input = ToTensor()(input)
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

def calculate_class_weights(class_counts, total_pixels, num_classes):
    class_weights = torch.zeros(num_classes)
    
    for class_id in range(num_classes):
        # Get the frequency of the class (use 1 if class does not exist in the dataset)
        count = class_counts.get(class_id, 0)
        
        # Calculate the weight: inverse of the class frequency (scaled by total number of pixels)
        class_weights[class_id] = total_pixels / (count + 1e-5)  # Add small epsilon to avoid division by zero

    # Normalize the weights so that the sum is 1
    class_weights = class_weights / class_weights.sum()

    return class_weights

def calculate_class_weights2(class_counts, total_pixels, num_classes):
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

import torch

def calculate_class_weights_enet(class_counts, total_pixels, num_classes, c=1.02):
    """
    Calculate class weights using the custom ENet scheme: w_class = 1 / ln(c + p_class)

    Args:
        class_counts (dict): Dictionary with class_id as keys and pixel counts as values.
        total_pixels (int): Total number of pixels in the dataset.
        num_classes (int): Total number of classes.
        c (int): Hyperparameter to control weight scaling. Default is 102.

    Returns:
        torch.Tensor: Tensor of class weights.
    """
    class_weights = torch.zeros(num_classes)
    
    for class_id in range(num_classes):
        # Get the pixel count for the class (use 0 if class does not exist)
        count = class_counts.get(class_id, 0)
        
        # Calculate the class probability
        p_class = count / total_pixels if total_pixels > 0 else 0
        
        # Calculate the weight using the custom formula
        if p_class > 0:
            class_weights[class_id] = 1 / torch.log(torch.tensor(c + p_class, dtype=torch.float))
        else:
            class_weights[class_id] = 0  # Assign zero weight if the class has no pixels

    return class_weights


if __name__ == '__main__':
    datadir = "..\\cityscapes"  # Adjust path if needed
    co_transform = BiSeNetCoTransform(height=512)
    co_transform_val = BiSeNetCoTransform(height=512)
    
    dataset_train = cityscapes(datadir, co_transform, 'train')
    dataset_val = cityscapes(datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, num_workers=4, batch_size=16, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=16, shuffle=False)

    class_counts, total_pixels = calculate_class_histogram(loader)
    print(f"Class Counts: {class_counts}, Total Pixels: {total_pixels}")

    class_weights = calculate_class_weights(class_counts, total_pixels, NUM_CLASSES)
    print(f"Class Weights: {class_weights}")
