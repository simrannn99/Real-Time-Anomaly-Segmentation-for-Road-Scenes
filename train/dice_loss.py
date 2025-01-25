"""
Paper:      Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations
Url:        https://arxiv.org/abs/1707.03237
Create by:  yassouali
Code:       https://github.com/yassouali/pytorch-segmentation/blob/master/utils/losses.py

"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from focal_loss import FocalLoss

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss
    
class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)
    
class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = CrossEntropyLoss2d(weight=weight)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss
    
class FocalLoss_DiceLoss(nn.Module):
    def __init__(self, smooth=1, gamma=2.0, alpha= [1] * 20):
        super(FocalLoss_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
    
    def forward(self, output, target):
        focal_loss = self.focal(output, target)
        dice_loss = self.dice(output, target)
        return focal_loss + dice_loss