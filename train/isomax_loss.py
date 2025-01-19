"""
Paper:      Enhanced Isotropy Maximization Loss
Url:        https://arxiv.org/pdf/2105.14399
Create by:  dlmacedo
Code:       https://github.com/dlmacedo/entropic-out-of-distribution-detection/blob/master/losses/isomaxplus.py

"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class IsoMaxPlusLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""
    def __init__(self, num_classes, encoder, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.Tensor(1, num_classes, 1, 1))
        self.distance_scale = nn.Parameter(torch.Tensor(1))
        if encoder:
            self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)
        else:
            self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        features = self.output_conv(features)
        distances = torch.abs(self.distance_scale) * torch.abs(features - self.prototypes)
        logits = -distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature