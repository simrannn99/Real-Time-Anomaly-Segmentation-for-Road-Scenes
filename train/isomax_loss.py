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
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        """
        features: B x C x H x W
        Returns logits: B x num_classes x H x W
        """
        B, C, H, W = features.size()  # Batch, Channels, Height, Width
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # Reshape to (B*H*W) x C
        distances = torch.abs(self.distance_scale) * torch.cdist(
            F.normalize(features), F.normalize(self.prototypes), p=2.0
        )
        logits = -distances  # Negative distances as logits
        logits = logits.view(B, H, W, self.num_classes).permute(0, 3, 1, 2)  # Reshape to B x num_classes x H x W
        return logits / self.temperature
    
class IsoMaxPlusLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets, debug=False):
        #############################################################################
        #############################################################################
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        #############################################################################
        #############################################################################
        """
        logits: Logits of shape (B, num_classes, H, W).
        targets: Ground truth labels of shape (B, H, W).
        
        """

        B, C, H, W = logits.shape

        # Reshape logits to (B*H*W, num_classes) and flatten targets to (B*H*W)
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets = targets.view(-1)

        # Compute negative logits and apply softmax scaling
        distances = -logits
        probabilities = F.softmax(-self.entropic_scale * distances, dim=1)

        # Extract probabilities corresponding to target classes
        target_probabilities = probabilities[torch.arange(targets.numel()), targets]

        # Compute entropic loss
        loss = -torch.log(target_probabilities + 1e-12).mean()
        
        return loss