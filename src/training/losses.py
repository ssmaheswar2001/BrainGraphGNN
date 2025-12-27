"""
Custom loss functions for training
"""

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced datasets.
    Focuses training on hard-to-classify examples.
    
    Args:
        alpha (float): Weighting factor in range (0,1) to balance positive/negative examples
        gamma (float): Exponent of the modulating factor (1 - p_t)^gamma
        reduction (str): 'mean', 'sum' or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss