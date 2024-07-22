from typing import Optional

import torch
import torch.nn as nn
from torchgeometry.losses import DiceLoss

class WeightedDiceCE(nn.Module):
    def __init__(self, w1: float, w2: float, weight: torch.tensor):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.weights = weight
        self.dice_loss = DiceLoss()
        self.ce = nn.CrossEntropyLoss(weight=self.weights.float().cuda())

    def forward(self, output: torch.tensor, target: torch.tensor) -> torch.tensor:
        loss = self.w1 * self.ce(output, target) + self.w2 * self.dice_loss(output, target)
        return loss


class VanillaDiceCE(nn.Module):
    def __init__(self, w1: float, w2: float, weight: torch.tensor):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.ignore_index = 5
        self.dice_loss = DiceLoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, output: torch.tensor, target: torch.tensor) -> torch.tensor:
        print(output.shape, target.shape)
        loss = self.w1 * self.ce(output, target) + self.w2 * self.dice_loss(output, target)
        return loss
