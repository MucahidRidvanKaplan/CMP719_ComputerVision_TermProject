import torch.nn as nn
import numpy as np


"""
ISNetLoss class defines the necessary loss function for the ISNet model.
"""

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, predictions, gt_masks):
        if isinstance(predictions, (list, tuple)):
            loss_total = 0
            for pred in predictions:
                loss_total += self.calculate_loss(pred, gt_masks)
            return loss_total / len(predictions)
        else:
            return self.calculate_loss(predictions, gt_masks)

    def calculate_loss(self, predictions, gt_masks):
        smooth = 1
        intersection = predictions * gt_masks
        loss = (intersection.sum() + smooth) / (predictions.sum() + gt_masks.sum() - intersection.sum() + smooth)
        loss = 1 - loss.mean()
        return loss


class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()

    def forward(self, predictions, gt_masks, labels):
        edge_gt = gt_masks.clone()

        ### img loss
        loss_img = self.softiou(predictions[0], labels)

        #λ = 10

        ### edge loss
        loss_edge = 10 * self.bce(predictions[1], edge_gt) + self.softiou(predictions[1].sigmoid(), edge_gt)

        return loss_img + loss_edge
      

