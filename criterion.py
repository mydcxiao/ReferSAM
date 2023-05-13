import torch
from torch import nn
import torch.nn.functional as F

class Criterion(nn.Module):
    def __init__(self,
                 num_masks, # B
                 weight_focal, # weight for focal loss
                 weight_dice, # weight for dice loss
                 weight_iou, # weight for iou pred loss
                 ):
        super().__init__()
        self.num_masks = num_masks
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.weight_iou = weight_iou


    def loss_masks(self, 
                   src_masks, 
                   target_masks, 
                #    num_masks,
                   ):
        """Compute the losses related to the masks: the focal loss and the dice loss.
            src_masks: b, 3, h, w
            targets masks: b, h, w
            num_masks: b
        """
        # upsample predictions to the target size
        num_multimask = src_masks.size(1)
        target_masks = target_masks.flatten(1)
        # wf = self.weight_focal / (self.weight_focal + self.weight_dice)
        # wd = 1 - wf
        wf = self.weight_focal
        wd = self.weight_dice
        loss_masks_list = []
        for i in range(num_multimask):
            src_mask = src_masks[:, i, :, :].flatten(1)
            # target_masks = target_masks.view(src_masks.shape)
            loss_masks_list.append(wf * self.sigmoid_focal_loss(src_mask, target_masks) \
                                  + wd * self.dice_loss(src_mask, target_masks))
        loss_masks_tensor = torch.stack(loss_masks_list, dim=1)
        loss_masks_min, _ = loss_masks_tensor.min(1)
        loss_masks = loss_masks_min.sum() / self.num_masks
        return loss_masks


    def dice_loss(self,
                  inputs, 
                  targets, 
                #   num_masks,
                  ):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                        classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        # inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        # return loss.sum() / num_masks
        return loss # (B,)


    def sigmoid_focal_loss(self,
                           inputs, 
                           targets, 
                        #    num_masks, 
                           alpha: float = -1, #0.25, 
                           gamma: float = 2,
                           ):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        # return loss.mean(1).sum() / num_masks
        return loss.mean(1) # (B,)

    def loss_iou(self,
                 iou_pred, # B x C
                 target, # B x C
                 ):
        loss = nn.functional.mse_loss(iou_pred, target, reduction='none')
        return loss.mean(1).sum() / self.num_masks
    
    def forward(self, src_masks, target_masks, iou_pred, iou_gt):
        loss_masks = self.loss_masks(src_masks, target_masks)
        loss_iou = self.loss_iou(iou_pred, iou_gt)
        return loss_masks + self.weight_iou * loss_iou

# c = Criterion(8, 20.0, 1.0, 1.0)
# for name, param in c.named_parameters():
#     print(name)
# for param in list(c.parameters()):
#     print(param.size())
# print(len(list(c.parameters())))
    


