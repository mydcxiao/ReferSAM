def loss_masks(self, 
               src_masks, 
               target_masks, 
               num_masks):
    """Compute the losses related to the masks: the focal loss and the dice loss.
        src_masks: b, 3, h, w
        targets masks: b, h, w
        num_masks: b
    """
    # upsample predictions to the target size
    num_multimask = src_masks.size(1)
    target_masks = target_masks.flatten(1)
    loss_focal = 0
    loss_dice = 0
    for i in range(num_multimask):
        src_mask = src_masks[:, i, :, :].flatten(1)
        # target_masks = target_masks.view(src_masks.shape)
        if i == 0:
            loss_focal = sigmoid_focal_loss(src_mask, target_masks, num_masks)
            loss_dice = dice_loss(src_mask, target_masks, num_masks)
        else:
            loss_focal = min(loss_focal, sigmoid_focal_loss(src_mask, target_masks, num_masks))
            loss_dice = min(loss_dice, dice_loss(src_mask, target_masks, num_masks))
    return loss_focal, loss_dice


def dice_loss(inputs, targets, num_maskes):
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
inputs = inputs.flatten(1)
numerator = 2 * (inputs * targets).sum(1)
denominator = inputs.sum(-1) + targets.sum(-1)
loss = 1 - (numerator + 1) / (denominator + 1)
return loss.sum() / num_maskes


def sigmoid_focal_loss(inputs, targets, num_maskes, alpha: float = 0.25, gamma: float = 2):
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

    return loss.mean(1).sum() / num_maskes