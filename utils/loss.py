import torch.nn as nn
import torch.nn.functional as F


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
    
class JointsKLLoss(nn.Module):
    """
    KL Divergence for keypoint detection proposed by
    `Regressive Domain Adaptation for Unsupervised Keypoint Detection <https://arxiv.org/abs/2103.06175>`_.
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``
    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.
    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.
    """
    def __init__(self, reduction='mean', epsilon=0.):
        super(JointsKLLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
        heatmaps_gt = target.reshape((B, K, -1))
        heatmaps_gt = heatmaps_gt + self.epsilon
        heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True)
        loss = self.criterion(heatmaps_pred, heatmaps_gt).sum(dim=-1)
        if target_weight is not None:
            loss = loss * target_weight.view((B, K))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)