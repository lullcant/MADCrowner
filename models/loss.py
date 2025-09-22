from torch_geometric.nn import knn
import torch
from torch import nn
import torch.nn.functional as F
# from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
# chamfer_dist = chamfer_3DDist()

def  curvature_penalty_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    gt_curvatures: torch.Tensor,
    gt_margins: torch.Tensor,
    weight_lambda
) -> torch.Tensor:
    """
    实现 L_Refine2 损失。

    参数:
        pred_points (torch.Tensor): 预测点云 (B, N1, 3)
        gt_points (torch.Tensor): GT点云 (B, N2, 3)
        gt_curvatures (torch.Tensor): GT点云曲率 (B, N2)
        gt_margins (torch.Tensor): GT点云是否为margin (B, N2)，取值0或1
    返回:
        torch.Tensor: 平均的 refine2 损失。
    """
    assert pred_points.dim() == 3
    assert gt_points.dim() == 3
    assert gt_curvatures.dim() == 2
    assert gt_margins.dim() == 2
    assert gt_points.size(0) == gt_curvatures.size(0) == gt_margins.size(0)
    assert pred_points.size(0) == gt_points.size(0)

    B, N1, D = pred_points.shape
    _, N2, _ = gt_points.shape

    # pred -> gt
    distances_pred_to_gt = torch.cdist(pred_points, gt_points, p=2)  # (B, N1, N2)
    min_dist_pred, _ = distances_pred_to_gt.min(dim=2)  # (B, N1)

   
    gt_curvatures_clamped = torch.clamp(gt_curvatures, -5, 5)
    abs_curvatures = torch.abs(gt_curvatures_clamped)

 
    _, min_indices_pred = distances_pred_to_gt.min(dim=2)  # (B, N1)
    matched_curvatures_pred = torch.gather(abs_curvatures, 1, min_indices_pred)  # (B, N1)
    matched_margins_pred = torch.gather(gt_margins, 1, min_indices_pred)  # (B, N1)

    weights_pred = torch.exp(weight_lambda*matched_curvatures_pred) + matched_margins_pred  # (B, N1)
    loss_pred = (weights_pred * min_dist_pred).mean(dim=1)  # (B,)

    # gt -> pred
    distances_gt_to_pred = torch.cdist(gt_points, pred_points, p=2)  # (B, N2, N1)
    min_dist_gt, _ = distances_gt_to_pred.min(dim=2)  # (B, N2)

    weights_gt = torch.exp(abs_curvatures*weight_lambda) + gt_margins  # (B, N2)
    loss_gt = (weights_gt * min_dist_gt).mean(dim=1)  # (B,)

    loss = (loss_pred + loss_gt) / 2  # (B,)

    return loss.mean()


