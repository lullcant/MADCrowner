import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points as furthest_point_sample
from models.SAP.src.dpsr import DPSR
from models.SAP.src.model import Encode2Points


def gather_points(points, idx):
    batch_size, _, _ = points.size()
    _, num_points = idx.size()
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_points)
    gathered_points = points[batch_indices, :, idx].transpose(1, 2)
    return gathered_points


class cross_transformer(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.activation1 = torch.nn.GELU()
        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def forward(self, src1, src2, if_act: bool = False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)
        b, c, _ = src1.shape
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)
        src1 = self.norm13(src1)
        src2 = self.norm13(src2)
        src12 = self.multihead_attn1(query=src1, key=src2, value=src2)[0]
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = src1.permute(1, 2, 0)
        return src1


class PCT_refine_mod(nn.Module):
    def __init__(self, channel: int = 128, ratio: int = 1, final: bool = False, feature_level: int = 1):
        super().__init__()
        self.ratio = ratio
        self.final = final
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(feature_level * 64, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)
        self.sa1 = cross_transformer(channel, 512)
        self.sa2 = cross_transformer(512, 512)
        self.sa3 = cross_transformer(512, channel * ratio)
        self.relu = nn.GELU()
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.channel = channel
        self.conv_delta = nn.Conv1d(channel * 2, channel * 1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel * ratio, channel * ratio, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        if final:
            self.conv_normal = nn.Conv1d(channel, 3, kernel_size=1)

    def forward(self, x, coarse, feat_g):
        batch_size, _, N = coarse.size()
        y = self.conv_x1(self.relu(self.conv_x(coarse)))
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))
        y1 = self.sa1(y, feat_g)
        y2 = self.sa2(y1, y1)
        y3 = self.sa3(y2, y2)
        y3 = self.conv_ps(y3).reshape(batch_size, -1, N * self.ratio)
        y_up = y.repeat(1, 1, self.ratio)
        y_cat = torch.cat([y3, y_up], dim=1)
        y4 = self.conv_delta(y_cat)
        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1, 1, self.ratio)
        normals = self.conv_normal(y4) if self.final else None
        return x, y3, normals


class PCT_encoder(nn.Module):
    def __init__(self, channel: int = 64):
        super().__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(4, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)
        self.sa1 = cross_transformer(channel, channel)
        self.sa1_1 = cross_transformer(channel * 2, channel * 2)
        self.sa2 = cross_transformer((channel) * 2, channel * 2)
        self.sa2_1 = cross_transformer((channel) * 4, channel * 4)
        self.sa3 = cross_transformer((channel) * 4, channel * 4)
        self.sa3_1 = cross_transformer((channel) * 8, channel * 8)
        self.relu = nn.GELU()
        self.template_mlp = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            self.relu,
            nn.Conv1d(64, channel * 8, kernel_size=1),
        )
        self.sa0_d = cross_transformer(channel * 8, channel * 8)
        self.sa1_d = cross_transformer(channel * 8, channel * 8)
        self.sa2_d = cross_transformer(channel * 8, channel * 8)
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel * 8, 64, kernel_size=1)

    def forward(self, points, template):
        batch_size, _, N = points.size()
        x = self.relu(self.conv1(points))
        x0 = self.conv2(x)
        idx_0 = furthest_point_sample(points[:, :3, :].transpose(1, 2).contiguous(), K=N // 4)[1]
        x_g0 = gather_points(x0, idx_0)
        points = gather_points(points, idx_0)
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        x1 = self.sa1_1(x1, x1).contiguous()
        idx_1 = furthest_point_sample(points[:, :3, :].transpose(1, 2).contiguous(), K=N // 8)[1]
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()
        x2 = torch.cat([x_g1, x2], dim=1)
        x2 = self.sa2_1(x2, x2).contiguous()
        idx_2 = furthest_point_sample(points[:, :3, :].transpose(1, 2).contiguous(), K=N // 16)[1]
        x_g2 = gather_points(x2, idx_2)
        points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()
        x3 = torch.cat([x_g2, x3], dim=1)
        x3 = self.sa3_1(x3, x3).contiguous()
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        template_embed = self.template_mlp(template)
        x0_d = self.sa0_d(template_embed, x_g)
        x1_d = self.sa1_d(x0_d, x0_d)
        x2_d = self.sa2_d(x1_d, x1_d)
        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))
        return x0, x1, x_g, fine



class CrownDeformer(nn.Module):
    def __init__(self, steps=(2, 2)):
        super().__init__()
        step1, step2 = steps
        self.encoder = PCT_encoder()
        self.refine = PCT_refine_mod(ratio=step1, feature_level=2)
        self.refine1 = PCT_refine_mod(ratio=step2, final=False, feature_level=1)
        self.dpsr = DPSR(res=(128, 128, 128), sig=0)
        self.model_sap = Encode2Points(
            "/mnt/disk1/linda/DCrownFormer/models/SAP/configs/learning_based/noise_small/ours.yaml"
        )

    def forward(self, x, template=None, is_training=True):
        x = x.transpose(1, 2)
        template = template.transpose(1, 2)
        x0, x1, feat_g, seed = self.encoder(x, template)
        fine, feat_fine, _ = self.refine(None, seed, x1)
        fine1, feat_fine1, _ = self.refine1(feat_fine, fine, x0)
        seed = seed.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()
        fine_output = (fine1 + 1) / 2
        fine_output, normals = self.model_sap(fine_output)
        pred_psr = self.dpsr(fine_output, normals)
        return pred_psr, seed, fine, fine1


