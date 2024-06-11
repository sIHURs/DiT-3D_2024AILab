import torch
import torch.nn as nn

import modules.functional as F

__all__ = ['Voxelization']


class Voxelization(nn.Module):
    # resolution 图像分辨率
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords): 
        # features: 点云特征， coords:点云坐标
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True) # 中心处理化，
        if self.normalize:
            # 归一化处理，[0,1], 将中心移动到0.5上
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            # 将坐标缩放到 [0, 1] 范围内。
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
