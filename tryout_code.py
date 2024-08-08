import numpy as np 
import torch
import os

from tqdm import tqdm
import open3d as o3d

import open3d as o3d

# # 读取 PLY 文件
# pcd = o3d.io.read_point_cloud("checkpoints/shortOutput/gen_pcd_chair.ply")

# # 打印点云信息
# print(pcd)

# # 可视化点云
# o3d.visualization.draw_geometries([pcd])


# 创建第一个点云
# points1 = np.load("datasets/data/ShapeNetCompletion/val/complete/03001627/1f8e18d42ddded6a4b3c42e318f3affc.npy")
# pcd1 = o3d.geometry.PointCloud()
# pcd1.points = o3d.utility.Vector3dVector(points1)

pcd1 = o3d.io.read_point_cloud("checkpoints/shortOutput/gen_pcd_chair.ply")

pcd1.paint_uniform_color([0.5, 0.5, 0.5])  # 将第一个点云设为红色

# 创建第二个点云
# points2 = np.load("datasets/data/ShapeNetCompletion/val/partial/03001627/1f8e18d42ddded6a4b3c42e318f3affc/00.npy")
# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(points2)
pcd2 = o3d.io.read_point_cloud("checkpoints/shortOutput/part_pcd_chair.ply")
pcd2.paint_uniform_color([1, 0, 0])  # 将第二个点云设为绿色

# 将两个点云加入到一个列表中
pcd_list = [pcd1, pcd2]

# 可视化两个点云
o3d.visualization.draw_geometries(pcd_list)





# num_classes = 1


# def token_drop(labels, dropout_prob, force_drop_ids=None):
#     """
#     Drops labels to enable classifier-free guidance.
#     """
#     if force_drop_ids is None:
#         drop_ids = torch.rand(labels.shape[0]) < dropout_prob
#     else:
#         drop_ids = force_drop_ids == 1
#     labels = torch.where(drop_ids, num_classes, labels)
#     return labels


