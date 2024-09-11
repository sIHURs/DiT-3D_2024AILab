import numpy as np 
import torch
import os
import open3d as o3d

'''togehter'''


points1 = np.load("pipeline/gt_pcd_chair.npy")
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points1)
pcd1.paint_uniform_color([0.5, 0.75, 0.7])

# # 创建第二个点云
points2 = np.load("pipeline/part_pcd_chair.npy")
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points2)

pcd2.paint_uniform_color([1, 0, 0])

shift_axis = 2

points3 = np.load("pipeline/gen_averagenorm_pcd_chair.npy") 
points3[:, shift_axis] = points3[:, shift_axis] + np.ones_like(points3[:, shift_axis])
pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(points3)

pcd3.paint_uniform_color([0.5, 0.85, 0.5])

points4 = points2
points4[:, shift_axis] = points4[:, shift_axis] + np.ones_like(points4[:, shift_axis])
pcd4 = o3d.geometry.PointCloud()
pcd4.points = o3d.utility.Vector3dVector(points4)
pcd4.paint_uniform_color([1, 0, 0])

# 将两个点云加入到一个列表中
pcd_list = [pcd1, pcd2, pcd3, pcd4]

# 可视化两个点云
o3d.visualization.draw_geometries(pcd_list)