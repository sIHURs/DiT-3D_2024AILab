import numpy as np 
import torch
import os
import open3d as o3d


# points = np.load("/home/yifan/studium/dataset/ShapeNetCompletion/test/partial/03001627/3f5f14f6156261473b194e6e71487571/02.npy")
# print(points.shape)
# pcd1 = o3d.geometry.PointCloud()
# pcd1.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd1])
# '''seprate'''

# gen = torch.load("checkpoints/test_S4_chair_completion_finetune_adapter/syn/8649_samples_gen.pth")
# gt = torch.load("checkpoints/test_S4_chair_completion_finetune_adapter/syn/8649_samples_gt.pth")
# part = torch.load("checkpoints/test_S4_chair_completion_finetune_adapter/syn/8649_samples_part.pth")

# gen = gen.to('cpu').numpy()
# gt = gt.to('cpu').numpy()
# part = part.to('cpu').numpy()

# idx = 0 # 0, 12 147 112  115

# # 146 59 114 113 116 126 132
# shift_axis = 2

# points1 = gt[idx]
# # points1 = np.load("pipeline/gen_pcd_chair.npy")
# pcd1 = o3d.geometry.PointCloud()
# pcd1.points = o3d.utility.Vector3dVector(points1)
# pcd1.paint_uniform_color([0.5, 0.5, 0.5])

# points2 = part[idx]
# points2[:, shift_axis] = points2[:, shift_axis] - np.ones_like(points2[:, shift_axis])
# # points2 = np.load("pipeline/part_pcd_chair.npy")
# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(points2)
# pcd2.paint_uniform_color([1, 0, 0])


# points3 = gen[idx]
# points3[:, shift_axis] = points3[:, shift_axis] + np.ones_like(points3[:, shift_axis])
# pcd3 = o3d.geometry.PointCloud()
# pcd3.points = o3d.utility.Vector3dVector(points3)

# pcd3.paint_uniform_color([0.5, 0.5, 0.5])

# # save points cloud
# combined_points = np.vstack([points1, points2, points3])
# print(combined_points.shape)
# np.save("example0.npy", combined_points)

# # 将两个点云加入到一个列表中
# pcd_list = [pcd1, pcd2, pcd3]

# # 可视化两个点云
# o3d.visualization.draw_geometries(pcd_list)

# 在每个点上放置一个小球
def ball(points, radius):
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
        sphere.translate(point)
        mesh += sphere

    return mesh

'''togehter'''

gen = torch.load("checkpoints/test_S4_chair_completion_finetune_adapter/syn/8649_samples_gen.pth")
gt = torch.load("checkpoints/test_S4_chair_completion_finetune_adapter/syn/8649_samples_gt.pth")
part = torch.load("checkpoints/test_S4_chair_completion_finetune_adapter/syn/8649_samples_part.pth")

gen = gen.to('cpu').numpy()
gt = gt.to('cpu').numpy()
part = part.to('cpu').numpy()

idx = 132 # 0, 12 147 112  115

# 146 59 114 113 116 126 132

points1 = gt[idx]

# # 创建第一个点云
# points1 = np.load("pipeline/gen_pcd_chair.npy")
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points1)
pcd1.paint_uniform_color([0.5, 0.75, 0.7])

# # 创建第二个点云
# points2 = np.load("pipeline/part_pcd_chair.npy")

points2 = part[idx]
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points2)

pcd2.paint_uniform_color([1, 0, 0])

shift_axis = 2

points3 = gen[idx]
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

