import numpy as np 
import torch

from metrics.evaluation_metrics import compute_all_metrics, EMD_CD

# validation of each checkpoints pth, calculate the std from CD EMD f1?
# std from other four metrics

# 

# now its test cd emd f1 from testset epoch 8649
gen = torch.load("checkpoints/Different_CondtionPoints_S4_chair_sparsecompletion_finetune_adapter/syn/128_samples_gen.pth")
gt = torch.load("checkpoints/Different_CondtionPoints_S4_chair_sparsecompletion_finetune_adapter/syn/128_samples_gt.pth")
# part = torch.load("checkpoints/test_S4_chair_sparsecompletion_finetune_adapter/syn/8649_samples_part.pth")

print(gen.shape)
print(gt.shape)
# print(part.shape)

# batch_size, num_channels, num_points = 4, 5, 3
# x = torch.rand(batch_size, num_channels, num_points)  # 例如：x 的形状为 [16, 3, 1024]
# gen = torch.rand(batch_size, num_channels, num_points)
results = EMD_CD(gen.to('cuda'), gt.to('cuda'), 150, reduced=True) # True for output mean
print(results)


# gen = gen.to('cpu').numpy()
# gt = gt.to('cpu').numpy()
# part = part.to('cpu').numpy()

# idx = 132 # 0, 12 147 112  115

# # 146 59 114 113 116 126 132

# points1 = gt[idx]

# # # 创建第一个点云
# # points1 = np.load("pipeline/gen_pcd_chair.npy")
# pcd1 = o3d.geometry.PointCloud()
# pcd1.points = o3d.utility.Vector3dVector(points1)
# pcd1.paint_uniform_color([0.5, 0.75, 0.7])

# # # 创建第二个点云
# # points2 = np.load("pipeline/part_pcd_chair.npy")

# points2 = part[idx]
# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(points2)

# pcd2.paint_uniform_color([1, 0, 0])

# shift_axis = 2

# points3 = gen[idx]
# points3[:, shift_axis] = points3[:, shift_axis] + np.ones_like(points3[:, shift_axis])
# pcd3 = o3d.geometry.PointCloud()
# pcd3.points = o3d.utility.Vector3dVector(points3)

# pcd3.paint_uniform_color([0.5, 0.85, 0.5])

# points4 = points2
# points4[:, shift_axis] = points4[:, shift_axis] + np.ones_like(points4[:, shift_axis])
# pcd4 = o3d.geometry.PointCloud()
# pcd4.points = o3d.utility.Vector3dVector(points4)
# pcd4.paint_uniform_color([1, 0, 0])

# # 将两个点云加入到一个列表中
# pcd_list = [pcd1, pcd2, pcd3, pcd4]

# # 可视化两个点云
# o3d.visualization.draw_geometries(pcd_list)
