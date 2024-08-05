import open3d as o3d
import numpy as np
import click
        
                                                                                                                                     
# @click.command()
# @click.option('--path', '-p', type=str, help='path to pcd')
# @click.option('--radius', '-r', type=float, default=0, help='range to filter pcd')
# def main(path, radius):
#     # pcd = o3d.io.read_point_cloud(path)
#     points = np.load(path, allow_pickle=True)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     o3d.visualization.draw_geometries([pcd])
                                                                                                                                     
# if __name__ == '__main__':
#     main()

# 读取PLY文件
pcd = o3d.io.read_point_cloud("checkpoints/pipeline_car/gen_pcd_car.ply")

# 可视化点云
o3d.visualization.draw_geometries([pcd])