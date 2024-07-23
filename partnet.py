import numpy as np
import open3d as o3d
import copy

# load pcd
points = np.load('datasets/data/ShapeNetCore.v2.PC15k/02691156/train/1a04e3eab45ca15dd86060f189eb133.npy')
print(points.shape)
# set camera
object_center = np.mean(points, axis=0)
print("object center: ", object_center)

radius = 1  # the distance to the shape center
theta = np.pi / 4 
phi = np.pi / 4 

camera_position = object_center + radius * np.array([
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta)
])
print("camera position: ", camera_position)

vector = camera_position - object_center
print("vector: ", vector)
vector = vector / np.linalg.norm(vector)

A = vector[0]
B = vector[1]
C = vector[2]
D = -np.dot(vector, camera_position)

# # test case for plane
# x_min, x_max = -1, 1
# y_min, y_max = -1, 1
# num_points = 10 

# x = np.linspace(x_min, x_max, num_points)
# y = np.linspace(y_min, y_max, num_points)
# X, Y = np.meshgrid(x, y)
# Z = (-D - A * X - B * Y) / C

# points_plane = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

# points = np.vstack([points, points_plane])
# points = np.vstack([points, camera_position])

numerator = np.abs(A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D)
denominator = np.sqrt(A**2 + B**2 + C**2)
distances = numerator / denominator



def distance_calc(point, A, B, C, D):
    numerator = np.abs(A * point[0] + B * point[1] + C * point[2] + D)
    denominator = np.sqrt(A**2 + B**2 + C**2)
    distance = numerator / denominator

    return distance

threshold = distance_calc(object_center, A, B, C, D)

view_points = []
for point in points:
    distance = distance_calc(point, A, B, C, D)
    if distance < threshold:
        view_points.append(point)

view_points.append(camera_position)
view_points = np.array(view_points)
print(view_points.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(view_points)
pcd.estimate_normals()
o3d.visualization.draw_geometries([pcd])
