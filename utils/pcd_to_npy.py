import numpy as np 
import os

from tqdm import tqdm
import open3d as o3d

subd = '03001627'

# NOTE: [subd] here is synset id
# sub_path = os.path.join(root_dir, subd, self.split)

root_dir = "/home/yifan/studium/dataset/ShapeNetCompletion"
split = "test"
sub_path_complete = os.path.join(root_dir, split, 'complete', subd)
sub_path_part = os.path.join(root_dir, split, 'partial', subd)

if not os.path.isdir(sub_path_complete) and not os.path.isdir(sub_path_part):
    print("Directory missing : %s" % sub_path_complete)
    print("or directory missing : %s" % sub_path_part)
    raise NotImplementedError

all_mids = []

for x in os.listdir(sub_path_complete):
    if not x.endswith('.npy'):
        continue
    all_mids.append(x[:-len('.npy')]) # object file name
print("all_mids: ", len(all_mids))
    
# get complete point cloud
# NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
for mid in tqdm(all_mids, desc="processing"):
    # add complete points cloud
    obj_fname = os.path.join(root_dir, split, 'complete', subd, mid + ".npy")
    # obj_save_fname = os.path.join(root_dir, split, 'complete', subd, mid + ".npy")
    try:
        point_cloud = np.load(obj_fname)
        idxs = np.random.choice(point_cloud.shape[0], 512, replace=False)
        point_cloud = point_cloud[idxs]

    except:
        # print("load1 failed")
        continue
    assert point_cloud.shape[0] == 512

    # save complete pcds

    # add partial points cloud
    part_obj_file = os.path.join(root_dir, split, 'partial', subd, mid)

    part_obj_fname_save = os.path.join(part_obj_file, '02' + ".npy")

    np.save(part_obj_fname_save, point_cloud)

# subd = '03001627'

# # NOTE: [subd] here is synset id
# # sub_path = os.path.join(root_dir, subd, self.split)

# root_dir = "/home/yifan/studium/dataset/ShapeNetCompletion"
# split = "test"
# sub_path_complete = os.path.join(root_dir, split, 'complete', subd)
# sub_path_part = os.path.join(root_dir, split, 'partial', subd)

# if not os.path.isdir(sub_path_complete) and not os.path.isdir(sub_path_part):
#     print("Directory missing : %s" % sub_path_complete)
#     print("or directory missing : %s" % sub_path_part)
#     raise NotImplementedError

# all_mids = []

# for x in os.listdir(sub_path_complete):
#     if not x.endswith('.npy'):
#         continue
#     all_mids.append(x[:-len('.npy')]) # object file name
# print("all_mids: ", len(all_mids))
    
# # get complete point cloud
# # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
# for mid in tqdm(all_mids, desc="processing"):
#     # add complete points cloud
#     obj_fname = os.path.join(root_dir, split, 'complete', subd, mid + ".pcd")
#     obj_save_fname = os.path.join(root_dir, split, 'complete', subd, mid + ".npy")
#     try:

#         pcd_o3d = o3d.io.read_point_cloud(obj_fname)
#         point_cloud = np.asarray(pcd_o3d.points)
#         # point_cloud = np.load(obj_fname)
#     except:
#         # print("load1 failed")
#         continue
#     assert point_cloud.shape[0] == 16384

#     # save complete pcds
#     np.save(obj_save_fname, point_cloud)
#     os.remove(obj_fname)

#     # add partial points cloud
#     part_obj_file = os.path.join(root_dir, split, 'partial', subd, mid)

#     for part_obj in ['00']: # ['00', '01', '02', '03', '04', '05', '06', '07'] cuz limit of RAM
#         part_obj_fname = os.path.join(part_obj_file, part_obj + ".pcd")
#         part_obj_fname_save = os.path.join(part_obj_file, part_obj + ".npy")
#         try:
#             # part_point_cloud = np.load(part_obj_fname)
#             part_pcd_o3d = o3d.io.read_point_cloud(part_obj_fname)

#             point_cloud_part = np.asarray(part_pcd_o3d.points)
#         except:
#             # print("load2 failed")
#             continue
#         np.save(part_obj_fname_save, point_cloud_part)
#         os.remove(part_obj_fname)
