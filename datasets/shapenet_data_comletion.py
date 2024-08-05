import os
import torch
import numpy as np
from torch.utils.data import Dataset
# from torch.utils import data
import random
import open3d as o3d
import numpy as np
# import torch.nn.functional as F

from tqdm import tqdm


# open train folder, duplicate complete pointcloud 8 times
# open train folder, concatenate partial partial pointcloud


# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class Uniform15KPC_PART(Dataset):
    def __init__(self, root_dir, subdirs, 
                 tr_sample_size=10000,te_sample_size=10000, 
                 tr_sample_part_size = 512, te_sample_part_size=512,
                 split='train', scale=1.,
                 normalize_per_shape=False, box_per_shape=False,
                 random_subsample=False,
                 normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None,
                 all_part_points_mean=None, all_part_points_std=None,
                 input_dim=3, use_mask=False):
        self.root_dir = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        self.use_mask = use_mask
        self.box_per_shape = box_per_shape
        if use_mask:
            self.mask_transform = PointCloudMasks(radius=5, elev=5, azim=90)

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        self.all_part_points = []

        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            # sub_path = os.path.join(root_dir, subd, self.split)
            sub_path_complete = os.path.join(root_dir, self.split, 'complete', subd)
            sub_path_part = os.path.join(root_dir, self.split, 'partial', subd)
            if not os.path.isdir(sub_path_complete) or not os.path.isdir(sub_path_part):
                print("Directory missing : %s" % sub_path_complete)
                print("or directory missing : %s" % sub_path_part)
                continue 

            all_mids = []

            for x in os.listdir(sub_path_complete):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(x[:-len('.npy')]) # object file name
            print("all_mids: ", len(all_mids))
                
            # get complete point cloud
            # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
            for mid in tqdm(all_mids, desc="Processing directories"):
                # add complete points cloud
                obj_fname = os.path.join(root_dir, split, 'complete', subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)
                except:
                    # print("load1 failed")
                    continue
                assert point_cloud.shape[0] == 16384

                # add partial points cloud
                part_obj_file = os.path.join(root_dir, split, 'partial', subd, mid)
                for part_obj in ['00']: # ['00', '01', '02', '03', '04', '05', '06', '07'] cuz limit of RAM
                    part_obj_fname = os.path.join(part_obj_file, part_obj + ".npy")
                    try:
                        part_point_cloud = np.load(part_obj_fname)

                        part_point_idx = np.random.choice(part_point_cloud.shape[0], 512) # keep partial pcd (512, 3)
                        part_point_cloud = part_point_cloud[part_point_idx, :]

                        # add all the pcd to list
                        self.all_points.append(point_cloud[np.newaxis, ...])
                        self.all_part_points.append(part_point_cloud[np.newaxis, ...])
                        self.cate_idx_lst.append(cate_idx)
                        self.all_cate_mids.append((subd, mid))
                    except:
                        # print("load2 failed")
                        continue
                    # break
                # break
        print("dataset complete pcd shape: ", point_cloud.shape)
        print("dataset partial pcd shape: ", part_point_cloud.shape)

                
        print("all_points: ", len(self.all_points))
        print("all_part_ points: ", len(self.all_part_points))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_part_points = [self.all_part_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3), converge to one tensor, before is a list
        self.all_part_points = np.concatenate(self.all_part_points)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std

            self.all_part_points_mean = all_part_points_mean
            self.all_part_points_std = all_part_points_std

        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            B_part, N_part = self.all_part_points.shape[:2]

            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            self.all_part_points_mean = self.all_part_points.mean(axis=1).reshape(B_part, 1, input_dim)

            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
                self.all_part_points_std = self.all_part_points.reshape(B_part, N_part, -1).std(axis=1).reshape(B_part, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
                self.all_part_points_std = self.all_part_points.reshape(B_part, -1).std(axis=1).reshape(B_part, 1, 1)
        elif self.box_per_shape:
            B, N = self.all_points.shape[:2]
            B_part, N_part = self.all_part_points.shape[:2]

            self.all_points_mean = self.all_points.min(axis=1).reshape(B, 1, input_dim)
            self.all_part_points_mean = self.all_part_points.min(axis=1).reshape(B_part, 1, input_dim)

            self.all_points_std = self.all_points.max(axis=1).reshape(B, 1, input_dim) - self.all_points.min(axis=1).reshape(B, 1, input_dim)
            self.all_part_points_std = self.all_part_points.max(axis=1).reshape(B_part, 1, input) - self.all_part_points.min(axis=1).reshape(B_part, 1, input_dim)
        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            self.all_part_points_mean = self.all_part_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
                self.all_part_points_std = self.all_part_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)
                self.all_part_points_std = self.all_part_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.all_part_points = (self.all_part_points - self.all_part_points_mean) / self.all_part_points_std

        if self.box_per_shape:
            self.all_points = self.all_points - 0.5
            self.all_part_points = self.all_part_points - 0.5

        self.train_points = self.all_points[:, :10000] # ????? why need to do this
        self.test_points = self.all_points[:, 10000:]

        # keep the same to do on the partial pcd
        # self.train_part_points = self.all_part_points[:, :384]
        # self.test_part_points = self.all_part_points[:, 384:]
        self.train_part_points = self.all_part_points
        self.test_part_points = self.all_part_points

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        self.tr_sample_part_size = min(512, tr_sample_part_size)
        self.te_sample_part_size = min(512, te_sample_part_size)

        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d"
              % (self.tr_sample_size, self.te_sample_size))
        print("Min number of parital points: (train)%d (test)%d"
              % (self.tr_sample_part_size, self.te_sample_part_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape or self.box_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            m_part = self.all_part_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            s_part = self.all_part_points_std[idx].reshape(1, -1)
            return m, m_part, s, s_part 
        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1), self.all_part_points_mean.reshape(1, -1), self.all_part_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        tr_part_out = self.train_part_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
            tr_part_idxs = np.random.choice(tr_part_out.shape[0], self.tr_sample_part_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
            tr_part_idxs = np.arange(self.tr_sample_part_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()
        tr_part_out = torch.from_numpy(tr_part_out[tr_part_idxs, :]).float()

        te_out = self.test_points[idx]
        te_part_out = self.test_part_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
            te_part_idxs = np.random.choice(te_part_out.shape[0], self.te_sample_part_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
            te_part_idxs = np.arange(self.te_sample_part_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()
        te_part_out = torch.from_numpy(te_part_out[te_part_idxs, :]).float()

        m, m_part, s, s_part = self.get_pc_stats(idx)

        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        out = {
            'idx': idx,
            'train_points': tr_out, 'train_partial_points': tr_part_out,
            'test_points': te_out, 'test_partial_points': te_part_out,
            'mean': m, "part_mean": m_part, 'std': s, 'part_std': s_part, 
            'cate_idx': cate_idx,
            'sid': sid, 'mid': mid
        }

        if self.use_mask:
            # masked = torch.from_numpy(self.mask_transform(self.all_points[idx]))
            # ss = min(masked.shape[0], self.in_tr_sample_size//2)
            # masked = masked[:ss]
            #
            # tr_mask = torch.ones_like(masked)
            # masked = torch.cat([masked, torch.zeros(self.in_tr_sample_size - ss, 3)],dim=0)#F.pad(masked, (self.in_tr_sample_size-masked.shape[0], 0), "constant", 0)
            #
            # tr_mask =  torch.cat([tr_mask, torch.zeros(self.in_tr_sample_size- ss, 3)],dim=0)#F.pad(tr_mask, (self.in_tr_sample_size-tr_mask.shape[0], 0), "constant", 0)
            # out['train_points_masked'] = masked
            # out['train_masks'] = tr_mask
            tr_mask = self.mask_transform(tr_out)
            out['train_masks'] = tr_mask

        return out


class ShapeNet15kPointCloudsPart(Uniform15KPC_PART):
    def __init__(self, root_dir="datasets/data/ShapeNetCompletion",
                 categories=['airplane'], tr_sample_size=10000, te_sample_size=2048,
                 tr_sample_part_size = 512, te_sample_part_size=512,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False, box_per_shape=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None,
                 all_part_points_mean=None, all_part_points_std=None,
                 use_mask=False):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ShapeNet15kPointCloudsPart, self).__init__(
            root_dir, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            tr_sample_part_size = tr_sample_part_size,
            te_sample_part_size = te_sample_part_size,
            split=split, scale=scale,
            normalize_per_shape=normalize_per_shape, box_per_shape=box_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            all_part_points_mean = all_part_points_mean, all_part_points_std=all_part_points_std,
            input_dim=3, use_mask=use_mask)



class PointCloudMasks(object):
    '''
    render a view then save mask
    '''
    def __init__(self, radius : float=10, elev: float =45, azim:float=315, ):

        self.radius = radius
        self.elev = elev
        self.azim = azim


    def __call__(self, points):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        camera = [self.radius * np.sin(90-self.elev) * np.cos(self.azim),
                  self.radius * np.cos(90 - self.elev),
                  self.radius * np.sin(90 - self.elev) * np.sin(self.azim),
                  ]
        # camera = [0,self.radius,0]
        _, pt_map = pcd.hidden_point_removal(camera, self.radius)

        mask = torch.zeros_like(points)
        mask[pt_map] = 1

        return mask #points[pt_map]


####################################################################################

# test case
if __name__ == '__main__':

    def get_dataset(dataroot, npoints,category):
        tr_dataset = ShapeNet15kPointCloudsPart(root_dir=dataroot,
            categories=category.split(','), split='train',
            tr_sample_size=npoints,
            te_sample_size=npoints,
            scale=1.,
            normalize_per_shape=False,
            normalize_std_per_axis=False,
            random_subsample=True)
        
        return tr_dataset

    def get_dataloader(train_dataset, test_dataset=None):

        train_sampler = None
        test_sampler = None

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,sampler=train_sampler,
                                                    shuffle=train_sampler is None, num_workers=4, drop_last=True)

        if test_dataset is not None:
            test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,sampler=test_sampler,
                                                    shuffle=False, num_workers=1, drop_last=False)
        else:
            test_dataloader = None

        return train_dataloader, test_dataloader, train_sampler, test_sampler

    dataroot = "datasets/data/ShapeNetCompletion"
    npoints = 2048
    category = "car"

    train_dataset = get_dataset(dataroot, npoints, category)
    dataloader, _, train_sampler, _ = get_dataloader(train_dataset, None)
    print("len dataloader: ", len(dataloader))

    batch = next(iter(dataloader))
    print("batch len:", len(batch))
    print("batch keys:", batch.keys())