import glob
import os.path as osp

sources = glob.glob('*.cpp')+glob.glob('*.cu')+glob.glob(osp.join('voxelization','*.cpp'))+glob.glob(osp.join('voxelization','*.cu'))
print(sources)

include = glob.glob('*.hpp')+glob.glob('*.cuh')+glob.glob(osp.join('voxelization','*.hpp'))+glob.glob(osp.join('voxelization','*.cuh'))
print(include)

ROOT_DIR = osp.dirname(osp.abspath(__file__))
print(ROOT_DIR)

include_dirs = [osp.join(ROOT_DIR, "include")]
print(include_dirs)