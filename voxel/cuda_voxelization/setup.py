import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')
# sources = glob.glob('*.cpp')+glob.glob('*.cu')+glob.glob(osp.join('voxelization','*.cpp'))+glob.glob(osp.join('voxelization','*.cu'))
# include = glob.glob('*.hpp')+glob.glob('*.cuh')+glob.glob(osp.join('voxelization','*.hpp'))+glob.glob(osp.join('voxelization','*.cuh'))

setup(
    name='project_voxelization',
    version='1.0',
    author='who',
    author_email='yifanwang812@gmail.com',
    description='nothing',
    long_description='none',
    ext_modules=[
        CUDAExtension(
            name='project_voxelization',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)