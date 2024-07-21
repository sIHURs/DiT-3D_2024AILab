import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import argparse
from torch.distributions import Normal

# from utils.file_utils import *
# from utils.visualize import *
# import torch.distributed as dist
from datasets.shapenet_data_pc import ShapeNet15kPointClouds

# from copy import deepcopy
# from collections import OrderedDict

from models.dit3d import DiT3D_models
from models.dit3d_window_attn import DiT3D_models_WindAttn

# from tensorboardX import SummaryWriter


# first to see the trainable parameter of dit model here we use Dit S4
# model = DiT3D_models['DiT-S/4']()
# model_keys = model.state_dict().keys()
# print("model weight keys")
# for key in model_keys:
# 	print(key)
	
'''
model weight keys
pos_embed
x_embedder.proj.weight
x_embedder.proj.bias
t_embedder.mlp.0.weight
t_embedder.mlp.0.bias
t_embedder.mlp.2.weight
t_embedder.mlp.2.bias
y_embedder.embedding_table.weight
blocks.0.attn.qkv.weight
blocks.0.attn.qkv.bias
blocks.0.attn.proj.weight
blocks.0.attn.proj.bias
blocks.0.mlp.fc1.weight
blocks.0.mlp.fc1.bias
blocks.0.mlp.fc2.weight
blocks.0.mlp.fc2.bias
blocks.0.adaLN_modulation.1.weight
blocks.0.adaLN_modulation.1.bias
blocks.1.attn.qkv.weight
blocks.1.attn.qkv.bias
blocks.1.attn.proj.weight
blocks.1.attn.proj.bias
blocks.1.mlp.fc1.weight
blocks.1.mlp.fc1.bias
blocks.1.mlp.fc2.weight
blocks.1.mlp.fc2.bias
blocks.1.adaLN_modulation.1.weight
blocks.1.adaLN_modulation.1.bias
blocks.2.attn.qkv.weight
blocks.2.attn.qkv.bias
blocks.2.attn.proj.weight
blocks.2.attn.proj.bias
blocks.2.mlp.fc1.weight
blocks.2.mlp.fc1.bias
blocks.2.mlp.fc2.weight
blocks.2.mlp.fc2.bias
blocks.2.adaLN_modulation.1.weight
blocks.2.adaLN_modulation.1.bias
blocks.3.attn.qkv.weight
blocks.3.attn.qkv.bias
blocks.3.attn.proj.weight
blocks.3.attn.proj.bias
blocks.3.mlp.fc1.weight
blocks.3.mlp.fc1.bias
blocks.3.mlp.fc2.weight
blocks.3.mlp.fc2.bias
blocks.3.adaLN_modulation.1.weight
blocks.3.adaLN_modulation.1.bias
blocks.4.attn.qkv.weight
blocks.4.attn.qkv.bias
blocks.4.attn.proj.weight
blocks.4.attn.proj.bias
blocks.4.mlp.fc1.weight
blocks.4.mlp.fc1.bias
blocks.4.mlp.fc2.weight
blocks.4.mlp.fc2.bias
blocks.4.adaLN_modulation.1.weight
blocks.4.adaLN_modulation.1.bias
blocks.5.attn.qkv.weight
blocks.5.attn.qkv.bias
blocks.5.attn.proj.weight
blocks.5.attn.proj.bias
blocks.5.mlp.fc1.weight
blocks.5.mlp.fc1.bias
blocks.5.mlp.fc2.weight
blocks.5.mlp.fc2.bias
blocks.5.adaLN_modulation.1.weight
blocks.5.adaLN_modulation.1.bias
blocks.6.attn.qkv.weight
blocks.6.attn.qkv.bias
blocks.6.attn.proj.weight
blocks.6.attn.proj.bias
blocks.6.mlp.fc1.weight
blocks.6.mlp.fc1.bias
blocks.6.mlp.fc2.weight
blocks.6.mlp.fc2.bias
blocks.6.adaLN_modulation.1.weight
blocks.6.adaLN_modulation.1.bias
blocks.7.attn.qkv.weight
blocks.7.attn.qkv.bias
blocks.7.attn.proj.weight
blocks.7.attn.proj.bias
blocks.7.mlp.fc1.weight
blocks.7.mlp.fc1.bias
blocks.7.mlp.fc2.weight
blocks.7.mlp.fc2.bias
blocks.7.adaLN_modulation.1.weight
blocks.7.adaLN_modulation.1.bias
blocks.8.attn.qkv.weight
blocks.8.attn.qkv.bias
blocks.8.attn.proj.weight
blocks.8.attn.proj.bias
blocks.8.mlp.fc1.weight
blocks.8.mlp.fc1.bias
blocks.8.mlp.fc2.weight
blocks.8.mlp.fc2.bias
blocks.8.adaLN_modulation.1.weight
blocks.8.adaLN_modulation.1.bias
blocks.9.attn.qkv.weight
blocks.9.attn.qkv.bias
blocks.9.attn.proj.weight
blocks.9.attn.proj.bias
blocks.9.mlp.fc1.weight
blocks.9.mlp.fc1.bias
blocks.9.mlp.fc2.weight
blocks.9.mlp.fc2.bias
blocks.9.adaLN_modulation.1.weight
blocks.9.adaLN_modulation.1.bias
blocks.10.attn.qkv.weight
blocks.10.attn.qkv.bias
blocks.10.attn.proj.weight
blocks.10.attn.proj.bias
blocks.10.mlp.fc1.weight
blocks.10.mlp.fc1.bias
blocks.10.mlp.fc2.weight
blocks.10.mlp.fc2.bias
blocks.10.adaLN_modulation.1.weight
blocks.10.adaLN_modulation.1.bias
blocks.11.attn.qkv.weight
blocks.11.attn.qkv.bias
blocks.11.attn.proj.weight
blocks.11.attn.proj.bias
blocks.11.mlp.fc1.weight
blocks.11.mlp.fc1.bias
blocks.11.mlp.fc2.weight
blocks.11.mlp.fc2.bias
blocks.11.adaLN_modulation.1.weight
blocks.11.adaLN_modulation.1.bias
final_layer.linear.weight
final_layer.linear.bias
final_layer.adaLN_modulation.1.weight
final_layer.adaLN_modulation.1.bias
pretrained weight keys
'''
	
pretrained_weights_paths = "checkpoints/checkpoint.pth"
state_dict = torch.load(pretrained_weights_paths)

pretrained_keys = state_dict['model_state'].keys()
print("pretrained weight keys")
for key in pretrained_keys:
	print(key)
	
'''
pretrained weight keys
model.module.pos_embed
model.module.x_embedder.proj.weight
model.module.x_embedder.proj.bias
model.module.t_embedder.mlp.0.weight
model.module.t_embedder.mlp.0.bias
model.module.t_embedder.mlp.2.weight
model.module.t_embedder.mlp.2.bias
model.module.y_embedder.embedding_table.weight
model.module.blocks.0.attn.qkv.weight
model.module.blocks.0.attn.qkv.bias
model.module.blocks.0.attn.proj.weight
model.module.blocks.0.attn.proj.bias
model.module.blocks.0.mlp.fc1.weight
model.module.blocks.0.mlp.fc1.bias
model.module.blocks.0.mlp.fc2.weight
model.module.blocks.0.mlp.fc2.bias
model.module.blocks.0.adaLN_modulation.1.weight
model.module.blocks.0.adaLN_modulation.1.bias
model.module.blocks.1.attn.qkv.weight
model.module.blocks.1.attn.qkv.bias
model.module.blocks.1.attn.proj.weight
model.module.blocks.1.attn.proj.bias
model.module.blocks.1.mlp.fc1.weight
model.module.blocks.1.mlp.fc1.bias
model.module.blocks.1.mlp.fc2.weight
model.module.blocks.1.mlp.fc2.bias
model.module.blocks.1.adaLN_modulation.1.weight
model.module.blocks.1.adaLN_modulation.1.bias
model.module.blocks.2.attn.qkv.weight
model.module.blocks.2.attn.qkv.bias
model.module.blocks.2.attn.proj.weight
model.module.blocks.2.attn.proj.bias
model.module.blocks.2.mlp.fc1.weight
model.module.blocks.2.mlp.fc1.bias
model.module.blocks.2.mlp.fc2.weight
model.module.blocks.2.mlp.fc2.bias
model.module.blocks.2.adaLN_modulation.1.weight
model.module.blocks.2.adaLN_modulation.1.bias
model.module.blocks.3.attn.qkv.weight
model.module.blocks.3.attn.qkv.bias
model.module.blocks.3.attn.proj.weight
model.module.blocks.3.attn.proj.bias
model.module.blocks.3.mlp.fc1.weight
model.module.blocks.3.mlp.fc1.bias
model.module.blocks.3.mlp.fc2.weight
model.module.blocks.3.mlp.fc2.bias
model.module.blocks.3.adaLN_modulation.1.weight
model.module.blocks.3.adaLN_modulation.1.bias
model.module.blocks.4.attn.qkv.weight
model.module.blocks.4.attn.qkv.bias
model.module.blocks.4.attn.proj.weight
model.module.blocks.4.attn.proj.bias
model.module.blocks.4.mlp.fc1.weight
model.module.blocks.4.mlp.fc1.bias
model.module.blocks.4.mlp.fc2.weight
model.module.blocks.4.mlp.fc2.bias
model.module.blocks.4.adaLN_modulation.1.weight
model.module.blocks.4.adaLN_modulation.1.bias
model.module.blocks.5.attn.qkv.weight
model.module.blocks.5.attn.qkv.bias
model.module.blocks.5.attn.proj.weight
model.module.blocks.5.attn.proj.bias
model.module.blocks.5.mlp.fc1.weight
model.module.blocks.5.mlp.fc1.bias
model.module.blocks.5.mlp.fc2.weight
model.module.blocks.5.mlp.fc2.bias
model.module.blocks.5.adaLN_modulation.1.weight
model.module.blocks.5.adaLN_modulation.1.bias
model.module.blocks.6.attn.qkv.weight
model.module.blocks.6.attn.qkv.bias
model.module.blocks.6.attn.proj.weight
model.module.blocks.6.attn.proj.bias
model.module.blocks.6.mlp.fc1.weight
model.module.blocks.6.mlp.fc1.bias
model.module.blocks.6.mlp.fc2.weight
model.module.blocks.6.mlp.fc2.bias
model.module.blocks.6.adaLN_modulation.1.weight
model.module.blocks.6.adaLN_modulation.1.bias
model.module.blocks.7.attn.qkv.weight
model.module.blocks.7.attn.qkv.bias
model.module.blocks.7.attn.proj.weight
model.module.blocks.7.attn.proj.bias
model.module.blocks.7.mlp.fc1.weight
model.module.blocks.7.mlp.fc1.bias
model.module.blocks.7.mlp.fc2.weight
model.module.blocks.7.mlp.fc2.bias
model.module.blocks.7.adaLN_modulation.1.weight
model.module.blocks.7.adaLN_modulation.1.bias
model.module.blocks.8.attn.qkv.weight
model.module.blocks.8.attn.qkv.bias
model.module.blocks.8.attn.proj.weight
model.module.blocks.8.attn.proj.bias
model.module.blocks.8.mlp.fc1.weight
model.module.blocks.8.mlp.fc1.bias
model.module.blocks.8.mlp.fc2.weight
model.module.blocks.8.mlp.fc2.bias
model.module.blocks.8.adaLN_modulation.1.weight
model.module.blocks.8.adaLN_modulation.1.bias
model.module.blocks.9.attn.qkv.weight
model.module.blocks.9.attn.qkv.bias
model.module.blocks.9.attn.proj.weight
model.module.blocks.9.attn.proj.bias
model.module.blocks.9.mlp.fc1.weight
model.module.blocks.9.mlp.fc1.bias
model.module.blocks.9.mlp.fc2.weight
model.module.blocks.9.mlp.fc2.bias
model.module.blocks.9.adaLN_modulation.1.weight
model.module.blocks.9.adaLN_modulation.1.bias
model.module.blocks.10.attn.qkv.weight
model.module.blocks.10.attn.qkv.bias
model.module.blocks.10.attn.proj.weight
model.module.blocks.10.attn.proj.bias
model.module.blocks.10.mlp.fc1.weight
model.module.blocks.10.mlp.fc1.bias
model.module.blocks.10.mlp.fc2.weight
model.module.blocks.10.mlp.fc2.bias
model.module.blocks.10.adaLN_modulation.1.weight
model.module.blocks.10.adaLN_modulation.1.bias
model.module.blocks.11.attn.qkv.weight
model.module.blocks.11.attn.qkv.bias
model.module.blocks.11.attn.proj.weight
model.module.blocks.11.attn.proj.bias
model.module.blocks.11.mlp.fc1.weight
model.module.blocks.11.mlp.fc1.bias
model.module.blocks.11.mlp.fc2.weight
model.module.blocks.11.mlp.fc2.bias
model.module.blocks.11.adaLN_modulation.1.weight
model.module.blocks.11.adaLN_modulation.1.bias
model.module.final_layer.linear.weight
model.module.final_layer.linear.bias
model.module.final_layer.adaLN_modulation.1.weight
model.module.final_layer.adaLN_modulation.1.bias
'''