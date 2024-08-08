# use adapter and freeze parameters except adaptermlp, adaLN_modulation, partial pcd embedding
# start from init adamW optimizer with lr = 1e-4
# load pretrained DiT3D chair weights, model: S4
python train_completion.py --distribution_type 'single' --gpu 0\
    --dataroot /home/yifan/studium/3D_Completion/DiT-3D_2024AILab/datasets/data/ShapeNetCompletion/ \
    --category chair \
    --experiment_name train_S4_chair_completion_finetune_adapter_02 \
    --model checkpoints/train_S4_chair_completion_finetune_adapter_02/epoch_8049.pth \
    --freeze \
    --model_type 'DiT-S/4' \
    --window_size 4 --window_block_indexes '0,3,6,9' \
    --bs 16 \
    --voxel_size 32 \
    --lr 1e-4 \
    --use_tb \
    --niter 8201 \
    --saveIter 50\
    --diagIter 50 \
    --vizIter 50
