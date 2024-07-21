
python train.py --distribution_type 'single' \
    --dataroot /home/yifan/studium/3D_Completion/DiT-3D_2024AILab/datasets/data/ShapeNetCore.v2.PC15k/ \
    --category chair \
    --experiment_name experiments/ \
    --model_type 'DiT-S/4' \
    --window_size 4 --window_block_indexes '0,3,6,9' \
    --bs 16 \
    --voxel_size 32 \
    --lr 1e-4 \
    --use_tb
