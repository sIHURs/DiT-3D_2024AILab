
python train.py --distribution_type 'single' --gpu 0\
    --dataroot /home/yifan/studium/3D_Completion/DiT-3D_2024AILab/datasets/data/ShapeNetCore.v2.PC15k/ \
    --category chair \
    --num_classes 1 \
    --experiment_name train_S4_chair_epoch_tryout/ \
    --model /home/yifan/studium/3D_Completion/DiT-3D_2024AILab/checkpoints/checkpoint.pth \
    --model_type 'DiT-S/4' \
    --window_size 4 --window_block_indexes '0,3,6,9' \
    --bs 16 \
    --voxel_size 32 \
    --lr 1e-5 \
    --use_tb \
    --niter 7651 \
    --saveIter 100 \
    --diagIter 1 \
    --vizIter 1
