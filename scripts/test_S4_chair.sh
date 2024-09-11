python test.py --dataroot /home/yifan/studium/3D_Completion/DiT-3D_2024AILab/datasets/data/ShapeNetCore.v2.PC15k/ \
    --category chair --num_classes 1 \
    --bs 16 \
    --model_type 'DiT-S/4' \
    --voxel_size 32 \
    --model /home/yifan/studium/3D_Completion/DiT-3D_2024AILab/checkpoints/checkpoint.pth \
    --gpu 0 \
    --condition_npoints 128 \
