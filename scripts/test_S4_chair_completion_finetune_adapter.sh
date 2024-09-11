python validation.py --dataroot datasets/data/ShapeNetCompletion/ \
    --category chair --num_classes 1 \
    --experiment_name Different_input_points_S4_chair_sparsecompletion_finetune_adapter \
    --niter 8649 \
    --bs 64 \
    --model_type 'DiT-S/4' \
    --voxel_size 32 \
    --model checkpoints/epoch_8649.pth \
    --gpu 0 \
    --condition_npoints 1024

python validation.py --dataroot datasets/data/ShapeNetCompletion/ \
    --category chair --num_classes 1 \
    --experiment_name Different_input_points_S4_chair_sparsecompletion_finetune_adapter \
    --niter 8649 \
    --bs 64 \
    --model_type 'DiT-S/4' \
    --voxel_size 32 \
    --model checkpoints/epoch_8649.pth \
    --gpu 0 \
    --condition_npoints 1280

python validation.py --dataroot datasets/data/ShapeNetCompletion/ \
    --category chair --num_classes 1 \
    --experiment_name Different_input_points_S4_chair_sparsecompletion_finetune_adapter \
    --niter 8649 \
    --bs 64 \
    --model_type 'DiT-S/4' \
    --voxel_size 32 \
    --model checkpoints/epoch_8649.pth \
    --gpu 0 \
    --condition_npoints 1536

python validation.py --dataroot datasets/data/ShapeNetCompletion/ \
    --category chair --num_classes 1 \
    --experiment_name Different_input_points_S4_chair_sparsecompletion_finetune_adapter \
    --niter 8649 \
    --bs 64 \
    --model_type 'DiT-S/4' \
    --voxel_size 32 \
    --model checkpoints/epoch_8649.pth \
    --gpu 0 \
    --condition_npoints 1792