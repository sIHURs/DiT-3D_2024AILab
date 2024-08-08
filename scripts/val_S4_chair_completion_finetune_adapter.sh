python validation.py --dataroot datasets/data/ShapeNetCompletion/ \
    --category chair --num_classes 1 \
    --experiment_name validation_epoch_8049 \
    --bs 16 \
    --model_type 'DiT-S/4' \
    --voxel_size 32 \
    --model checkpoints/train_S4_chair_completion_finetune_adapter_02/epoch_8049.pth \
    --gpu 0