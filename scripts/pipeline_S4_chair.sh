python pipeline.py --dataroot datasets/data/ShapeNetCompletion/ \
    --category chair --num_classes 1 \
    --bs 1 \
    --model_type 'DiT-S/4' \
    --voxel_size 32 \
    --model checkpoints/train_S4_chair_completion_finetune_adapter/epoch_7799.pth \
    --gpu 0