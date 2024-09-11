python pipeline/pipeline.py --dataroot datasets/data/ShapeNetCompletion/ \
    --category chair --num_classes 1 \
    --bs 16 \
    --model_type 'DiT-S/4' \
    --voxel_size 32 \
    --model checkpoints/epoch_8649.pth \
    --gpu 0 \
    --manualSeed 66