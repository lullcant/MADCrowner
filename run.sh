#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29502 train_crown_deformer.py  --batch_size 8 \
                        --num_steps 400001 \
                        --lr 1e-4 \
                        --weight_decay 1e-4 \
                        --accumulation_steps 1 \
                        --save_path ./CrownDeformer_curvature_6400_final_all_teeth.pth \
                        --train_path /mnt/disk1/linda/clean_mesh_with_margin \
                        --validation_interval 20000 \
                        --lambda_w 1 \
                        --sample_points 1600 \
                       