#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 accelerate launch --main_process_port 29509 inference.py --batch_size 1 \
                       --test_path /mnt/disk1/linda/clean_mesh_with_margin \
                       --model_path /mnt/disk1/linda/DCrownFormer/checkpoints/CrownDeformer_curvature_6400_final_all_teeth_step_400000.pth \
                    
                   