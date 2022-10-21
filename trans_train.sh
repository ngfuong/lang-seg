#!/bin/bash
python -u debug_model.py --dataset pascal --data_path data/Dataset_HSN \
--batch_size 1 --exp_name debug_model \
--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv \
--backbone clip_vitl16_384
