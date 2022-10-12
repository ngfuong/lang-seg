#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                                    # total number of GPUs
#SBATCH --output=slurm-out/train_vit_coco-%j.out             # output file location
#export OMP_NUM_THREADS=32
source env/bin/activate
export https_proxy=http://acct:pwd@10.16.29.21:8080

for fold in 0 1 2 3; do

srun python train_lseg_zs.py \
--dataset coco \
--data_path data/Dataset_HSN \
--backbone clip_vitl16_384 \
--fold ${fold} \
--batch_size 4 \
--base_lr 0.016 \
--weight_decay 1e-4 \
--no-scaleinv \
--widehead \
--exp_name train_vit_coco_bsz8_fold${fold} \

done