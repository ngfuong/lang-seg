#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                                    # total number of GPUs
#SBATCH --output=slurm-out/train-vit-coco-lr1e4-fold0-%j.out             # output file location
#export OMP_NUM_THREADS=32
source env/bin/activate
export WANDB_MODE=offline
export https_proxy=http://acct:pwd@10.16.29.21:8080

srun python train_lseg_zs.py \
--exp_name train_vit_coco_bsz4_lr1e4_fold0 --project_name lightseg \
--backbone clip_vitl16_384 \
--dataset coco --data_path data/Dataset_HSN \
--fold 0 --nshot 0 \
--batch_size 4 --base_lr 0.0001 --max_epochs 200 \
--weight_decay 1e-4 --no-scaleinv --widehead