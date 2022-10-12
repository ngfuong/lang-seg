#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                                    # total number of GPUs
#SBATCH --output=slurm-out/train-vitb32-coco-fold0-%j.out             # output file location
#export OMP_NUM_THREADS=32
source env/bin/activate
export WANDB_MODE=offline
export https_proxy=http://acct:pwd@10.16.29.21:8080

srun python train_lseg_zs.py \
--exp_name train_vitb32_coco_fold0 --project_name lightseg \
--backbone clip_vitb32_384 \
--dataset coco --data_path data/Dataset_HSN \
--fold 0 --nshot 0 \
--batch_size 4 --base_lr 0.0004 \
--weight_decay 1e-5 --no-scaleinv --widehead