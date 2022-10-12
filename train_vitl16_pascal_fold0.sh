#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                                    # total number of GPUs
#SBATCH --output=slurm-out/pascal/train-vitl16-fold0-bsz8-lr7e-3-decay0.9-e200-%j.out             # output file location
#export OMP_NUM_THREADS=32
source env/bin/activate
export WANDB_MODE=offline
export https_proxy=http://acct:pwd@10.16.29.21:8080

srun python train_lseg_zs.py \
--exp_name train-vitl16-fold0-bsz8-lr7e-3-decay0.9-e200 --project_name lightseg_pascal \
--backbone clip_vitl16_384 \
--dataset pascal --data_path data/Dataset_HSN \
--fold 0 --nshot 0 \
--max_epoch 200 \
--batch_size 8 --base_lr 7e-3 \
--weight_decay 0.9 --no-scaleinv --widehead 