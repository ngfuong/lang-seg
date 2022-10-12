#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                                    # total number of GPUs
#SBATCH --output=slurm-out/train-resnet-pascal-fold0-lr4e-3-decay1e-4-%j.out             # output file location
#export OMP_NUM_THREADS=32
source env/bin/activate
export WANDB_MODE=offline
export https_proxy=http://acct:pwd@10.16.29.21:8080

srun python train_lseg_zs.py \
--exp_name train_resnet_pascal_fold0-lr4e-3-decay1e-4 --project_name lightseg \
--backbone clip_resnet101 \
--dataset pascal --data_path data/Dataset_HSN \
--fold 0 --nshot 0 \
--batch_size 8 --base_lr 0.004 \
--weight_decay 1e-4 --no-scaleinv --widehead