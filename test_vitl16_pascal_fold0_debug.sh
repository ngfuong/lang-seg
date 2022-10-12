#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                  # total number of GPUs
#SBATCH --output=slurm-out/test-vitl16-pascal-debug-%j.out
export OMP_NUM_THREADS=32
export WANDB_MODE=offline
export https_proxy=http://acct:pwd@10.16.29.21:8080
source env/bin/activate

srun python test_lseg_zs.py --backbone clip_vitl16_384 \
--module clipseg_DPT_test_v2 \
--dataset pascal --datapath data/Dataset_HSN \
--widehead --no-scaleinv \
--fold 0 --nshot 0 \
--weights checkpoints/train_vit_pascal_bsz4_fold0/version_0/checkpoints/last.ckpt