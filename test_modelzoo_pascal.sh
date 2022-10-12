#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                  # total number of GPUs
#SBATCH --output=slurm-out/test-modelzoo-pascal-resnet-%j.out
export OMP_NUM_THREADS=32
export WANDB_MODE=offline
source env/bin/activate

srun python test_lseg_zs.py --backbone clip_resnet101 \
--module clipseg_DPT_test_v2 \
--dataset pascal --datapath data/Dataset_HSN \
--widehead --no-scaleinv \
--fold 0 --nshot 0 \
--weights checkpoints/pascal_fold0.ckpt
