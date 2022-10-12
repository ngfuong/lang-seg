#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                  # total number of GPUs
#SBATCH --output=slurm-out/test-vitb32-coco-fold0-%j.out
export OMP_NUM_THREADS=32
export WANDB_MODE=offline
source env/bin/activate

srun python test_lseg_zs.py --backbone clip_vitb32_384 \
--module clipseg_DPT_test_v2 \
--dataset coco --datapath data/Dataset_HSN \
--widehead --no-scaleinv --arch_option 0 --ignore_index 255 \
--fold 0 --nshot 0 \
--weights checkpoints/train_vitb32_coco_fold0/version_0/checkpoints/last.ckpt