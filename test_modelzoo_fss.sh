#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                  # total number of GPUs
#SBATCH --output=slurm-out/test_modelzoo_fss-%j.out
export OMP_NUM_THREADS=32
export WANDB_MODE=offline
source env/bin/activate

srun python test_lseg_zs.py --backbone clip_vitl16_384 \
--module clipseg_DPT_test_v2 \
--dataset fss --datapath data/Dataset_HSN \
--widehead --no-scaleinv --arch_option 0 --ignore_index 255 \
--fold 0 --nshot 0 \
--weights checkpoints/fss_l16.ckpt
