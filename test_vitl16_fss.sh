#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                  # total number of GPUs
#SBATCH --output=slurm-out/test-vitl16-fss-fold0-%j.out
export OMP_NUM_THREADS=32
export WANDB_MODE=offline
source env/bin/activate

srun python test_lseg_zs.py --backbone clip_vitl16_384 \
--module clipseg_DPT_test_v2 \
--dataset fss --datapath data/Dataset_HSN \
--widehead --no-scaleinv --arch_option 0 --ignore_index 255 \
--fold 0 --nshot 0 \
--weights checkpoints/train_vitl16_fss/version_0/checkpoints/result-epoch=45-fewshot_val_iou=85.43.ckpt
