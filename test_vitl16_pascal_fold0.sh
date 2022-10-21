#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                  # total number of GPUs
#SBATCH --output=slurm-out/pascal/test-vitl16-fold0-bsz1-lr2e-4-decay1e-5-e10-resume-ckpt1-%j.out
#SBATCH --nodelist=ithndgx003
source env/bin/activate

export OMP_NUM_THREADS=32
export https_proxy=http://acct:pwd@10.16.29.21:8080


srun python test_lseg_zs.py --backbone clip_vitl16_384 \
--module clipseg_DPT_test_v2 \
--dataset pascal --datapath data/Dataset_HSN \
--widehead --no-scaleinv \
--test-batch-size 4 \
--fold 0 --nshot 0 \
--weights checkpoints/train-vitl16-fold0-bsz1-lr2e-4-decay1e-5-e10/version_0/checkpoints/result-epoch=1-fewshot_val_iou=63.88.ckpt
