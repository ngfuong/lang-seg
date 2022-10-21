#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                  # total number of GPUs
#SBATCH --output=slurm-out/pascal/debug-test-modelzoo-fold0-%j.out
#SBATCH --nodelist=ithndgx003
export https_proxy=http://acct:pwd@10.16.29.21:8080
source env/bin/activate

srun python test_lseg_zs_debug.py --backbone clip_resnet101 \
--module clipseg_DPT_test_v2 \
--dataset pascal --datapath data/Dataset_HSN \
--widehead --no-scaleinv \
--fold 0 --nshot 0 \
--weights checkpoints/pascal_fold0.ckpt \
--bsz 4 \
--logpath debug/test_pascal_fold0_debug