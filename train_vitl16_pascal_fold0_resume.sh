#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                                    # total number of GPUs
#SBATCH --output=slurm-out/pascal/train-vitl16-fold0-bsz1-lr2e-4-decay1e-5-e10-%j.out             # output file location
#SBATCH --nodelist=ithndgx003
export OMP_NUM_THREADS=32
source env/bin/activate
export https_proxy=http://acct:pwd@10.16.29.21:8080

srun python train_lseg_zs.py \
--exp_name train-vitl16-fold0-bsz1-lr2e-4-decay1e-5-e10 --project_name lightseg_pascal \
--logpath logs/train-vitl16-fold0-bsz1-lr2e-4-decay1e-5-e10 \
--backbone clip_vitl16_384 \
--dataset pascal --data_path data/Dataset_HSN \
--fold 0 --nshot 0 \
--max_epochs 10 \
--batch_size 1 --base_lr 2e-4 \
--weight_decay 1e-5 --no-scaleinv --widehead \
 