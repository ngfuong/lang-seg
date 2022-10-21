#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                                    # total number of GPUs
#SBATCH --output=slurm-out/coco/train-vitb32-fold0-bsz1-lr1e-4-decay1e-5-e5-%j.out             # output file location
#export OMP_NUM_THREADS=32
source env/bin/activate
export https_proxy=http://acct:pwd@10.16.29.21:8080

srun python train_lseg_zs.py \
--exp_name train_vitb32_pascal_bsz1_fold0 --project_name lightseg \
--backbone clip_vitb32_384 \
--dataset pascal --data_path data/Dataset_HSN \
--fold 0 --nshot 0 \
--batch_size 1 --base_lr 1e-4 \
--weight_decay 1e-5 --no-scaleinv --widehead 