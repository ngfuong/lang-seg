#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                                    # total number of GPUs
#SBATCH --output=slurm-out/coco/train-vitl16-fold0-bsz4-lr1e-4-decay1e-5-e5-%j.out             # output file location
export OMP_NUM_THREADS=32
source env/bin/activate
export https_proxy=http://acct:pwd@10.16.29.21:8080

srun python train_lseg_zs.py \
--exp_name train-vitl16-fold0-bsz4-lr1e-4-decay1e-5-10 --project_name lightseg_coco \
--logpath logs/coco/train-vitl16-fold0-bsz4-lr1e-4-decay1e-5-e5 \
--backbone clip_vitl16_384 \
--dataset coco --data_path data/Dataset_HSN \
--fold 0 --nshot 0 \
--max_epochs 5 \
--batch_size 4 --base_lr 1e-4 \
--weight_decay 1e-5 --no-scaleinv --widehead 