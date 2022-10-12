#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                                    # total number of GPUs
#SBATCH --output=slurm-out/train_zs_pascal-%j.out             # output file location
#export OMP_NUM_THREADS=32
source env/bin/activate
export https_proxy=http://acct:pwd@10.16.29.21:8080

srun python train_lseg_zs.py --dataset pascal --data_path data/Dataset_HSN \
--batch_size 1 --exp_name train_zs_pascal_l16_bsz1 \
--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --widehead \
--accumulate_grad_batches 2 --backbone clip_vitl16_384

