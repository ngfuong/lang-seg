#!/bin/bash
#SBATCH --gpus=1                                    # total number of GPUs
#SBATCH --output=slurm-out/train-%j.out             # output file location
#export OMP_NUM_THREADS=32
source env/bin/activate
export https_proxy=http://acct:pwd@10.16.29.21:8080

srun python -u train_lseg.py --dataset ade20k --data_path /cm/archive/phuongln6/ --batch_size 8 --exp_name lseg_ade20k_l16 \
--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384

#python -u train_lseg.py --dataset ade20k --data_path ../datasets --batch_size 4 --exp_name lseg_ade20k_l16 \
#--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384

