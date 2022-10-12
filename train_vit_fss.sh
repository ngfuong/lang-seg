#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                                    # total number of GPUs
#SBATCH --output=slurm-out/train_vitl16_fss-%j.out             # output file location
#export OMP_NUM_THREADS=32
source env/bin/activate
export https_proxy=http://acct:pwd@10.16.29.21:8080

srun python train_lseg_zs.py \
--exp_name train_vitl16_fss --project_name lightseg \
--backbone clip_vitl16_384 \
--dataset fss --data_path data/Dataset_HSN \
--fold 0 --nshot 0 \
--batch_size 4 --base_lr 0.016 --max_epochs 200 \
--weight_decay 1e-4 --no-scaleinv --widehead \
# --accumulate_grad_batches 2 \

