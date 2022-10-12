#!/bin/bash
#SBATCH --account=phuongln6
#SBATCH --gpus=1                                            # total number of GPUs
#SBATCH --output=slurm-out/test-zs-pascal-%j.out             # output file location
#export OMP_NUM_THREADS=32
source env/bin/activate
export https_proxy=http://acct:pwd@10.16.29.21:8080

# srun python -u train_lseg_zs.py --dataset pascal --data_path data/Dataset_HSN --batch_size 4 --exp_name lseg_pascal_l16 \
# --base_lr 0.016 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384

# export CUDA_VISIBLE_DEVICES=0; python test_lseg.py --backbone clip_vitl16_384 --eval --dataset ade20k --data-path ../datasets/ \
# --weights checkpoints/lseg_ade20k_l16.ckpt --widehead --no-scaleinv 

for fold in 0 1 2 3; do

srun python -u test_lseg_zs.py --backbone clip_vitl16_384 --module clipseg_DPT_test_v2 \
--dataset pascal --datapath data/Dataset_HSN \
--widehead --no-scaleinv --arch_option 0 --ignore_index 255 --fold ${fold} --nshot 0 \
--weights checkpoints/lseg_pascal_l16/version_0/checkpoints/last.ckpt

done