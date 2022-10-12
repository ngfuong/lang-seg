#!/bin/bash
#SBATCH --gpus=1                  # total number of GPUs
#SBATCH --output=slurm-out/%j-clipvit_fss.out
export OMP_NUM_THREADS=8
source env/bin/activate

#srun python test_lseg.py --backbone clip_vitl16_384 --eval --dataset ade20k --data-path /cm/archive/phuongln6/datasets/ \
#--weights checkpoints/demo_e200.ckpt --widehead --no-scaleinv 

srun python test_lseg_zs.py --backbone clip_vitl16_384 --module clipseg_DPT_test_v2 \
--dataset fss --widehead --no-scaleinv --arch_option 0 --ignore_index 255 --fold 0 --nshot 0 \
--weights checkpoints/lseg_ade20k_l16/version_0/checkpoints/last.ckpt
