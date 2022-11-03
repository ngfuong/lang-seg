python train_lseg_zs.py \
--exp_name train-vitl16-bsz1-lr2e-4-decay1e-5-e10 --project_name lightseg_pascal \
--logpath trans/train-vitl16-bsz1-lr2e-4-decay1e-5-e10 \
--backbone clip_vitl16_384 \
--dataset pascal --data_path data/Dataset_HSN \
--fold 0 --nshot 0 \
--max_epochs 10 \
--batch_size 2 --base_lr 2e-4 --weight_decay 1e-5 \
--no-scaleinv --widehead \
--debug 