python test_trans_lseg_zs.py \
--backbone clip_vitl16_384 \
--module clipseg_DPT_test_v2 \
--dataset pascal --datapath data/Dataset_HSN \
--test-batch-size 1 \
--fold 0 --nshot 0 \
--weights checkpoints/train-trans-vitl16-bsz2-lr1e-4-decay1e-5-e10/version_0/checkpoints/result-epoch=1-fewshot_val_iou=41.99.ckpt