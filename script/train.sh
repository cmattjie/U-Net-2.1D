python train.py \
--argsed True \
--batch_size 4 \
--gpu 1 \
--load_dir './checkpoints/load/10_best.tar' \
--dataset 'kaggle' \
--slice 0 \
--load_model False \
--name 'lr1e-5_kaggle-slice0' \
--save_dir './checkpoints/lr1e-5_kaggle' \
--epochs 10 \
--lr 0.00001
