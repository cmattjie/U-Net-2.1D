export CUDA_DEVICE_ORDER=PCI_BUS_ID

python train.py \
--argsed True \
--batch_size 4 \
--early_stop 15 \
--gpu 0 \
--load_dir './checkpoints/load/10_best.tar' \
--dataset 'LITSkaggle' \
--slice 0 \
--load_model False \
--name 'LITSkaggle/lr5e-5/slice0' \
--epochs 100 \
--lr 0.00005