export CUDA_DEVICE_ORDER=PCI_BUS_ID

python train_unet21d.py \
--batch_size 4 \
--early_stop 10 \
--dataset LITSkaggle \
--gpu 0 \
--load_model False \
--load_dir ./checkpoints/load/10_best.tar \
--name LITSkaggle/lr1e-4/slice1 \
--slice 1 \
--epochs 100 \
--lr 1e-4