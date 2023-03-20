export CUDA_DEVICE_ORDER=PCI_BUS_ID

python train_unet21d.py \
--batch_size 4 \
--early_stop 10 \
--dataset LITSkaggle \
--gpu 1 \
--load_dir None \
--name LITSkaggle/lr1e-4/slice3 \
--slice 3 \
--epochs 100 \
--lr 1e-4