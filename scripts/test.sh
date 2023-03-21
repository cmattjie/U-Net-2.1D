export CUDA_DEVICE_ORDER=PCI_BUS_ID

python train_unet21d.py \
--batch_size 40 \
--early_stop 10 \
--dataset LITSkaggle \
--gpu 3 \
--load_dir None \
--name test_20-03_1e-1 \
--slice 0 \
--epochs 100 \
--lr 1e-1