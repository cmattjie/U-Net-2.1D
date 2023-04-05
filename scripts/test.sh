export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 train_unet21d.py \
--batch_size 12 \
--early_stop 25 \
--dataset LITSkaggle \
--gpu 0 \
--load_dir None \
--name test_LITSkaggle \
--slice 1 \
--epochs 170 \
--lr 1e-4 \
--loss dice