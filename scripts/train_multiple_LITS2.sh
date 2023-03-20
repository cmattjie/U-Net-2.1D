export CUDA_DEVICE_ORDER=PCI_BUS_ID

for i in 0 1 2 3 4 5 6; do
alpha=$i
echo $alpha 

python3 train_unet21d.py \
--batch_size 40 \
--early_stop 10 \
--dataset LITSkaggle \
--gpu 1 \
--load_dir None \
--name LITSkaggle/lr1e-5/slice$alpha \
--slice $alpha \
--epochs 150 \
--lr 1e-5

done