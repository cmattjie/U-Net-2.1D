export CUDA_DEVICE_ORDER=PCI_BUS_ID

for i in 0 1 2 3 4 5; do
alpha=$i
echo $alpha 

python3 train_unet21d.py \
--batch_size 25 \
--early_stop 15 \
--dataset MSD_HepaticVessel \
--gpu 1 \
--load_dir None \
--name MSD_HV/loss_focal/slice$alpha/lr1e-4 \
--slice $alpha \
--epochs 200 \
--lr 1e-4 \
--loss focal

done