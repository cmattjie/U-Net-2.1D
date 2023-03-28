export CUDA_DEVICE_ORDER=PCI_BUS_ID

for i in 0 1 2 3 4; do
alpha=$i
echo $alpha 

python3 train_unet21d.py \
--batch_size 25 \
--early_stop 15 \
--dataset MSD_Lung \
--gpu 2 \
--load_dir None \
--name MSD_Lung_drop0.5/loss_focal/slice$alpha/lr1e-4 \
--slice $alpha \
--epochs 200 \
--lr 1e-4 \
--loss focal \
--dropout 0.5

done