export CUDA_DEVICE_ORDER=PCI_BUS_ID

for i in 0 1; do
alpha=$i
echo $alpha 

python3 train_unet21d.py \
--batch_size 25 \
--early_stop 20 \
--dataset amos22 \
--gpu 2 \
--load_dir None \
--name amos22/loss_dicefocal/slice$alpha/lr1e-4 \
--slice $alpha \
--epochs 200 \
--lr 1e-4 \
--loss dicefocal \


done