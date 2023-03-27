export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 train_unet21d.py \
--batch_size 12 \
--early_stop 25 \
--dataset MSD_Lung \
--gpu 4 \
--load_dir None \
--name test_MSD_Lung/slice1/focal/lr1e-4 \
--slice 1 \
--epochs 170 \
--lr 1e-4 \
--loss focal