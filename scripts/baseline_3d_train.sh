export CUDA_DEVICE_ORDER=PCI_BUS_ID

python baseline3d.py \
--argsed True \
--batch_size 1 \
--early_stop 10 \
--gpu 1 \
--load_dir './checkpoints/load/10_best.tar' \
--dataset 'LITSkaggle' \
--load_model False \
--name 'LITSkaggle/lr1e-6/baseline3d' \
--epochs 100 \
--lr 0.000001