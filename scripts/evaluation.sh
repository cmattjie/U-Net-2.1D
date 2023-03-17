export CUDA_DEVICE_ORDER=PCI_BUS_ID

python evaluation.py \
--argsed True \
--batch_size 1 \
--gpu 1 \
--load_dir './checkpoints/LITSkaggle/lr1e-5/slice1/my_checkpoint_test.pth.tar' \
--dataset 'amos22' \
--slice 1 \
--name 'LITSkaggle-to-amos22/lr1e-5/slice1' \