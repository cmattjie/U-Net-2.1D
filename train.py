import argparse
import os
from glob import glob
from tqdm import tqdm
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNET
import monai
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

from monai.data import create_test_image_3d, list_data_collate, decollate_batch, CacheDataset, ThreadDataLoader
from transforms.transforms_custom import SliceFromVolumed
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image
from monai.inferers import sliding_window_inference

from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Flipd,
)

import warnings
warnings.filterwarnings("ignore", message="Modifying image pixdim from")

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--batch_size', default=1, type=int, help='Size for each mini batch.')
    parser.add_argument('--early_stop', default=15, type=int)
    parser.add_argument('--early_stop_eps', default=5e-6, type=float)
    parser.add_argument('--gpu', default='0', help='GPU Number.')
    parser.add_argument('--model_path', default='./checkpoints/', type=str)
    parser.add_argument('--name', default='test', help='Run name on Tensorboard.')
    parser.add_argument('--slice', default='1', help='Number of extra slices on each side')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    return args 

# Hyperparameters etc.
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False


# list files (CT and PET) with labels and crete dict as an input to MONAI

def plot_images(image3d, mask, slice, center):
    '''
    Plot 2D image from 3D image, given a slice and a center
    image3d: exame
    slice: número de cortes para cada lado do centro
    center: corte central a ser plotado
    salva a imagem
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    slices=2*slice+1
    #nrows=1, ncols=slices, sharex=True,
    fig2 = plt.figure(figsize=((2+4*slices), 6))
    ax=[]
    for i in range(slices):
        img=image3d[0, center, :, :, i]
        ax.append(fig2.add_subplot(1, slices, i+1))
        ax[-1].set_title('Image '+str(i-slice))
        plt.imshow(img, cmap='gray')

    fig2.suptitle('Example of images')
    plt.savefig(os.path.join('/mnt/hdd1/motomed/semantic_segmentation_unet', 'img_show.jpg'))

    maskfig = plt.figure(figsize=((6), 6)) 
    ax2=[]
    msk=mask[0, center, :, :, 0]
    ax2.append(maskfig.add_subplot(1, 1, 1))
    plt.imshow(msk, cmap='gray')

    maskfig.suptitle('Example of masks')
    plt.savefig(os.path.join('/mnt/hdd1/motomed/semantic_segmentation_unet', 'mask_show.jpg'))

def get_loader(args):
    slice=int(args.slice)

    DATA_SUPERVISED = os.path.join('/mnt/hdd1/motomed/datasets/liver', 'supervised')
    DATA_SSL = os.path.join('/mnt/hdd1/motomed/datasets/liver', 'ssl')

    ct = sorted(glob(os.path.join(DATA_SUPERVISED, 'CT', '*.nii')))
    pt = sorted(glob(os.path.join(DATA_SUPERVISED, 'PT', '*.nii')))
    mask = sorted(glob(os.path.join(DATA_SUPERVISED, 'mask', '*.nii')))

    num_img = len(ct)
    num_train = int(num_img * 0.75)
    num_test = num_img - num_train

    train_files = [{'ct': ct_, 'pt': pt_, 'mask': mask_} for ct_, pt_, mask_ in zip(ct[:num_train], pt[:num_train], mask[:num_train])]
    val_files = [{'ct': ct_, 'pt': pt_, 'mask': mask_}  for ct_, pt_, mask_ in zip(ct[num_train:], pt[num_train:], mask[num_train:])]



    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=['ct', 'mask'], image_only=True),
            EnsureChannelFirstd(keys=['ct', 'mask']),
            Flipd(keys=['mask'], spatial_axis=1),
            ScaleIntensityRanged(
            keys=['ct'], a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
            ),
            #RandCropByPosNegLabeld(
            #    keys=['ct', 'mask'], label_key='mask', spatial_size=[96, 96, -1], pos=1, neg=1, num_samples=1
            #),
            SliceFromVolumed(keys=['ct', 'mask'], num_slices=slice),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=['ct', 'mask'], image_only=True),
            EnsureChannelFirstd(keys=['ct', 'mask']),
            Flipd(keys=['mask'], spatial_axis=1),
            ScaleIntensityRanged(
            keys=['ct'], a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
            ),
            SliceFromVolumed(keys=['ct', 'mask'], num_slices=slice),
        ]
    )


    #img = train_transforms(train_files[0]['ct'])
    #print(img[0].shape)
    # define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=0, collate_fn=list_data_collate)
    check_data = monai.utils.misc.first(check_loader)
    print(check_data['ct'].shape, check_data['mask'].shape)
    
    #plot images and mask from dataset
    plot_images(check_data['ct'], check_data['mask'], slice, 75)
   
    #exit()
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    #train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    train_loader = ThreadDataLoader(train_ds, num_workers=1, batch_size=1, shuffle=True)

    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    #val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_loader = ThreadDataLoader(val_ds, num_workers=1, batch_size=1)
    #val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    return train_loader, val_loader

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader)
    batch_size = 5

    for batch_data in loop:
        data_, target_ = batch_data["ct"].to('cpu'), batch_data["mask"].to('cpu')
        size = target_.shape[1]
        #print(size)

        for mini_batch in range(0, size, batch_size):
            #print(mini_batch)
            data = data_[0, mini_batch:mini_batch+batch_size, :, :, :].unsqueeze(0).to(device)
            target = target_[0, mini_batch:mini_batch+batch_size, :, :, :].unsqueeze(0).to(device)

            #data = data_[0, mini_batch:mini_batch+batch_size, :, :, :].unsqueeze(0).to(DEVICE)
            #target = target_[0, mini_batch:mini_batch+batch_size, :, :, :].unsqueeze(0).to(DEVICE)
            
            #print(data.shape, target.shape)
            data=torch.swapaxes(data,0,1)
            target=torch.swapaxes(target,0,1)
            data=torch.swapaxes(data,2,4)
            target=torch.swapaxes(target,2,4)

            #print(data.shape, target.shape)

            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                if len(target.size()) == 5:
                    #print("target size is 5", target.size())
                    target = torch.squeeze(target,1)
                    #print("squeeze target: ", target.size())
                loss = loss_fn(predictions, target)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
        #break


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device='cuda:0'
    slice=int(args.slice)
    model = UNET(in_channels=1, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loader(args)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        #print(len(val_loader))   
        #check_accuracy(val_loader, model, device=device)
        #exit()
        # save_predictions_as_imgs(
        #     val_loader, 
        #     model, 
        #     folder="/mnt/hdd1/motomed/semantic_segmentation_unet/saved_images/2.5D", 
        #     device=device
        # )
        # print("exiting...")
        # exit()  
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device=device)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=device)
        
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, 
            model, 
            folder=os.path.join("/mnt/hdd1/motomed/semantic_segmentation_unet/saved_images/2.5D", epoch), 
            device=device
        )
if __name__ == "__main__":
    args = get_args()
    main(args)

    