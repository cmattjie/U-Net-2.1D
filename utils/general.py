import torch
import os
from glob import glob
import numpy as np
import random

from loader.LITS import MotomedDataset
from tqdm import tqdm
from monai import metrics
from monai.data import decollate_batch, ThreadDataLoader
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    ScaleIntensityRange,
    Resize,
)

def get_loader(args):
    slice=int(args.slice)

    dataset = args.dataset
    data_paths = {
        'hmd': '/mnt/B-SSD/unet21d_slices/datasets/processed/hmd',
        'LITSkaggle': '/mnt/B-SSD/unet21d_slices/datasets/processed/LITSkaggle',
        'amos22': '/mnt/B-SSD/unet21d_slices/datasets/processed/amos22', # multiple organs, take care :D
        'MSD_Colon': '/mnt/B-SSD/unet21d_slices/datasets/processed/MSD_Colon',
        'MSD_HepaticVessel': '/mnt/B-SSD/unet21d_slices/datasets/processed/MSD_HepaticVessel',
        'MSD_Hippocampus': '/mnt/B-SSD/unet21d_slices/datasets/processed/MSD_Hippocampus',
        #'MSD_Liver': '/mnt/B-SSD/unet21d_slices/datasets/processed/MSD_Liver',
        'MSD_Lung': '/mnt/B-SSD/unet21d_slices/datasets/processed/MSD_Lung',
        'MSD_Pancreas': '/mnt/B-SSD/unet21d_slices/datasets/processed/MSD_Pancreas',
        'MSD_Spleen': '/mnt/B-SSD/unet21d_slices/datasets/processed/MSD_Spleen',
    }
    DATA_PATH = data_paths[dataset]

    # Get list of patients to give as input to dataset
    imgs = sorted(os.listdir(os.path.join(DATA_PATH, 'CT')))
    patients_list = list()
    for img in imgs:
        if img[:4] not in patients_list:
            patients_list.append(img[:4])

    # TEST ONLY, REMOVE LATER
    #patients_list = patients_list[:2]
    
    num_total_img = len(patients_list)
    # Save data to test
    num_test = int(num_total_img*0.2)
    test_list = patients_list[:num_test]

    _ = patients_list[num_test:]
    num_img = len(_)
    num_train = int(num_img * 0.80)

    train_list = _[:num_train]
    # print('lista treino:', len(train_list))
    # print('quant traino:',  num_train)
    validation_list = _[num_train:]

    print('Number of patients: ', num_total_img)
    print('Number of training patients: ', len(train_list))
    print('Number of validation patients: ', len(validation_list))
    print('Number of test patients: ', len(test_list))

    #TODO change min and max values according to dataset and organ
    organ_intensity_range = {
        'hmd': (-175, 250),
        'LITSkaggle': (-175, 250),
        'amos22': (-991, 362),
        'MSD_Colon': (-991, 362),
        'MSD_HepaticVessel': (-175, 250),
        'MSD_Hippocampus': (-175, 250),   #MRI
        'MSD_Lung': (-1024, 250),      
        'MSD_Pancreas': (-175, 250),
        'MSD_Spleen': (-175, 250),
    }
    
    # define transforms for image and segmentation
    train_transforms_img = Compose(
        [
            EnsureChannelFirst(),
            #Resize(spatial_size=(256, 256), mode=("area")),
            #Flipd(keys=['mask'], spatial_axis=1),
            ScaleIntensityRange(
            a_min=organ_intensity_range[dataset][0], a_max=organ_intensity_range[dataset][1],
            b_min=0.0, b_max=1.0, clip=True,
            ),
        ]
    )
    train_transforms_mask = Compose(
        [
            EnsureChannelFirst(),
            #Resize(spatial_size=(256, 256), mode=("nearest")),
            #Flipd(keys=['mask'], spatial_axis=1),
        ]
    )

    val_transforms_img = Compose(
        [
            EnsureChannelFirst(),
            #Flipd(keys=['mask'], spatial_axis=1),
            #Resize(spatial_size=(256, 256), mode=("area")),
            ScaleIntensityRange(
            a_min=organ_intensity_range[dataset][0], a_max=organ_intensity_range[dataset][1],
            b_min=0.0, b_max=1.0, clip=True,
            ),
        ]
    )
    val_transforms_mask = Compose(
        [
            EnsureChannelFirst(),
            #Resize(spatial_size=(256, 256), mode=("nearest")),
            #Flipd(keys=['mask'], spatial_axis=1),
        ]
    )

    train_ds = MotomedDataset(main_dir=DATA_PATH, delta_slice=slice, subset=train_list, transform_img=train_transforms_img, transform_mask=train_transforms_mask)
    val_ds = MotomedDataset(main_dir=DATA_PATH, delta_slice=slice, subset=validation_list, transform_img=val_transforms_img, transform_mask=val_transforms_mask)

    #plot_images(train_ds[50]['ct'], train_ds[50]['mask'], slice)

    train_loader = ThreadDataLoader(train_ds, num_workers=1, batch_size=args.batch_size, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=1, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader

def plot_images(image3d, mask, slice):
    '''
    Plot 2D image from 3D image, given a slice and a center
    image3d: exame
    slice: número de cortes para cada lado do centro
    salva a imagem
    '''
    import matplotlib.pyplot as plt

    slices=2*slice+1
    #nrows=1, ncols=slices, sharex=True,
    fig2 = plt.figure(figsize=((2+4*slices), 6))
    ax=[]
    
    for i in range(slices):
        if slice==0:
            img=image3d[i, :, :]
        else:
            img=image3d[0, i, :, :]
        ax.append(fig2.add_subplot(1, slices, i+1))
        ax[-1].set_title('Image '+str(i-slice))
        plt.imshow(img, cmap='gray')

    fig2.suptitle('Example of images')
    plt.savefig(os.path.join('/A/motomed/semantic_segmentation_unet', 'img_show.jpg'))

    maskfig = plt.figure(figsize=((6), 6)) 
    ax2=[]
    msk=mask[0, :, :]
    ax2.append(maskfig.add_subplot(1, 1, 1))
    plt.imshow(msk, cmap='gray')

    maskfig.suptitle('Example of masks')
    plt.savefig(os.path.join('/A/motomed/semantic_segmentation_unet', 'mask_show.jpg'))

def set_seeds(seed: int) -> None:
    '''
    Sets the seeds for commonly-used pseudorandom number generators (python,
    numpy, torch). Works for CPU and GPU computations. Also configures the
    cuDNN backend to be as deterministic as possible.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # for CPU and GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True