import torch
import os
from glob import glob
import numpy as np
import random
import json

from loader.LITS import MotomedDataset
from tqdm import tqdm
from monai import metrics
from monai.data import decollate_batch, ThreadDataLoader
from monai.visualize import plot_2d_or_3d_image
from monai.transforms.utils import get_unique_labels
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    ScaleIntensityRange,
    Resize,
    Transform,
)

class ConvertToMultiChannelBasedOnClasses(Transform):
    
    def __init__(self, n_labels):
        self.n_labels = n_labels    
    """
    Convert labels to multi channels based on classes:
    Args:
        keys (list): list of keys to be transformed.
    """

    def __call__(self, data):
        # Convert labels to multi channels
        result = []
        for label in range(self.n_labels+1)[1:]:
            result.append(data == label)
        data = torch.squeeze(torch.stack(result, axis=0).float())
        return data

def get_loader(args):

    dicts = json.load(open('utils/dicts.json', 'r'))
    data_path = dicts['dataset_processed_path'][args.dataset]
    n_labels = dicts['out_channels'][args.dataset]

    # Get list of patients to give as input to dataset
    imgs = sorted(os.listdir(os.path.join(data_path, 'images')))
    patients_list = list()
    for img in imgs:
        if img[:4] not in patients_list:
            patients_list.append(img[:4])

    # TEST ONLY, REMOVE LATER
    # patients_list = patients_list[:2]
    
    num_total_img = len(patients_list)
    # Save data to test
    print("Not keeping data for testing!")
    # num_test = int(num_total_img*0.2)
    # test_list = patients_list[:num_test]

    # _ = patients_list[num_test:]
    num_img = len(patients_list)
    num_train = int(num_img * 0.80)

    train_list = patients_list[:num_train]
    validation_list = patients_list[num_train:]

    print('Number of patients: ', num_total_img)
    print('Number of training patients: ', len(train_list))
    print('Number of validation patients: ', len(validation_list))
    #print('Number of test patients: ', len(test_list))

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
            Resize(spatial_size=(256, 256), mode=("area")),
            #Flipd(keys=['mask'], spatial_axis=1),
            ScaleIntensityRange(
            a_min=organ_intensity_range[args.dataset][0], a_max=organ_intensity_range[args.dataset][1],
            b_min=0.0, b_max=1.0, clip=True,
            ),
        ]
    )
    train_transforms_mask = Compose(
        [
            EnsureChannelFirst(),
            Resize(spatial_size=(256, 256), mode=("nearest")),
            #Flipd(keys=['mask'], spatial_axis=1),
            ConvertToMultiChannelBasedOnClasses(n_labels=n_labels),
        ]
    )

    val_transforms_img = Compose(
        [
            EnsureChannelFirst(),
            #Flipd(keys=['mask'], spatial_axis=1),
            Resize(spatial_size=(256, 256), mode=("area")),
            ScaleIntensityRange(
            a_min=organ_intensity_range[args.dataset][0], a_max=organ_intensity_range[args.dataset][1],
            b_min=0.0, b_max=1.0, clip=True,
            ),
        ]
    )
    val_transforms_mask = Compose(
        [
            EnsureChannelFirst(),
            Resize(spatial_size=(256, 256), mode=("nearest")),
            #Flipd(keys=['mask'], spatial_axis=1),
            ConvertToMultiChannelBasedOnClasses(n_labels=n_labels),
        ]
    )

    train_ds = MotomedDataset(main_dir=data_path, delta_slice=args.slice, subset=train_list, transform_img=train_transforms_img, transform_mask=train_transforms_mask)
    val_ds = MotomedDataset(main_dir=data_path, delta_slice=args.slice, subset=validation_list, transform_img=val_transforms_img, transform_mask=val_transforms_mask)

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