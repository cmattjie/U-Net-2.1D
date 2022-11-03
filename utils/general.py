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

def valid(loader, model, loss_fn, device, writer, epoch, metric):
    post_trans= Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    model.eval()
    epoch_loss=[]
    #Arranjar um jeito de salvar apenas a melhor época baseada na "metric" (dsc_bg da última época)

    with torch.no_grad():

        loop = tqdm(loader, ncols=110)

        dice_bg = metrics.DiceMetric(include_background=True)
        dice= metrics.DiceMetric(include_background=False)
        iou_predbg= metrics.MeanIoU(include_background=True)
        iou_pred= metrics.MeanIoU(include_background=False)

        save_batch = True
        count = 0
        for batch_data in loop:
            data, target = batch_data["ct"].to(device), batch_data["mask"].to(device)

            pred_raw = model(data)
            preds=[post_trans(i) for i in decollate_batch(pred_raw)]

            loss = loss_fn(pred_raw, target)
            epoch_loss.append(loss.item())

            dice_bg(preds, target)
            dice(preds, target)
            iou_predbg(preds, target)
            iou_pred(preds, target)

            count += 1

            # save one image for visualization of the curent epoch (only once per epoch)
            if save_batch and torch.sum(target) > 10000:
                save_batch = False
                plot_2d_or_3d_image(data, epoch+1, writer, index=0, tag="images/image")
                plot_2d_or_3d_image(target, epoch+1, writer, index=0, tag="images/label")
                plot_2d_or_3d_image(preds, epoch+1, writer, index=0, tag="images/prediction")
                plot_2d_or_3d_image(pred_raw, epoch+1, writer, index=0, tag="images/prediction_raw")
            
            # save all images from the last epoch
            if torch.sum(target) > 2000 and (count%5)==0:
                plot_2d_or_3d_image(data, count, writer, index=0, tag="best_epoch/image")
                plot_2d_or_3d_image(target, count, writer, index=0, tag="best_epoch/label")
                plot_2d_or_3d_image(preds, count, writer, index=0, tag="best_epoch/prediction")
                plot_2d_or_3d_image(pred_raw, count, writer, index=0, tag="best_epoch/prediction_raw")

            #update loop
            loop.set_postfix(loss=loss.item())
            loop.set_description_str(
                    desc=f'{"Valid"} {epoch}',
                    refresh=True,
            )

        dsc_bg = dice_bg.aggregate().item()
        dsc_no_bg = dice.aggregate().item()
        iou_bg = iou_predbg.aggregate().item()
        iou_no_bg = iou_pred.aggregate().item()

        #métricas para tensorboard
        writer.add_scalar("val/mean_dice_bg", dsc_bg, epoch + 1)
        writer.add_scalar("val/mean_dice_no_bg", dsc_no_bg, epoch + 1)
        writer.add_scalar("val/iou_bg", iou_bg, epoch + 1)
        writer.add_scalar("val/iou_no_bg", iou_no_bg, epoch + 1)
        writer.add_scalar("val/loss", np.mean(epoch_loss), epoch + 1)

        dice_bg.reset()
        dice.reset()
        iou_predbg.reset()
        iou_pred.reset()

    model.train()
    return dsc_bg

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

def get_loader(args):
    slice=int(args.slice)

    dataset = args.dataset
    data_paths = {
        'hmd': '/A/motomed/datasets/processed/hmd',
        'LITSkaggle': '/A/motomed/datasets/processed/LITSkaggle',
        'amos22': '/A/motomed/datasets/processed/amos22', # multiple organs, take care :D
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
    print('lista treino:', len(train_list))
    print('quant traino:',  num_train)
    validation_list = _[num_train:]

    print('Number of patients: ', num_total_img)
    print('Number of training patients: ', len(train_list))
    print('Number of validation patients: ', len(validation_list))
    print('Number of test patients: ', len(test_list))

    # define transforms for image and segmentation
    train_transforms_img = Compose(
        [
            EnsureChannelFirst(),
            Resize(spatial_size=(256, 256), mode=("area")),
            #Flipd(keys=['mask'], spatial_axis=1),
            ScaleIntensityRange(
            a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
            ),
        ]
    )
    train_transforms_mask = Compose(
        [
            EnsureChannelFirst(),
            Resize(spatial_size=(256, 256), mode=("nearest")),
            #Flipd(keys=['mask'], spatial_axis=1),
        ]
    )

    val_transforms_img = Compose(
        [
            EnsureChannelFirst(),
            #Flipd(keys=['mask'], spatial_axis=1),
            Resize(spatial_size=(256, 256), mode=("area")),
            ScaleIntensityRange(
            a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
            ),
        ]
    )
    val_transforms_mask = Compose(
        [
            EnsureChannelFirst(),
            Resize(spatial_size=(256, 256), mode=("nearest")),
            #Flipd(keys=['mask'], spatial_axis=1),
        ]
    )

    train_ds = MotomedDataset(main_dir=DATA_PATH, delta_slice=slice, subset=train_list, transform_img=train_transforms_img, transform_mask=train_transforms_mask)
    val_ds = MotomedDataset(main_dir=DATA_PATH, delta_slice=slice, subset=validation_list, transform_img=val_transforms_img, transform_mask=val_transforms_mask)

    #plot_images(train_ds[50]['ct'], train_ds[50]['mask'], slice)

    train_loader = ThreadDataLoader(train_ds, num_workers=1, batch_size=args.batch_size, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=1, batch_size=1, shuffle=False)

    return train_loader, val_loader

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