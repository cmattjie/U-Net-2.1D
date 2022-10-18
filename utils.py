import torch
import torchvision
import os
from glob import glob
import numpy as np

import monai
from dataset import MotomedDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics as sk
from monai import metrics
from monai.data import decollate_batch, ThreadDataLoader
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    ScaleIntensityRange,
    Flip,
)

def check_accuracy(loader, model, device, writer, epoch):
    post_trans= Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    model.eval()

    with torch.no_grad():
        loop = tqdm(loader)

        dice_bg = metrics.DiceMetric(include_background=True)
        dice= metrics.DiceMetric(include_background=False)
        iou_predbg= metrics.MeanIoU(include_background=True)
        iou_pred= metrics.MeanIoU(include_background=False)

        all_y_pred = 0
        all_y_true = 0
        save_batch = True
        count = 0
        for batch_data in loop:
            data, target = batch_data["ct"].to(device), batch_data["mask"].to(device)

            pred_raw = model(data)
            preds=[post_trans(i) for i in decollate_batch(pred_raw)]

            dice_bg(preds, target)
            dice(preds, target)
            iou_predbg(preds, target)
            iou_pred(preds, target)
            
            # IOU calculation
            y_pred = pred_raw.cpu().numpy()
            y_pred = y_pred > 0.5
            y_pred = y_pred.reshape(-1)
            y_pred = y_pred.astype(np.uint8)
            all_y_pred = np.append(all_y_pred, y_pred)
            y_true = target.cpu().numpy()
            y_true = y_true > 0.5
            y_true = y_true.reshape(-1)
            y_true = y_true.astype(np.uint8)
            all_y_true = np.append(all_y_true, y_true)

            count += 1

            # save one image for visualization of the curent epoch (only once per epoch)
            if np.sum(y_true) > 50 and save_batch:
                save_batch = False
                plot_2d_or_3d_image(data, epoch+1, writer, index=0, tag="images/image")
                plot_2d_or_3d_image(target, epoch+1, writer, index=0, tag="images/label")
                plot_2d_or_3d_image(preds, epoch+1, writer, index=0, tag="images/prediction")
                plot_2d_or_3d_image(pred_raw, epoch+1, writer, index=0, tag="images/prediction_raw")
            
            # save all images from the last epoch
            plot_2d_or_3d_image(data, count, writer, index=0, tag="best_epoch/image")
            plot_2d_or_3d_image(target, count, writer, index=0, tag="best_epoch/label")
            plot_2d_or_3d_image(preds, count, writer, index=0, tag="best_epoch/prediction")
            plot_2d_or_3d_image(pred_raw, count, writer, index=0, tag="best_epoch/prediction_raw")

        fpr, tpr, _ = sk.roc_curve(all_y_true, all_y_pred)
        roc_auc = sk.auc(fpr, tpr)
    
        dsc_bg = dice_bg.aggregate().item()
        dsc_no_bg = dice.aggregate().item()
        iou_bg = iou_predbg.aggregate().item()
        iou_no_bg = iou_pred.aggregate().item()

        writer.add_scalar("val/mean_dice_bg", dsc_bg, epoch + 1)
        writer.add_scalar("val/mean_dice_no_bg", dsc_no_bg, epoch + 1)
        writer.add_scalar("val/iou_bg", iou_bg, epoch + 1)
        writer.add_scalar("val/iou_no_bg", iou_no_bg, epoch + 1)
        writer.add_scalar("val/roc_auc", roc_auc, epoch + 1)

        dice_bg.reset()
        dice.reset()
        iou_predbg.reset()
        iou_pred.reset()

    model.train()
    return roc_auc

def save_predictions_as_imgs(
    #falta arrumar
    loader, model, folder="saved_images", device = 'cuda'):
    model.eval()
    with torch.no_grad():
        
        nsaves = 1
        for batch_data in loader:
            while nsaves>0:
                #pegar apenas imagens e não exames + ou batchs (depois fazer for para exame e slice)
                data_   = batch_data["ct"]
                target_ = batch_data["mask"]
                
                #print(data_.shape, target_.shape)
                data = data_[:, 50:-10:15, :, :, :].to(device)
                target = target_[:, 50:-10:15, :, :, :].to(device)
                    
                data=torch.swapaxes(data,0,1)
                target=torch.swapaxes(target,0,1)
                data=torch.swapaxes(data,2,4)
                target=torch.swapaxes(target,2,4)
                
                #print(data.shape)
                #print(target.shape)
                                
                preds = torch.sigmoid(model(data))
                preds = (preds > 0.5).float()
                
                # Check whether the specified path exists or not
                isExist = os.path.exists(folder)
                if not isExist:
                    os.makedirs(folder)
                for i in range(data.shape[0]):
                    torchvision.utils.save_image(data[i], os.path.join(folder, f"image_{i}.png"))
                    torchvision.utils.save_image(target[i], os.path.join(folder, f"label_{i}.png"))
                    torchvision.utils.save_image(preds[i], os.path.join(folder, f"pred_{i}.png"))
                nsaves -= 1

    model.train()

def plot_images(image3d, mask, slice):
    '''
    Plot 2D image from 3D image, given a slice and a center
    image3d: exame
    slice: número de cortes para cada lado do centro
    salva a imagem
    '''
    import matplotlib.pyplot as plt
    import numpy as np

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

    dataset = 'kaggle_liver'
    data_paths = {
        'hmd': '/A/motomed/datasets/processed/hmd',
        'kaggle_liver': '/A/motomed/datasets/processed/kaggle_liver',
    }
    DATA_PATH = data_paths[dataset]

    # Get list of patients to give as input to dataset
    imgs = sorted(os.listdir(os.path.join(DATA_PATH, 'CT')))
    patients_list = list()
    for img in imgs:
        if img[:4] not in patients_list:
            patients_list.append(img[:4])

    # TEST ONLY, REMOVE LATER
    patients_list = patients_list[:2]
    
    num_img = len(patients_list)
    num_train = int(num_img * 0.75)

    train_list = patients_list[:num_train]
    test_list = patients_list[num_train:]

    print('Number of patients: ', num_img)
    print('Number of training patients: ', len(train_list))
    print('Number of testing patients: ', len(test_list))

    # define transforms for image and segmentation
    train_transforms_img = Compose(
        [
            EnsureChannelFirst(),
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
            #Flipd(keys=['mask'], spatial_axis=1),
        ]
    )

    val_transforms_img = Compose(
        [
            EnsureChannelFirst(),
            #Flipd(keys=['mask'], spatial_axis=1),
            ScaleIntensityRange(
            a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
            ),
        ]
    )
    val_transforms_mask = Compose(
        [
            EnsureChannelFirst(),
            #Flipd(keys=['mask'], spatial_axis=1),
        ]
    )

    train_ds = MotomedDataset(main_dir=DATA_PATH, delta_slice=slice, subset=train_list, transform_img=train_transforms_img, transform_mask=train_transforms_mask)
    val_ds = MotomedDataset(main_dir=DATA_PATH, delta_slice=slice, subset=test_list, transform_img=val_transforms_img, transform_mask=val_transforms_mask)

    plot_images(train_ds[50]['ct'], train_ds[50]['mask'], slice)

    train_loader = ThreadDataLoader(train_ds, num_workers=1, batch_size=args.batch_size, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=2, batch_size=1, shuffle=False)

    return train_loader, val_loader