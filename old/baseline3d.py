from monai.networks.nets import UNet
import argparse
import os
from tqdm import tqdm
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.utils import first

from monai import metrics
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image

from monai.data import decollate_batch, ThreadDataLoader, Dataset
from monai.transforms import (
    LoadImaged,
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    ScaleIntensityRanged,
    Resized,
    LabelToMaskd,
    MapTransform,
    Rotate90d,
)

import warnings
warnings.filterwarnings("ignore", message="Modifying image pixdim from")


def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--argsed', default=False)
    parser.add_argument('--batch_size', default=4, type=int, help='Size for each mini batch.')
    parser.add_argument('--early_stop', default=15, type=int, help='Early stop criterion.')
    #parser.add_argument('--early_stop_eps', default=5e-6, type=float)
    parser.add_argument('--dataset', default='hmd', help='hmd or kaggle')
    parser.add_argument('--gpu', default='0', help='GPU Number.')
    parser.add_argument('--load_dir', default='./checkpoints/load/my_checkpoint.pth.tar', type=str)
    parser.add_argument('--name', default='test', help='Run name on Tensorboard and savedirs.')
    parser.add_argument('--slice', default='1', help='Number of extra slices on each side')
    parser.add_argument('--load_model', default=False, help='Load model from checkpoint?')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train.')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate.')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args

class DepthAdjustmentd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if img.shape[1]%8 != 0:
                tmp = torch.mul(torch.ones((1, 1, img.shape[2], img.shape[3])), torch.min(img))
                for i in range(8-(img.shape[1]%8)):
                    img = torch.concat((img, tmp), dim=1)
                d[key] = img
        return d
def get_loader(args):
    dataset = args.dataset
    data_paths = {
        'hmd': '/A/motomed/datasets/processed/liverHMD',
        'LITSkaggle': '/A/motomed/datasets/LITSkaggle',
    }
    DATA_PATH = data_paths[dataset]

    # Get list of patients to give as input to dataset
    patients_list = sorted(os.listdir(os.path.join(DATA_PATH, 'CT')))

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
    validation_list = _[num_train:]
    print('Number of patients: ', num_total_img)
    print('Number of training patients: ', len(train_list))
    print('Number of validation patients: ', len(validation_list))
    print('Number of test patients: ', len(test_list))

    train_list_mask = [os.path.join(DATA_PATH, 'mask', patient) for patient in train_list]
    for i in range(len(train_list_mask)):
        train_list_mask[i]=train_list_mask[i].replace('volume-', 'segmentation-')
    train_list_img = [os.path.join(DATA_PATH, 'CT', patient) for patient in train_list]
    
    data_train = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_list_img, train_list_mask)]

    val_list_mask = [os.path.join(DATA_PATH, 'mask', patient) for patient in validation_list]
    for i in range(len(val_list_mask)):
        val_list_mask[i]=val_list_mask[i].replace('volume-', 'segmentation-')
    val_list_img = [os.path.join(DATA_PATH, 'CT', patient) for patient in validation_list]
    
    data_val = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_list_img, val_list_mask)]


    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Rotate90d(keys=["image", "label"], spatial_axes=[0, 2]),
            Rotate90d(keys=["image", "label"], spatial_axes=[1, 2]),
            LabelToMaskd(keys=['label'], select_labels=[1, 2], merge_channels=True),
            DepthAdjustmentd(keys=['image', 'label']),
            Resized(keys=["image", "label"], spatial_size=(-1, 128, 128), mode=("area", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    train_ds = Dataset(data=data_train, transform=train_transforms)
    val_ds = Dataset(data=data_val, transform=train_transforms)

    check_ds = Dataset(data=data_val, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    print(f"image shape: {image.shape}, label shape: {label.shape}")

    train_loader = ThreadDataLoader(train_ds, num_workers=1, batch_size=args.batch_size, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=1, batch_size=1, shuffle=False)

    return train_loader, val_loader

def train_fn(loader, model, optimizer, loss_fn, scaler, device, epoch, writer):
    loop = tqdm(loader, ncols=140)
    epoch_loss=[]

    dice_bg = metrics.DiceMetric(include_background=True)
    dice= metrics.DiceMetric(include_background=False)
    iou_predbg= metrics.MeanIoU(include_background=True)
    iou_pred= metrics.MeanIoU(include_background=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    for batch_data in loop:
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        # forward
        with torch.cuda.amp.autocast():
            pred_raw = model(data)
            predictions = [post_trans(i) for i in decollate_batch(pred_raw)]
            try:
                loss = loss_fn(pred_raw, target)
            except:
                print('Error in loss function')
                continue
            epoch_loss.append(loss.item())

        #calcular métricas
        dice_bg(predictions, target)
        dice(predictions, target)
        iou_predbg(predictions, target)
        iou_pred(predictions, target)
    
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        loop.set_description_str(
                desc=f'{"Train"} {epoch}',
                refresh=True,
        )
        #loop.format_meter(ncols=100)

    dsc_bg = dice_bg.aggregate().item()
    dsc_no_bg = dice.aggregate().item()
    iou_bg = iou_predbg.aggregate().item()
    iou_no_bg = iou_pred.aggregate().item()

    #métricas para tensorboard
    writer.add_scalar("train/mean_dice_bg", dsc_bg, epoch + 1)
    writer.add_scalar("train/mean_dice_no_bg", dsc_no_bg, epoch + 1)
    writer.add_scalar("train/iou_bg", iou_bg, epoch + 1)
    writer.add_scalar("train/iou_no_bg", iou_no_bg, epoch + 1)  
    writer.add_scalar("train/loss", np.mean(epoch_loss), epoch + 1)

def check_accuracy(loader, model, loss_fn, device, writer, epoch, metric):
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
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)

            pred_raw = model(data)
            preds=[post_trans(i) for i in decollate_batch(pred_raw)]
            try:
                loss = loss_fn(pred_raw, target)
            except:
                print('Error in loss function')
                continue
            
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

def main(args):
    torch.backends.cudnn.benchmark = True
    save_path = os.path.join("./checkpoints", args.name, "my_checkpoint_test.pth.tar")
    os.makedirs(os.path.join("./checkpoints", args.name), exist_ok=True)
    if args.argsed:
        print('loading args script')
        print('name:', args.name)
        print('learning rate:', args.lr)
        print('batch size:', args.batch_size)
        print('early stop:', args.early_stop)
        print('dataset:', args.dataset)
        print('gpu:', args.gpu)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device='cuda'

    model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(64, 128, 256, 512), strides=(2, 2, 2))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    #tensorboard
    writer = SummaryWriter(f'runs/{args.name}')

    #Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #Get Loaders
    train_loader, val_loader = get_loader(args)

    #Load model if specified
    if args.load_model==str(True):
        print("Loading model...")
        checkpoint = torch.load(args.load_dir)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Model loaded!")

    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    best_dsc_bg = 0
    early_stop = 0
    dsc_bg=0
    for epoch in range(args.epochs):   

        # training model        
        train_fn(train_loader,
            model, 
            optimizer, 
            loss_fn, 
            scaler, 
            device=device,
            epoch=epoch,
            writer=writer
            )

        # check accuracy
        dsc_bg = check_accuracy(val_loader, model, loss_fn=loss_fn, device=device, epoch=epoch, writer=writer, metric=dsc_bg)
        if dsc_bg > best_dsc_bg:
            early_stop=0
            # save model
            print("New best dice score: ", dsc_bg, "         Saving Checkpoint...")
            checkpoint = {
                "epoch": epoch,            
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
                    
            torch.save(checkpoint, save_path)
            best_dsc_bg = dsc_bg

        # early stopping
        else:
            early_stop += 1
            if early_stop == args.early_stop:
                print("No improvement in {} epochs, early stopping".format(args.early_stop))
                writer.close()
                break
        
    writer.close()

if __name__ == "__main__":
    args = get_args()
    main(args)