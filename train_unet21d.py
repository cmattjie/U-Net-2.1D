import argparse
import os
from tqdm import tqdm
import torch
import numpy as np

#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from models.Unet21D import UNET21D

from utils import (
    get_loader,
)

from monai import metrics
from monai.data import decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)

from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image

import warnings
warnings.filterwarnings("ignore", message="Modifying image pixdim from")

def get_args():
    parser = argparse.ArgumentParser(description='parameters for training')

    parser.add_argument('--batch_size',         default=4, type=int, help='Size for each mini batch.')
    parser.add_argument('--early_stop',         default=10, type=int, help='Early stop criterion.')
    #parser.add_argument('--early_stop_eps', default=5e-6, type=float)
    parser.add_argument('--dataset',            default='hmd', choices=['hmd', 'LITSkaggle'], help='hmd or LITSkaggle')
    parser.add_argument('--gpu',                default=-1, help='GPU Number.')
    parser.add_argument('--load_model',         default=False, help='Load model from checkpoint?')
    parser.add_argument('--load_dir',           default='./checkpoints/load/my_checkpoint.pth.tar', type=str , help='Load model from checkpoint?')
    parser.add_argument('--name',               default='test', type=str, help='Run name on Tensorboard and savedirs.')
    parser.add_argument('--slice',              default=1, type=int,  help='Number of extra slices on each side')
    parser.add_argument('--epochs',             default=100, type=int, help='Number of epochs to train.')
    parser.add_argument('--lr',                 default=1e-4, type=float, help='Learning rate.')
    
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    return args 

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def train_fn(loader, model, optimizer, loss_fn, scaler, device, epoch, writer):
    loop = tqdm(loader, ncols=140)
    epoch_loss=[]

    dice_bg = metrics.DiceMetric(include_background=True)
    dice= metrics.DiceMetric(include_background=False)
    iou_predbg= metrics.MeanIoU(include_background=True)
    iou_pred= metrics.MeanIoU(include_background=False)

    for batch_data in loop:
        data, target = batch_data["ct"].to(device), batch_data["mask"].to(device)
        # forward
        with torch.cuda.amp.autocast():
            pred_raw = model(data)
            predictions = [post_trans(i) for i in decollate_batch(pred_raw)]
            loss = loss_fn(pred_raw, target)
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

if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.benchmark = True
    
    # Create dirs
    save_path = os.path.join("./checkpoints", args.name, "my_checkpoint_test.pth.tar")
    os.makedirs(os.path.join("./checkpoints", args.name), exist_ok=True)

    # Print args
    print('loading args script')
    print('name:', args.name)
    print('learning rate:', args.lr)
    print('slice:', args.slice)
    print('batch size:', args.batch_size)
    print('early stop:', args.early_stop)
    print('dataset:', args.dataset)
    print('gpu:', args.gpu)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device='cuda:0'

    model = UNET21D(in_channels=1, out_channels=1, slice=int(args.slice)).to(device)

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
        #load_checkpoint(torch.load(load_dir), model)
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
        
        if early_stop == args.early_stop:
            print('Early stop reached. Stopping training...')
            break
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
        dsc_bg = valid(val_loader, model, loss_fn=loss_fn, device=device, epoch=epoch, writer=writer, metric=dsc_bg)
        if dsc_bg > best_dsc_bg:
            early_stop=0
            # save model
            print("New best dice score: ", dsc_bg, ". Saving Checkpoint...")
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
        
    writer.close()