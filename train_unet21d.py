import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import json

#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models.Unet21D import UNET21D

from utils import (
    get_loader,
    set_seeds,
    plot_images,
    LinearWarmupCosineAnnealingLR,    
    BinaryDiceLoss,
    Multi_BCELoss,
    DiceLoss
)

from monai import metrics, losses
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
    parser.add_argument('--dataset',            default='hmd', type=str, help='hmd or LITSkaggle or multiple MSD datasets')
    parser.add_argument('--gpu',                default=-1, help='GPU Number.')
    parser.add_argument('--load_dir',           default=None, help='Load model from checkpoint?')
    parser.add_argument('--name',               default='test', type=str, help='Run name on Tensorboard and savedirs.')
    parser.add_argument('--slice',              default=1, type=int,  help='Number of extra slices on each side')
    parser.add_argument('--epochs',             default=100, type=int, help='Number of epochs to train.')
    parser.add_argument('--loss',               default='dice', type=str, help='Loss function.')
    parser.add_argument('--optimizer',          default='adamw', type=str, help='Optimizer.')
    parser.add_argument('--scheduler',          default='plateau', type=str, help='Scheduler.')
    parser.add_argument('--lr',                 default=1e-4, type=float, help='Learning rate.')
    parser.add_argument('--seed',               default=42, type=int, help='Random seed.')
    parser.add_argument('--dropout',            default=0.0, type=float, help='Dropout rate.')
    
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    return args 

if __name__ == "__main__":
    #loading dicts and args and setting seeds
    args = get_args()
    set_seeds(args.seed)
    dicts = json.load(open('utils/dicts.json', 'r'))

    # Create dirs if not exist
    save_path = os.path.join("./checkpoints", args.name, "my_checkpoint_test.pth.tar")
    if not os.path.exists(save_path):
        os.makedirs(os.path.join("./checkpoints", args.name), exist_ok=True)
    
    # Print args
    print('name:', args.name)
    print('loss:', dicts['loss'][args.loss])
    print('learning rate:', args.lr)
    print('optimizer:', args.optimizer)
    print('scheduler:', args.scheduler)
    print('slice:', args.slice)
    print('batch size:', args.batch_size)
    print('early stop:', args.early_stop)
    print('dataset:', args.dataset)
    print('gpu:', args.gpu)
    print('dropout:', args.dropout)
    print('seed:', args.seed)
    
    device=f'cuda:{args.gpu}'

    out_channels = dicts['out_channels'][args.dataset]
    model = UNET21D(in_channels=1, out_channels=out_channels, slice=int(args.slice), dropout=args.dropout).to(device)
    
    #tensorboard
    board = SummaryWriter(f'runs/{args.name}')
    
    #Loss, optimizer and scheduler
    #TODO incluir loss composto
    loss_fn = eval(dicts['loss'][args.loss])
    optimizer = eval(dicts['optimizer'][args.optimizer])
    scheduler = eval(dicts['scheduler'][args.scheduler])
    
    #Get Loaders
    train_loader, val_loader = get_loader(args)
    
    #Load model if specified
    if args.load_dir not in [None, 'None']:
        print(f"Loading model from {args.load_dir}")
        #load_checkpoint(torch.load(load_dir), model)
        checkpoint = torch.load(args.load_dir)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Model loaded!")
    
    #utils
    scaler = torch.cuda.amp.GradScaler()
    best_dsc_bg = 0
    early_stop = 0
    dsc_bg=0
    
    post_trans =  Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=0.5, num_classes=2)])
    #post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=(0.5, 1.5), num_classes=3)])
    #TODO multiclass
    #post_trans = Compose([Activations(sigmoid=False), AsDiscrete(argmax=True)])
    
    for epoch in range(args.epochs):
        #early stop
        if early_stop == args.early_stop:
            print('Early stop reached. Stopping training...')
            break
        
        for is_train, description in zip([True, False], ["Train", "Valid"]):
        
            # for testing only
            # if is_train and epoch==0:
            #     continue
            
            loop = tqdm(train_loader, ncols=140) if is_train else tqdm(val_loader, ncols=140)
            loop.set_description_str(desc=f'{description} {epoch}', refresh=True)
            
            epoch_loss = []
            save_batch = True
            count=0
            
            #TODO verificar se true ou false
            dice_pred = metrics.DiceMetric(include_background=True, reduction='mean_batch')
            iou_pred = metrics.MeanIoU(include_background=True, reduction='mean_batch')
            
            #set model to train or eval
            if is_train: 
                model.train()
                model.requires_grad_(True)
            else: 
                model.eval()
                model.requires_grad_(False)
            
            for batch_data in loop:
                data, target = batch_data['images'].to(device), batch_data['mask'].to(device)
                                
                with torch.cuda.amp.autocast():
                    pred_raw = model(data)
                    predictions = [post_trans(i) for i in decollate_batch(pred_raw)]
                    loss = loss_fn(pred_raw, target)
                    epoch_loss.append(loss.item())
                
                save_target_dict = dicts['save_target_dict'][args.dataset]
                
                #TODO save images from all channels
                # save one image for each epoch
                if not is_train and save_batch and (torch.sum(target[0]) > int(save_target_dict)):
                    save_batch = False
                    plot_2d_or_3d_image(data, epoch+1, board, index=0, tag="images/image")
                    plot_2d_or_3d_image(target, epoch+1, board, index=0, tag="images/label")
                    plot_2d_or_3d_image(predictions, epoch+1, board, index=0, tag="images/prediction")
                    plot_2d_or_3d_image(pred_raw, epoch+1, board, index=0, tag="images/prediction_raw")
            
                # save 5 images from the epoch after the one with the best dice
                #TODO keep images and only save them at the end if the dice is better
                if not is_train and (torch.sum(target[0]) > int(save_target_dict)) and (count<6) and early_stop==0:
                    count += 1
                    plot_2d_or_3d_image(data, count, board, index=0, tag="last_epoch/image")
                    plot_2d_or_3d_image(target, count, board, index=0, tag="last_epoch/label")
                    plot_2d_or_3d_image(predictions, count, board, index=0, tag="last_epoch/prediction")
                    plot_2d_or_3d_image(pred_raw, count, board, index=0, tag="last_epoch/prediction_raw")
                    
                #calcular métricas
                dice_pred(predictions, target)
                iou_pred(predictions, target)
                
                # backward
                if is_train:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                # update tqdm loop
                loop.set_postfix_str(
                    s=f'Mean loss: {np.mean(epoch_loss).mean():.4f}',
                    refresh=True
                )
                    
            #get metrics
            dice = dice_pred.aggregate()
            iou = iou_pred.aggregate()
            
            #tensorboard
            board.add_scalar(f'{description}/dice', dice.mean(), epoch)
            board.add_scalar(f'{description}/loss', np.mean(epoch_loss).mean(), epoch)
            for i in range(out_channels):
                board.add_scalar(f'{description}/dice_{i+1}', dice[i], epoch)
                board.add_scalar(f'{description}/iou_{i+1}', iou[i], epoch)
            
            #reset metrics
            dice_pred.reset()
            iou_pred.reset()
        
        #get mean dice
        print(dice)
        dice = dice.mean()
        print(dice)
        
        #scheduler
        scheduler.step(-dice)
        
        #only for validation 
        if dice > best_dsc_bg:
            best_dsc_bg = dice
            early_stop = 0
            print('New best dice: ', best_dsc_bg, 'Saving model...')
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, save_path)
        else:
            early_stop += 1                
                
    board.close()
                        
        
        