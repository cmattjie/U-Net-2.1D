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
    set_seeds,
    plot_images
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
    parser.add_argument('--dataset',            default='hmd', choices=['hmd', 'LITSkaggle'], help='hmd or LITSkaggle')
    parser.add_argument('--gpu',                default=-1, help='GPU Number.')
    parser.add_argument('--load_dir',           default=None, help='Load model from checkpoint?')
    parser.add_argument('--name',               default='test', type=str, help='Run name on Tensorboard and savedirs.')
    parser.add_argument('--slice',              default=1, type=int,  help='Number of extra slices on each side')
    parser.add_argument('--epochs',             default=100, type=int, help='Number of epochs to train.')
    parser.add_argument('--lr',                 default=1e-4, type=float, help='Learning rate.')
    parser.add_argument('--seed',               default=42, type=int, help='Random seed.')
    
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    return args 

if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.benchmark = True
    set_seeds(args.seed)
    
    # Create dirs if not exist
    save_path = os.path.join("./checkpoints", args.name, "my_checkpoint_test.pth.tar")
    if not os.path.exists(save_path):
        os.makedirs(os.path.join("./checkpoints", args.name), exist_ok=True)
        
    # Print args
    print('name:', args.name)
    print('learning rate:', args.lr)
    print('slice:', args.slice)
    print('batch size:', args.batch_size)
    print('early stop:', args.early_stop)
    print('dataset:', args.dataset)
    print('gpu:', args.gpu)
    
    device=f'cuda:{args.gpu}'
    
    model = UNET21D(in_channels=1, out_channels=1, slice=int(args.slice)).to(device)
    
    #tensorboard
    board = SummaryWriter(f'runs/{args.name}')
    
    #Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
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
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    for epoch in range(args.epochs):
        
        #early stop
        if early_stop == args.early_stop:
            print('Early stop reached. Stopping training...')
            break
        
        for is_train, description in zip([True, False], ["Train", "Valid"]):
        
            loop = tqdm(train_loader, ncols=140) if is_train else tqdm(val_loader, ncols=140)
            loop.set_description_str(desc=f'{description} {epoch}', refresh=True)
            
            epoch_loss = []
            save_batch = True
            count=0
            
            dice_pred = metrics.DiceMetric(include_background=True)
            iou_pred = metrics.MeanIoU(include_background=True)
            
            #set model to train or eval
            if is_train: 
                model.train()
                model.requires_grad_(True)
            else: 
                model.eval()
                model.requires_grad_(False)
            
            for batch_data in loop:
                data, target = batch_data['ct'].to(device), batch_data['mask'].to(device)
                count += 1
                
                with torch.cuda.amp.autocast():
                    pred_raw = model(data)
                    predictions = [post_trans(i) for i in decollate_batch(pred_raw)]
                    loss = loss_fn(pred_raw, target)
                    epoch_loss.append(loss.item())
                    
                #saving predictions
                # save one image for visualization of the curent epoch (only once per epoch)
                if not is_train and save_batch and torch.sum(target) > 10000:
                    save_batch = False
                    plot_2d_or_3d_image(data, epoch+1, board, index=0, tag="images/image")
                    plot_2d_or_3d_image(target, epoch+1, board, index=0, tag="images/label")
                    plot_2d_or_3d_image(predictions, epoch+1, board, index=0, tag="images/prediction")
                    plot_2d_or_3d_image(pred_raw, epoch+1, board, index=0, tag="images/prediction_raw")
            
                # save 5 images from the last epoch
                if not is_train and torch.sum(target) > 4000 and (count%5)==0:
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
            
            dice = dice_pred.aggregate().item()
            iou = iou_pred.aggregate().item()
            
            #tensorboard
            board.add_scalar(f'{description}/loss', np.mean(epoch_loss).mean(), epoch)
            board.add_scalar(f'{description}/dice', dice, epoch)
            board.add_scalar(f'{description}/iou', iou, epoch)
        
        #only for validation    
        dice_pred.reset()
        iou_pred.reset()
        
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
                        
        
        