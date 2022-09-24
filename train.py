import argparse
import os
from tqdm import tqdm
import torch

#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNET

from utils import (
    check_accuracy,
    save_predictions_as_imgs,
    get_loader,
)

from monai.metrics import DiceMetric
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
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--argsed', default=False)
    parser.add_argument('--batch_size', default=4, type=int, help='Size for each mini batch.')
    #parser.add_argument('--early_stop', default=15, type=int)
    #parser.add_argument('--early_stop_eps', default=5e-6, type=float)
    parser.add_argument('--dataset', default='hmd', help='hmd or kaggle')
    parser.add_argument('--gpu', default='0', help='GPU Number.')
    parser.add_argument('--load_dir', default='./checkpoints/load/my_checkpoint.pth.tar', type=str)
    parser.add_argument('--name', default='test', help='Run name on Tensorboard.')
    parser.add_argument('--slice', default='1', help='Number of extra slices on each side')
    parser.add_argument('--load_model', default=False, help='Load model from checkpoint?')
    parser.add_argument('--save_dir', default='./checkpoints', help='Number of epochs to train.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train.')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate.')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args 

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader)

    for batch_data in loop:
        data, target = batch_data["ct"].to(device), batch_data["mask"].to(device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, target)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main(args):
    if args.argsed:
        print('loading args script')
        print('learning rate', args.lr)
    epochs=args.epochs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device='cuda:0'

    model = UNET(in_channels=1, out_channels=1, slice=int(args.slice)).to(device)

    #tensorboard
    writer = SummaryWriter(f'runs/{args.name}')

    #Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #Get Loaders
    train_loader, val_loader = get_loader(args)
    
    #Load model is specified
    if args.load_model==str(True):
        print("Loading model...")
        load_dir=args.load_dir
        #load_checkpoint(torch.load(load_dir), model)
        checkpoint = torch.load(load_dir)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Model loaded!")

    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(epochs):

        #if args.load_model == True: # para teste, remover depois disso
            
        # training model
        print("Training Epoch: ", epoch)           
        train_fn(train_loader,
            model, 
            optimizer, 
            loss_fn, 
            scaler, 
            device=device,
            )
        
        # save model
        checkpoint = {
            "epoch": epoch,            
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_path = os.path.join(args.save_dir, "my_checkpoint_test.pth.tar")
        torch.save(checkpoint, save_path)
        print("Saved checkpoint")

        print("check performance on validation set")
        # check accuracy
        check_accuracy(val_loader, model, device=device, epoch=epoch, writer=writer)
        
        #print("saving predictions")
        # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, 
        #     model, 
        #     folder=os.path.join("/A/motomed/semantic_segmentation_unet/saved_images/2.5D", str(epoch)), 
        #     device=device
        # )
        
if __name__ == "__main__":
    args = get_args()
    main(args)

    