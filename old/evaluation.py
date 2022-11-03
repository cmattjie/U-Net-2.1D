import argparse
import os
from tqdm import tqdm
import torch
import numpy as np

#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.Unet21D import UNET
from loader.LITS import MotomedDataset

from monai import metrics
from monai.data import decollate_batch, ThreadDataLoader
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirst,
    ScaleIntensityRange,
    LabelToMask,
)

from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image

import warnings
warnings.filterwarnings("ignore", message="Modifying image pixdim from")

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--argsed', default=False)
    parser.add_argument('--batch_size', default=4, type=int, help='Size for each mini batch.')
    parser.add_argument('--dataset', default='hmd', help='hmd or LITSkaggle or amos22')
    parser.add_argument('--gpu', default='0', help='GPU Number.')
    parser.add_argument('--load_dir', default='./checkpoints/load/my_checkpoint.pth.tar', type=str)
    parser.add_argument('--name', default='test', help='Run name on Tensorboard and savedirs.')
    parser.add_argument('--slice', default='1', help='Number of extra slices on each side')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args

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

    print('Number of patients: ', len(patients_list))

    # define transforms for image and segmentation
    transforms_img = Compose(
        [
            EnsureChannelFirst(),
            #Flipd(keys=['mask'], spatial_axis=1),
            ScaleIntensityRange(
            a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
            ),
        ]
    )
    transforms_mask = Compose(
        [
            EnsureChannelFirst(),
            #Flipd(keys=['mask'], spatial_axis=1),
            LabelToMask(select_labels=[1, 2], merge_channels=True)
        ]
    )

    ds = MotomedDataset(main_dir=DATA_PATH, delta_slice=slice, subset=patients_list, transform_img=transforms_img, transform_mask=transforms_mask)
    loader = ThreadDataLoader(ds, num_workers=1, batch_size=args.batch_size, shuffle=True)
    return loader


def eval_fn(loader, model, device, writer):
    post_trans= Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    model.eval()

    with torch.no_grad():
        loop = tqdm(loader, ncols=110)

        dice_bg = metrics.DiceMetric(include_background=True)
        dice= metrics.DiceMetric(include_background=False)
        iou_predbg= metrics.MeanIoU(include_background=True)
        iou_pred= metrics.MeanIoU(include_background=False)
        count = 0
        for batch_data in loop:
            data, target = batch_data["ct"].to(device), batch_data["mask"].to(device)

            pred_raw = model(data)
            preds=[post_trans(i) for i in decollate_batch(pred_raw)]

            dice_bg(preds, target)
            dice(preds, target)
            iou_predbg(preds, target)
            iou_pred(preds, target)
            count += 1
            # save all images
            plot_2d_or_3d_image(data, count, writer, index=0, tag="evaluation/image")
            plot_2d_or_3d_image(target, count, writer, index=0, tag="evaluation/label")
            plot_2d_or_3d_image(preds, count, writer, index=0, tag="evaluation/prediction")
            plot_2d_or_3d_image(pred_raw, count, writer, index=0, tag="evaluation/prediction_raw")

    dsc_bg = dice_bg.aggregate().item()
    dsc_no_bg = dice.aggregate().item()
    iou_bg = iou_predbg.aggregate().item()
    iou_no_bg = iou_pred.aggregate().item()

    #métricas para tensorboard
    writer.add_scalar("evaluation/mean_dice_bg", dsc_bg, 1)
    writer.add_scalar("evaluation/mean_dice_no_bg", dsc_no_bg, 1)
    writer.add_scalar("evaluation/iou_bg", iou_bg, 1)
    writer.add_scalar("evaluation/iou_no_bg", iou_no_bg, 1)

    dice_bg.reset()
    dice.reset()
    iou_predbg.reset()
    iou_pred.reset()

def main(args):
    torch.backends.cudnn.benchmark = True

    if args.argsed:
        print('loading args script')
        print('name:', args.name)
        print('slice:', args.slice)
        print('batch size:', args.batch_size)
        print('dataset:', args.dataset)
        print('gpu:', args.gpu)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device='cuda:0'

    model = UNET(in_channels=1, out_channels=1, slice=int(args.slice)).to(device)
    
    print('Loading model...')
    checkpoint = torch.load(args.load_dir)
    model.load_state_dict(checkpoint['state_dict'])
    print("Model loaded!")

    #tensorboard
    writer = SummaryWriter(f'runs/evaluation/{args.name}')

    #Get Loaders
    loader = get_loader(args)

    # check accuracy
    eval_fn(loader, model, device, writer)

    writer.close()

if __name__ == "__main__":
    args = get_args()
    main(args)