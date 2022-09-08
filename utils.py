import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai import metrics
from monai.data import decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device):
    slicecounter=0
    dsc_predbg=0
    dsc_pred=0
    post_trans= Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    model.eval()

    with torch.no_grad():
        loop = tqdm(loader)
        batch_size = 5

        for batch_data in loop:
            data_, target_ = batch_data["ct"].to('cpu'), batch_data["mask"].to('cpu')
            size = target_.shape[1]
            slicecounter=slicecounter+size
            for mini_batch in range(0, batch_size, size):
                
                data = data_[0, mini_batch:mini_batch+batch_size, :, :, :].unsqueeze(0).to(device)
                target = target_[0, mini_batch:mini_batch+batch_size, :, :, :].unsqueeze(0).to(device)
                #print(data.shape, target.shape)
                
                data=torch.swapaxes(data,0,1)
                target=torch.swapaxes(target,0,1)
                data=torch.swapaxes(data,2,4)
                target=torch.swapaxes(target,2,4)
                print("checking accuracy...")
                target=target.squeeze(1)

                preds = model(data)
                preds=[post_trans(i) for i in decollate_batch(preds)]

                dice_bg = metrics.DiceMetric(include_background=True)
                dice= metrics.DiceMetric(include_background=False)

                #iou_predbg= metrics.compute_meaniou(preds, target, include_background=True)
                #iou_pred= metrics.compute_meaniou(preds, target, include_background=False)
                dsc_predbg+= dice_bg(preds, target)
                dsc_pred+= dice(preds, target)
                #print("iou_predbg: ", iou_predbg)
                #print("iou_pred: ", iou_pred)
                
                
        dsc_meanbg=dsc_predbg/slicecounter
        dsc_mean=dsc_pred/slicecounter

        print("dsc_meanbg: ", dsc_meanbg)
        print("dsc_mean: ", dsc_mean)

    #print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    #falta arrumar
    loader, model, folder="saved_images", device = 'cuda'):
    model.eval()
    #for idx, (x, y) in enumerate(loader):
        #x = x.to(device)
    nslices = 5

    #arrumar esse valor como mini batch para ter imagens mais espaçadas (dar len /2 para pegar o meio)
    nst = 90

    nsaves = 1
    for batch_data in loader:
        while nsaves>0:
            #pegar apenas imagens e não exames + ou batchs (depois fazer for para exame e slice)
            data_   = batch_data["ct"]
            target_ = batch_data["mask"]
            #print(data_.shape, target_.shape)
            data = data_[:, 15:-15:30, :, :, :].to(device)
            target = target_[:, 15:-15:30, :, :, :].to(device)
            #print(data.shape, target.shape)
                
            data=torch.swapaxes(data,0,1)
            target=torch.swapaxes(target,0,1)
            data=torch.swapaxes(data,2,4)
            target=torch.swapaxes(target,2,4)
            
            print(data.shape)
            print(target.shape)
            

            
            preds = torch.sigmoid(model(data))
            #print(preds.shape)
            preds = (preds > 0.5).float()

            
            #print(preds.shape)
            #print(target.shape)
            for i in range(data.shape[0]):
                torchvision.utils.save_image(
                    preds[i,0,:,:], f"{folder}/pred_{i}.png"
                )
                torchvision.utils.save_image(
                    target[i,0,0,:,:], f"{folder}/target_{i}.png"
                )
                torchvision.utils.save_image(
                    data[i,0,0,:,:], f"{folder}/data_{i}.png"
                )
            #torchvision.utils.save_image(preds, f"{folder}/pred_{nsaves}.png")
            #torchvision.utils.save_image(target.unsqueeze(1), f"{folder}/{nsaves}.png")
            nsaves -= 1

    model.train()