import os
from torch.utils.data import Dataset
import numpy as np
import torch
from monai.transforms import (
    LoadImage,
)

class MotomedDataset(Dataset):
    def __init__(self, main_dir, delta_slice, subset, transform_img, transform_mask):
        '''
        main_dir: path to the dataset
        transform: transformation to apply to the images
        delta_slice: number of slices for each side of the center slice
        subset -> list of patients that will be used in this dataset
        '''

        imgs = sorted(os.listdir(os.path.join(main_dir, 'images')))

        patients_list = list()
        for img in imgs:
            if img[:4] not in patients_list:
                patients_list.append(img[:4])

        patients_count = {patient : [] for patient in patients_list}
        img_slices = list()
        mask_slices = list()

        # get list of number of slices for each patient
        for img in imgs:
            patients_count[img[:4]].append(img)

        # for each patient, get list of slices using the center slice and delta_slice
        # delta_slice is the number of slices before and after the center slice
        # for mask, we only need the center slice
        for patient in patients_count:
            if patient in subset:
                total = len(patients_count[patient])
                for slice in range(total):

                    # for the first slices
                    if slice < delta_slice:
                        group = patients_count[patient][slice-slice : slice+delta_slice+1]
                        for _ in range(delta_slice-slice):
                            group.insert(0, group[0])
                    
                    # for the last slices
                    elif slice > total - delta_slice - 1:
                        group = patients_count[patient][slice-delta_slice : slice+delta_slice+1]
                        for _ in range(delta_slice*2+1-len(group)):
                            group.append(group[-1])

                    # for the slices in the middle
                    else:
                        group = patients_count[patient][slice-delta_slice : slice+delta_slice+1]

                    img_slices.append(group)
                    mask_slices.append(patients_count[patient][slice])

        self.img_slices = img_slices
        self.mask_slices = mask_slices
        self.main_dir = main_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.delta_slice = delta_slice

    def __len__(self):
        return len(self.img_slices)

    def __getitem__(self, idx):
        img_slices = self.img_slices[idx]
        mask_slices = self.mask_slices[idx]

        loader = LoadImage(image_only=True)

        img = list()
        for slice in img_slices:
            img_path = os.path.join(self.main_dir, 'images', slice)
            tmp = loader(img_path)
            tmp = self.transform_img(tmp)
            img.append(tmp)
        img = torch.cat(img, dim=0)
        if self.delta_slice > 0:
            img = img.unsqueeze(0)
        
        mask_path = os.path.join(self.main_dir, 'mask', mask_slices)
        mask = loader(mask_path)
        mask = self.transform_mask(mask)

        return {'images':img, 'mask': mask}