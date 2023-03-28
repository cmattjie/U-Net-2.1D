import os
from glob import glob

from torch.utils.data import DataLoader
from tqdm import tqdm
import monai
from monai.data import ThreadDataLoader, NibabelWriter
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Flipd,
    Rotate90d,
    LabelToMaskd,
)

import warnings
warnings.filterwarnings("ignore", message="Modifying image pixdim from")

# hmd → tem que flipar a imagem e rotacionar 90 graus 1 vez 
# ((Flipd(keys=['mask'], spatial_axis=1), Rotate90d(keys=['ct', 'mask'], k=2, spatial_axes=(0, 1))

# kaggle_liver → tem que rotacionar 90 graus 1 vez, e juntar os labels 1 (fígado) e 2 (lesões) em 1 só
# (LabelToMaskd(keys=['mask'], select_labels=[1, 2], merge_channels=True))

# AMOS22 → tem que rotacionar 90 graus 2 vezes, possúi varios labels
# (Rotate90d(keys=['ct', 'mask'], k=2, spatial_axes=(0, 1))

data_paths = {
    'hmd': '/mnt/B-SSD/unet21d_slices/datasets/liver/supervised',
    'LITSkaggle': '/mnt/B-SSD/unet21d_slices/datasets/LITSkaggle',
    'amos22': '/mnt/B-SSD/unet21d_slices/datasets/amos22/AMOS22',
    'MSD_Brain': '/mnt/B-SSD/unet21d_slices/datasets/MSD/Task01_BrainTumour',
    'MSD_Heart': '/mnt/B-SSD/unet21d_slices/datasets/MSD/Task02_Heart',
    'MSD_Liver': '/mnt/B-SSD/unet21d_slices/datasets/MSD/Task03_Liver',
    'MSD_Hippocampus': '/mnt/B-SSD/unet21d_slices/datasets/MSD/Task04_Hippocampus',
    'MSD_Prostate': '/mnt/B-SSD/unet21d_slices/datasets/MSD/Task05_Prostate',
    'MSD_Lung': '/mnt/B-SSD/unet21d_slices/datasets/MSD/Task06_Lung',
    'MSD_Pancreas': '/mnt/B-SSD/unet21d_slices/datasets/MSD/Task07_Pancreas',
    'MSD_HepaticVessel': '/mnt/B-SSD/unet21d_slices/datasets/MSD/Task08_HepaticVessel',
    'MSD_Spleen': '/mnt/B-SSD/unet21d_slices/datasets/MSD/Task09_Spleen',
    'MSD_Colon': '/mnt/B-SSD/unet21d_slices/datasets/MSD/Task10_Colon',
    
}
datasets = ['MSD_Hippocampus', 'MSD_Pancreas', 'MSD_HepaticVessel']

for dataset in datasets:
    dataset_path = data_paths[dataset]
    print('loading dataset from:', dataset_path)
    processed_path = os.path.join('/mnt/B-SSD/unet21d_slices/datasets/processed', dataset)
    imagefolder = 'images'
    maskfolder = 'mask'
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
        os.makedirs(os.path.join(processed_path, imagefolder))
        os.makedirs(os.path.join(processed_path, maskfolder))

    ct = sorted(glob(os.path.join(dataset_path, 'imagesTr', '*.nii*')))
    mask = sorted(glob(os.path.join(dataset_path, 'labelsTr', '*.nii*')))

    print("Found {} CT scans and {} masks".format(len(ct), len(mask)))

    files = [{'ct': ct_, 'mask': mask_} for ct_, mask_ in zip(ct, mask)]

    # define transforms for image and segmentation
    transforms = Compose(
        [
            LoadImaged(keys=['ct', 'mask'], image_only=True),
            EnsureChannelFirstd(keys=['ct', 'mask']),
            Rotate90d(keys=['ct', 'mask'], k=2, spatial_axes=(0, 1)),
            #LabelToMaskd(keys=['mask'], select_labels=[1, 2], merge_channels=True),
            #Flipd(keys=['mask'], spatial_axis=1), #hmd data are flipped
        ]
    )

    ds = monai.data.Dataset(data=files, transform=transforms)
    loader = ThreadDataLoader(ds, num_workers=1, batch_size=1, shuffle=False)

    writer_ct = NibabelWriter()
    writer_mask = NibabelWriter()

    patient_num = 0
    for batch_data in tqdm(loader):
        ct, mask = batch_data['ct'], batch_data['mask']

        for slice in tqdm(range(ct.shape[4]), leave=False):
            writer_ct.set_data_array(ct[0, 0, :, :, slice], channel_dim=None)
            # first four digits are patient number, last four are slice number
            writer_ct.write(os.path.join(processed_path, imagefolder, '{:04d}_{:04d}.nii'.format(patient_num, slice)))

            writer_mask.set_data_array(mask[0, 0, :, :, slice], channel_dim=None)
            # first four digits are patient number, last four are slice number
            writer_mask.write(os.path.join(processed_path, maskfolder, '{:04d}_{:04d}.nii'.format(patient_num, slice)))
                
        patient_num += 1