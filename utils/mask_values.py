import numpy as np
import os
import nibabel as nib

def get_mask_values(mask):
    """Return the set of unique values in a medical image segmentation mask"""
    return set(np.unique(mask))

#get mask
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
datasets=['LITSkaggle', 'amos22', 'MSD_Colon', 'MSD_HepaticVessel', 'MSD_Hippocampus', 'MSD_Lung', 'MSD_Pancreas', 'MSD_Spleen']

for dataset in datasets:
    DATA_PATH = os.path.join(data_paths[dataset], 'labelsTr')
    #get first file on folder without getting hidden files
    listfolder = [f for f in os.listdir(DATA_PATH) if not f.startswith('.')]
    
    #get mask
    mask_path = os.path.join(DATA_PATH, listfolder[0])
    
    #read nii mask using monai
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
        
    print(mask_path)
    print(get_mask_values(mask_data))
    
     
