from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union

from monai.transforms import Transform, MapTransform
import torch
import numpy as np

class SliceFromVolume(Transform):
    """
    Inputs:
        - image: 3D image (need to be load with LoadImage(image_only=True))
        - num_slices: number of slices to extract from each side of the image
    Outputs:
        - image: image with num_slices slices extracted from each side of the image
    """
    def __init__(self, num_slices):
        self.num_slices = num_slices

    def __call__(self, image):
        results: List[torch.Tensor] = []
        for i in range(self.num_slices, image.shape[-1] - self.num_slices):
            results.extend([image[0, :, :, i - self.num_slices:i + self.num_slices + 1].unsqueeze(0)])
       
        return torch.cat(results, 0)#.half()

class SliceFromVolumed(MapTransform):
    """
    Dictionary-based wrapper of abstract class SliceFromVolume.

    Inputs:
        - keys: keys of the corresponding items to be transformed.
        - image: 3D image (need to be load with LoadImage(image_only=True))
        - num_slices: number of slices to extract from each side of the image
    Outputs:
        - image: 3D image with num_slices slices extracted from each side of the image
    """
    def __init__(self, keys, num_slices):
        super().__init__(keys)
        self.num_slices = num_slices

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        count = 0
        for key in self.key_iterator(d):
            results: List[torch.Tensor] = []
            if count == 0:
                for i in range(self.num_slices, d[key].shape[-1] - self.num_slices):
                #for i in range(self.num_slices, 10):
                    results.extend([d[key][0, :, :, i - self.num_slices:i + self.num_slices + 1].unsqueeze(0)])
                count += 1
            else:
                for i in range(self.num_slices, d[key].shape[-1] - self.num_slices):
                #for i in range(self.num_slices, 10):
                    results.extend([d[key][0, :, :, i].unsqueeze(0).unsqueeze(-1)])

            d[key] = torch.cat(results, 0)#.half()
        return d

