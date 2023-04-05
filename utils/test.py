import numpy as np

def get_mask_values(mask):
    """Return the set of unique values in a medical image segmentation mask"""
    return set(np.unique(mask))

# Example usage
mask = np.array([
    [[0, 0, 0], [0, 1, 1], [0, 1, 2]],
    [[0, 0, 0], [0, 1, 1], [0, 1, 2]],
    [[0, 0, 0], [0, 1, 1], [0, 2, 2]],
])
mask2 = np.array([
    [[0, 0, 0], [0, 1, 1], [0, 1, 2]],
    [[0, 0, 0], [0, 1, 1], [0, 1, 2]],
    [[0, 0, 0], [0, 1, 1], [0, 2, 2]],
])
values = get_mask_values(mask)

# Create copy of mask with 2 channels
mask2 = np.stack([mask, mask], axis=0)

print(values) # {0, 1, 2}
