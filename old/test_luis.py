from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    ScaleIntensityRange,
    Flip,
    LabelToMask,
)

train_transforms = Compose(
    [
        EnsureChannelFirst(),
        #ARRUMAR MÁSCARA DOS TUMORES E FÍGADO PARA SER UMA SÓ USANDO label to mask
        #Flipd(keys=['mask'], spatial_axis=1),
    ]
    )

#ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)

print(train_transforms)