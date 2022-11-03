from typing import Dict, List, Tuple

import torch
from torch import nn
import torchvision.models as models


class Resnet(nn.Module):

    def __init__(self, classes: int, pretrained=True, fc_size: int = 512) -> None:
        super().__init__()
        self.normalize = pretrained

        self.encoder = models.resnet18(pretrained=pretrained)
        self.encoder.fc = nn.Sequential(
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, classes)
        )

    def forward(self, x: torch.Tensor, output_features: bool = False, layer: str = None) -> torch.Tensor:
        '''
            Forward the input through a resnet so we can get a single
            convolution layer.

            args:
                x (tensor): the input of the encoder
                output_features (bool): whether it should output a feature embedding.
        '''
        if output_features and layer is None:
            raise Exception('If output_features is True, you should inform a layer')

        if layer is not None:
            assert layer in ['preconv', 'conv1', 'conv2', 'conv3', 'conv4', 'all']

        img = x

        conv0 = self.encoder.conv1(img)
        x = self.encoder.bn1(conv0)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        conv1 = self.encoder.layer1(x)      # [b, 256, 56, 56]
        conv2 = self.encoder.layer2(conv1)  # [b, 512, 28, 28]
        conv3 = self.encoder.layer3(conv2)  # [b, 1024, 14, 14]
        conv4 = self.encoder.layer4(conv3)  # [b, 2048, 7, 7]

        feature_embedding = {
            'preconv': x,
            'conv1': conv1,
            'conv2': conv2,
            'conv3': conv3,
            'conv4': conv4,
            'all': [conv4, conv3, conv2, conv1, conv0, img]
        }
        features = feature_embedding[layer] if output_features else None

        x = self.encoder.avgpool(conv4)
        x = torch.squeeze(x)
        x = self.encoder.fc(x)

        #TODO do i want features?
        return features, x