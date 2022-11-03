from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET2D(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET2D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

if __name__ == "__main__":
    model=UNET2D(in_channels=1, out_channels=1)
    summary(model,(1,512,512))

def train(
    classifier: UNET2D,
    mini_batch: List[torch.Tensor],
    optimizer: nn.Module,
    criterion: nn.Module,
) -> Dict[str,float]:

    optimizer.zero_grad()

    if not classifier.training:
        classifier.train()
    
    loss, pred, labels, bias = iter(classifier, mini_batch, criterion)

    loss.backward()
    optimizer.step()

    #return get_metrics_cl(loss, pred, labels, bias)


def valid(
    classifier: UNET2D,
    mini_batch: List[torch.Tensor],
    criterion: nn.Module,
) -> Dict[str,float]:

    if classifier.training:
        classifier.eval()
        
    with torch.no_grad():
        loss, pred, labels, bias = iter(classifier, mini_batch, criterion)
        #return get_metrics_cl(loss, pred, labels, bias)