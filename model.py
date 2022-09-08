import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary

class FirstDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, slices):
        super(FirstDoubleConv, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            #nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(out_channels),
            #nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv2(x)

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

class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], slices=1
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        count=0
        for feature in features:
            if count==0: 
                self.downs.append(FirstDoubleConv(in_channels, feature, slices))
                count+=1
                in_channels = feature
            else:
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
        #print(x.shape)
        skip_connections = []

        for down in self.downs:
            #print(x.shape)
            x = down(x)
            count=0
            if count==0:
                #print("x",x.shape)
                x = torch.squeeze(x,2)
                #print("squeeze", x.shape)
                count+=1
            #print(x.shape)
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

# def test():
#     x = torch.randn((3, 1, 161, 161))
#     model = UNET(in_channels=1, out_channels=1, slices=1)
#     preds = model(x)
#     assert preds.shape == x.shape

if __name__ == "__main__":
    model=UNET(in_channels=1, out_channels=1, slices=1)
    #print(model)
    summary(model, (1, 3, 512, 512))
