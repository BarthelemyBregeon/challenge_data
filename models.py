import torch
import torch.nn as nn
import torch.nn.functional as F

class basic_FCN(nn.Module):
    def __init__(self, input_channels):
        super(basic_FCN, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # bottleneck
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # decoder
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(32, 3, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x

# U-net from scratch
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling Path)
        for feature in features:
            self.encoder.append(self._double_conv(in_channels, feature))
            in_channels = feature  # Update for next layer

        # Bottleneck
        self.bottleneck = self._double_conv(features[-1], features[-1] * 2)

        # Decoder (Upsampling Path)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._double_conv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _double_conv(self, in_channels, out_channels):
        """Helper function to create a double convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        # Encoder Pass
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder Pass (with skip connections)
        skip_connections = skip_connections[::-1]  # Reverse order for decoding

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Transposed Conv
            skip_connection = skip_connections[i // 2]

            # If needed, crop the skip connection
            if x.shape != skip_connection.shape:
                skip_connection = F.interpolate(skip_connection, size=x.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat((skip_connection, x), dim=1)  # Concatenate
            x = self.decoder[i + 1](x)  # Double Convolution

        return self.final_conv(x)