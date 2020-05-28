from torch import nn


class encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_features,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.pool = nn.MaxPool2d(2, 0)
        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = out_features,
            kernel_size = 3,
            padding = 1
        )

    def forward(self, x):
        return self.conv3(self.pool(self.conv2(self.conv1(x))))

class decoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(decoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_features,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.pool = nn.Upsample(scale_factor = 2)
        self.conv2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )
        self.conv3 = nn.Conv2d(
            in_channels = 32,
            out_channels = out_features,
            kernel_size = 3,
            padding = 1
        )

    def forward(self, x):
        return self.conv3(self.conv2(self.pool(self.conv1(x))))


class autoencoder(nn.Module):
    def __init__(self, in_feature, bottleneck):
        super(autoencoder, self).__init__()
        self.encoder = encoder(in_feature, bottleneck)
        self.decoder = decoder(bottleneck, in_feature)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded), encoded
