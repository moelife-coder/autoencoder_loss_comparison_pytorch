from torch import nn


class coder(nn.Module):
    def __init__(self, in_features, out_features):
        super(coder, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear1(x)


class autoencoder(nn.Module):
    def __init__(self, in_feature, bottleneck):
        super(autoencoder, self).__init__()
        self.encoder = coder(in_feature, bottleneck)
        self.decoder = coder(bottleneck, in_feature)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded), encoded
