import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class make_Dense(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(make_Dense, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int((3 - 1) / 2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)

    def initialize_weight(self, w_mean, w_std):
            for m in self.modules():
                classname = m.__class__.__name__
                if classname.find('Conv') == 0:
                    nn.init.normal(m.weight.data, w_mean, w_std)

class RDB(nn.Module):
    def __init__(self, in_channels, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[make_Dense(in_channels + 64 * i, 64) for i in range(num_layers)])
        # local feature fusion
        self.lff = nn.Conv2d(in_channels + 64 * num_layers, 64, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

    def initialize_weight(self, w_mean, w_std):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') == 0:
                nn.init.normal(m.weight.data, w_mean, w_std)


class PSRNet(nn.Module):
    def __init__(self, scale_factor, in_channels, num_blocks, num_layers):
        super(PSRNet, self).__init__()
        self.D = num_blocks
        self.C = num_layers
        self.c_dim=in_channels
        # shallow feature extraction
        self.sfe1 = nn.Conv2d(self.c_dim, 64, kernel_size=3, padding=int((3 - 1) / 2))
        self.sfe2 = nn.Conv2d(64, 64, kernel_size=3, padding=int((3 - 1) / 2))

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(64,  self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(64, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(64 * self.D, 64, kernel_size=1),
            nn.Conv2d(64 , 64, kernel_size=3, padding=int((3 - 1) / 2))
        )
        #upsampling
        self.upsampling=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5,padding=int((5 - 1) / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=int((3 - 1) / 2)),
            nn.ReLU(inplace=True)
        )
        # upscale
        assert 2 <= scale_factor <= 4
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * (2 ** 2), kernel_size=3, padding=int((3 - 1) / 2)),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 64 * (2 ** 2), kernel_size=3, padding=int((3 - 1) / 2)),
            nn.PixelShuffle(2)
        )
        self.final=nn.Sequential(
            nn.Conv2d(64, self.c_dim, kernel_size=3,padding=int((3 - 1) / 2)),
        )
    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = torch.add(self.gff(torch.cat(local_features, 1)), sfe1)
        x = self.upsampling(x)
        x = self.upscale(x)
        x = self.final(x)
        return x

    def initialize_weight(self, w_mean=0, w_std=0.01):
            for m in self.modules():
                classname = m.__class__.__name__
                if classname.find('Conv') == 0:
                    nn.init.normal_(m.weight.data, w_mean, w_std)
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = PSRNet(scale_factor = 4,
              in_channels=4,
              num_blocks = 16,
              num_layers = 8
              ).to(device)
    net.to(device)
    # net.initialize_weight(0,0.01)
    summary(net, (4, 32, 32), 16, device='cuda')
    print(sum(x.numel() for x in net.parameters()))

