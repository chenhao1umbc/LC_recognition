import torch
from torch import nn
from utils import KLD_gauss, KLD_laplace


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        mid_ch = max(in_ch//2, 1)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class VAE(nn.Module):
    def __init__(self, dimx=8, dimz=32, n_sources=2, variational=True):
        super(VAE, self).__init__()

        self.dimx = dimx
        self.dimz = dimz
        self.n_sources = n_sources
        self.variational = variational
        down_size = 8

        chans = (64, 64, 64, 64, 64)

        self.out_z = nn.Linear(4096, 2*self.n_sources*self.dimz)

        self.Encoder = nn.Sequential(
            DoubleConv(self.dimx, chans[0]),
            Down(chans[0],chans[1]),
            DoubleConv(chans[1],chans[2]),
            Down(chans[2],chans[3]),
            DoubleConv(chans[3],chans[4]),
            )

        self.Decoder = nn.Sequential(
            nn.Linear(self.dimz, down_size*down_size),
            Reshape(-1, 1, down_size, down_size),
            DoubleConv(1, chans[4]),
            Up(chans[4],chans[3]),
            DoubleConv(chans[3],chans[2]),
            Up(chans[2],chans[1]),
            DoubleConv(chans[1],chans[0]),
            nn.Conv2d(chans[0], self.dimx, kernel_size=1),
            )

    def encode(self, x):
        d = self.Encoder(x)
        dz = self.out_z(d.view(d.shape[0], -1))
        mu = dz[:,::2]
        logvar = dz[:,1::2]
        return mu,logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        d = self.Decoder(z.view(-1,self.dimz))
        recon_separate = d.reshape(-1, self.n_sources, self.dimx, 32, 32)
        recon_x = recon_separate.sum(1) 
        return recon_x, recon_separate

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.variational is True:
            z = self.reparameterize(mu, logvar)
            recon_x, recon_separate = self.decode(z)
        else:
            recon_x, recon_separate = self.decode(mu)
        return recon_x, mu, logvar, recon_separate
    
class VAE2(nn.Module):
    def __init__(self, dimx=8, dimz=32, n_sources=2, variational=True):
        super().__init__()

        self.dimx = dimx
        self.dimz = dimz
        self.n_sources = n_sources
        self.variational = variational
        down_size = 8

        chans = (64, 128, 128, 128, 64)

        self.out_z = nn.Linear(4096, 2*self.n_sources*self.dimz)

        self.Encoder = nn.Sequential(
            DoubleConv(self.dimx, chans[0]),
            Down(chans[0],chans[1]),
            DoubleConv(chans[1],chans[2]),
            Down(chans[2],chans[3]),
            DoubleConv(chans[3],chans[4]),
            )

        self.Decoder = nn.Sequential(
            nn.Linear(self.dimz, down_size*down_size),
            Reshape(-1, 1, down_size, down_size),
            DoubleConv(1, chans[4]),
            Up(chans[4],chans[3]),
            DoubleConv(chans[3],chans[2]),
            Up(chans[2],chans[1]),
            DoubleConv(chans[1],chans[0]),
            nn.Conv2d(chans[0], self.dimx, kernel_size=1),
            )

    def encode(self, x):
        d = self.Encoder(x)
        dz = self.out_z(d.view(d.shape[0], -1))
        mu = dz[:,::2]
        logvar = dz[:,1::2]
        return mu,logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        d = self.Decoder(z.view(-1,self.dimz))
        recon_separate = d.reshape(-1, self.n_sources, self.dimx, 32, 32)
        recon_x = recon_separate.sum(1) 
        return recon_x, recon_separate

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.variational is True:
            z = self.reparameterize(mu, logvar)
            recon_x, recon_separate = self.decode(z)
        else:
            recon_x, recon_separate = self.decode(mu)
        return recon_x, mu, logvar, recon_separate



class LaplaceLoss(nn.Module):
    def __init__(self, variance=None,reduction='sum'):
        super(LaplaceLoss, self).__init__()
        if variance is None:
            variance = torch.tensor(1.0)
        self.register_buffer("log2",torch.tensor(2.0).log())
        self.register_buffer("scale",torch.sqrt(0.5*variance))
        self.logscale = self.scale.log()

    def forward(self, estimate, target):
        return torch.sum((target-estimate).abs() / self.scale)


class Loss(nn.Module):
    def __init__(self, sources=2, alpha=None, likelihood='bernoulli',variational=True,prior='gauss',scale=1.0):
        super(Loss, self).__init__()
        self.variational = variational
        self.prior = prior
        self.scale = scale

        if likelihood == 'gauss':
            self.criterion = nn.MSELoss(reduction='sum')
        elif likelihood == 'laplace':
            self.criterion = LaplaceLoss()
        else:
            self.criterion = nn.BCELoss(reduction='sum')

        if alpha is None:
            self.alpha_pior = nn.Parameter(torch.ones(1,sources),requires_grad=False)
        else:
            self.alpha_prior = nn.Parameter(alpha,requires_grad=False)

    def forward(self, x, recon_x, mu, logvar, beta=1):
        ELL = self.criterion(recon_x, x)

        KLD = 0.0
        if self.variational is True:
            if self.prior == 'laplace':
                KLD = KLD_laplace(mu,logvar,scale=self.scale)
            else:
                KLD = KLD_gauss(mu,logvar)
        
        loss = ELL + beta*KLD
        return loss, ELL,  KLD

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 2)        
        )

    def forward(self, x):
        return self.mlp(x)