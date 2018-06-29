import torch.nn as nn
import torch


class _netG(nn.Module):
    def __init__(self, nz, nclass):
        super(_netG, self).__init__()
        ngf = 64
        self.ngf = ngf
        self.ln = nn.Sequential(nn.Linear(nz+nclass, ngf*8*4*4),
                                nn.BatchNorm1d(ngf*8*4*4),
                                nn.ReLU(True),)

        self.main = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 1, 3, 1, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        ln = self.ln(input)
        l1= ln.view(ln.shape[0], self.ngf*8, 4, 4)
        output = self.main(l1)
        return output


class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        ndf = 64
        self.main = nn.Sequential(
            # state size. 1 x 32 x 32
            nn.Conv2d(1, ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf , ndf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.ln = nn.Sequential(nn.Linear(ndf*8*4*4, 1),
                                nn.Sigmoid())

    def forward(self, input):
        out = self.main(input)
        out = out.view(out.shape[0], -1)
        out = self.ln(out)
        return out.squeeze()


class _netC(nn.Module):
    def __init__(self):
        super(_netC, self).__init__()
        ncf = 64
        self.main = nn.Sequential(
            # state size. 1 x 32 x 32
            nn.Conv2d(1, ncf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ncf , ncf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ncf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ncf * 2, ncf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ncf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ncf * 4, ncf * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ncf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.ln = nn.Sequential(nn.Linear(ncf*8*4*4, 10))

    def forward(self, input):
        out = self.main(input)
        out = out.view(out.shape[0], -1)
        out = self.ln(out)
        return out.squeeze()
