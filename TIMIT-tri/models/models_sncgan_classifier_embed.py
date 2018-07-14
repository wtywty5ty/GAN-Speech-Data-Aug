import torch.nn as nn
from src.snlayers.snconv2d import SNConv2d
from src.snlayers.snlinear import SNLinear
import torch


class _netG(nn.Module):
    def __init__(self, nz, nclass, dim):
        super(_netG, self).__init__()
        ngf = 64
        self.ngf = ngf
        self.emb = nn.Embedding(nclass,dim, max_norm=1)
        self.ln = nn.Sequential(nn.Linear(nz+dim, 1024),
                                nn.BatchNorm1d(1024),
                                nn.ReLU(True),
                                nn.Linear(1024, ngf*8*2*5),
                                nn.BatchNorm1d(ngf*8*2*5),
                                nn.ReLU(True),
                                )

        self.main = nn.Sequential(
            # state size. (ngf*8) x initial size
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x initial size x 2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x initial size x 4
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x initial size x 8
            nn.ConvTranspose2d(ngf, 1, 3, 1, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x initial size x 8
        )

    def forward(self, z, y):
        embd = self.emb(y).squeeze()
        input = torch.cat([z, embd], 1)
        ln = self.ln(input)
        l1= ln.view(ln.shape[0], self.ngf*8, 2, 5)
        output = self.main(l1)
        return output


class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        ndf = 64
        self.main = nn.Sequential(
            # state size. 1 x 32 x 32
            SNConv2d(1, ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 16 x 16
            SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 8 x 8
            SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*4) x 4 x 4
            SNConv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.ln = nn.Sequential(SNLinear(ndf*8*2*5, 1024),
                                nn.LeakyReLU(0.2, inplace=True),
                                SNLinear(1024, 1),
                                nn.Sigmoid())

    def forward(self, input):
        out = self.main(input)
        out = out.view(out.shape[0], -1)
        out = self.ln(out)
        return out.squeeze()


class _netC(nn.Module):
    def __init__(self, nclass):
        super(_netC, self).__init__()
        self.main = nn.Sequential(nn.Linear(16*40, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(True),
                                  nn.Linear(500, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(True),
                                  nn.Linear(500, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(True),
                                  nn.Linear(500, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(True),
                                  nn.Linear(500, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(True),
                                  nn.Linear(500, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(True),
                                  nn.Linear(500, nclass),
                                  )


    def forward(self, input):
        input = input.view(input.shape[0], -1)
        out = self.main(input)
        return out.squeeze()


