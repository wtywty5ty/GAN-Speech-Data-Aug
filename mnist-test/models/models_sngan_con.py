import torch.nn as nn
import torch
from src.snlayers.snconv2d import SNConv2d
from src.snlayers.snlinear import SNLinear

class _netG(nn.Module):
    def __init__(self, nz, nc, ngf, nclass):
        super(_netG, self).__init__()
        self.ngf = ngf
        self.ini_w = 4
        self.ini_h = 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*4*self.ini_w*self.ini_h))
        self.l2 = nn.Sequential(nn.Linear(nclass, ngf*4*self.ini_w*self.ini_h))
        self.bn_ReLU = nn.Sequential(
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        self.main = nn.Sequential(
            # state size. (ngf*16) x 4 x 4
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
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input, label):
        x = self.l1(input)
        x = x.view(x.shape[0], self.ngf * 4, self.ini_w, self.ini_h)
        x = self.bn_ReLU(x)
        y = self.l2(label)
        y = y.view(y.shape[0], self.ngf * 4, self.ini_w, self.ini_h)
        y = self.bn_ReLU(y)
        out = torch.cat([x, y], 1)
        output = self.main(out)
        return output

class _netD(nn.Module):
    def __init__(self, nc, ndf, nclass):
        super(_netD, self).__init__()
        self.ini_w = 4
        self.ini_h = 4
        self.conv1 = nn.Sequential(
            # input is (nc) x 32 x 32
            SNConv2d(nc, ndf//2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf//2, ndf//2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.conv2 = nn.Sequential(
            # input is (nc) x 32 x 32
            SNConv2d(nclass, ndf//2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf//2, ndf//2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.main = nn.Sequential(
            # state size. (ndf) x 16 x 16
            SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf*2, ndf * 2, 4, 2, 1, bias=True),
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
        self.snlinear = nn.Sequential(SNLinear(ndf * 8 * self.ini_w * self.ini_h, 1),
                                      nn.Sigmoid())

    def forward(self, input, label):
        x = self.conv1(input)
        y = self.conv2(label)
        output = torch.cat([x, y], 1)
        output = self.main(output)
        output = output.view(output.size(0), -1)
        output = self.snlinear(output)
        return output.view(-1, 1).squeeze()