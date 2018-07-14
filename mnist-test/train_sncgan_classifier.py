import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.utils.data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import random
import argparse
import pickle
import os,time
from models.models_sn_cgan_classifier import _netG, _netD, _netC


def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def concat(z,y):
    return torch.cat([z, y], 1)


def sample_yz(device):
    temp_z_ = torch.randn(10, 100)
    fixed_z_ = temp_z_
    fixed_y_ = torch.zeros(10, 1)
    for i in range(9):
        fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
        temp = torch.ones(10, 1) + i
        fixed_y_ = torch.cat([fixed_y_, temp], 0)

    fixed_z_ = fixed_z_.view(-1, 100)
    fixed_y_label_ = torch.zeros(100, 10)
    fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
    fixed_y_label_ = fixed_y_label_.view(-1, 10)

    return fixed_z_.to(device), fixed_y_label_.to(device)


class CDCGAN_Classifier(object):
    def __init__(self, generator, discriminator, classifier, opt, device):
        self.opt = opt
        self.device =device
        # data
        dataset = datasets.MNIST(root='../data/mnist', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.map_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize, shuffle=True)

        onehot = torch.zeros(10, 10)
        self.onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).to(device)
        self.fixed_z, self.fixed_y = sample_yz(device)

        # nets
        self.G = generator(opt.nz, opt.nclass).to(device)
        self.G.apply(weight_filler)
        self.D = discriminator().to(device)
        self.D.apply(weight_filler)
        self.C = classifier(opt.nclass).to(device)
        self.C.apply(weight_filler)

        # criteria
        self.criteria_DG = nn.BCELoss()
        self.criteria_C = nn.CrossEntropyLoss()

        # solver
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0, 0.9))
        self.C_optimizer = optim.Adam(self.C.parameters(), lr=0.0002, betas=(0, 0.9))
        self.G_scheduler = optim.lr_scheduler.ExponentialLR(self.G_optimizer, gamma=0.99)
        self.D_scheduler = optim.lr_scheduler.ExponentialLR(self.D_optimizer, gamma=0.99)
        self.C_scheduler = optim.lr_scheduler.ExponentialLR(self.C_optimizer, gamma=0.99)

    def train(self):
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []

        opt = self.opt
        device = self.device
        n_epochs = opt.n_epochs
        print('training start!')
        start_time = time.time()
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            for i, data in enumerate(self.dataloader, 0):
                step = epoch * len(self.dataloader) + i
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                self.D.zero_grad()
                real_image, real_y = data
                real_image, real_y = real_image.to(device), real_y.to(device)
                batch_size = real_image.size(0)

                label_real = torch.ones(batch_size, dtype=torch.float, device=device)

                outputD_real = self.D(real_image)
                errD_real = self.criteria_DG(outputD_real, label_real)
                errD_real.backward()
                D_X = outputD_real.data.mean()

                # train with fake
                noise = real_image.new_zeros(batch_size, opt.nz).normal_(0, 1)
                y_ = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).squeeze().to(device)
                y_fake_ = self.onehot[y_]

                fake = self.G(concat(noise, y_fake_))
                label_fake = torch.zeros_like(label_real)
                outputD_fake = self.D(fake.detach())
                errD_fake = self.criteria_DG(outputD_fake, label_fake)
                errD_fake.backward()
                D_G = outputD_fake.data.mean()
                errD = errD_real + errD_fake
                self.D_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))+ log(C(G(z)))
                ###########################
                if step % opt.n_dis == 0:
                    self.G.zero_grad()
                    label_Gfake = torch.ones_like(label_real)  # fake labels are real for generator cost
                    outputD = self.D(fake)
                    errG_D = self.criteria_DG(outputD, label_Gfake)
                    outputC = self.C(fake)
                    errG_C = self.criteria_C(outputC, y_)
                    errG = 0.01*errG_C + errG_D
                    errG.backward()
                    self.G_optimizer.step()

                ############################
                # (2) Update C network: maximize log(C(x))
                ###########################
                    self.C.zero_grad()
                    output = self.C(real_image)
                    errC = self.criteria_C(output, real_y)
                    errC.backward()
                    self.C_optimizer.step()

                    train_hist['D_losses'].append(errD.item())
                    train_hist['G_losses'].append(errG.item())

                if i % 20 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f (%.4f / %.4f) Loss_C: %.4f D(x): %.4f D(G(z)): %.4f '
                          % (epoch, n_epochs, i, len(self.dataloader),
                             errD.item(), errG.item(), errG_D, errG_C, errC.item(), D_X, D_G))
                if i % 100 == 0:
                    vutils.save_image(real_image,
                                      '%s/images/real_samples.png' % opt.outf,
                                      normalize=True)
                    fake = self.G(concat(self.fixed_z, self.fixed_y))
                    vutils.save_image(fake.data,
                                      '%s/images/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                                      nrow=10, normalize=True)

            self.D_scheduler.step()
            self.G_scheduler.step()
            self.C_scheduler.step()
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print('[%d/%d] - ptime: %.2f' % ((epoch + 1), n_epochs, per_epoch_ptime))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

            plt.figure()
            plt.plot(train_hist['D_losses'])
            plt.xlabel('Iterations')
            plt.ylabel('Discriminator\'s loss')
            plt.savefig('%s/d_loss.png' % opt.outf)

            plt.figure()
            plt.plot(train_hist['G_losses'])
            plt.xlabel('Iterations')
            plt.ylabel('Generator\'s loss')
            plt.savefig('%s/g_loss.png' % opt.outf)
            plt.close('all')
            # do checkpointing
            if epoch % 2 == 0:
                torch.save(self.G, '%s/checkpoints/netG_epoch_%d.pkl' % (opt.outf, epoch))
                torch.save(self.D, '%s/checkpoints/netD_epoch_%d.pkl' % (opt.outf, epoch))
                torch.save(self.C, '%s/checkpoints/netC_epoch_%d.pkl' % (opt.outf, epoch))

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)
        print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
        torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), n_epochs, total_ptime))
        print("Training finish!... save training results")
        torch.save(self.G, '%s/netG_.pkl' % opt.outf)
        torch.save(self.D, '%s/netD_.pkl' % opt.outf)
        torch.save(self.C, '%s/netC_.pkl' % opt.outf)

        with open('%s/train_hist.pkl' % opt.outf, 'wb') as f:
            pickle.dump(train_hist, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train DCGAN model')
    parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs of training')
    parser.add_argument('--gpu_ids', default=[0, 1, 2, 3], help='gpu ids: e.g. 0,1,2, 0,2.')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--n_dis', type=int, default=1, help='discriminator critic iters')
    parser.add_argument('--nz', type=int, default=100, help='dimention of lantent noise')
    parser.add_argument('--nclass', type=int, default=10, help='number of classes')
    parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
    parser.add_argument('--map_size', default=[32, 32], help='size of feature map')
    parser.add_argument('--outf', default='outf/sncgan_lowc', help="path to output files)")
    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs('%s/checkpoints' % opt.outf, exist_ok=True)
    os.makedirs('%s/images' % opt.outf, exist_ok=True)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.manual_seed_all(opt.manualSeed)
    else:
        device = torch.device('cpu')
    cudnn.benchmark = True

    CDCGAN_Classifier(_netG, _netD, _netC, opt, device).train()
