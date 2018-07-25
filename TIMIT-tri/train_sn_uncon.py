import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import random
import argparse
import pickle
import os, time
import subprocess
from utils.ProcessRawData import *
from utils.GenerateData import *
from utils.TestData import *
from models.models_uncon import _netG, _netD

plt.switch_backend('agg')


def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class CDCGAN_Classifier(object):
    def __init__(self, generator, discriminator, opt, device):
        self.opt = opt
        self.device = device
        # data
        # read data from TIMIT
        DIR = '/home/ty/tw472/triphone/temp.tri_Z/dnntrain'
        self.HTKcmd = '%s/hmm0/HNTrainSGD -B -C %s/basic.cfg -C %s/finetune_gpu%d.cfg -S %s/lib/flists/train.scp -l LABEL -I %s/train.mlf -H %s/hmm0/MMF -M %s/hmm0 %s/hmms.mlist' % (
            DIR, DIR, DIR, opt.gpu_id, DIR, DIR, DIR, DIR, DIR)

        # nets
        self.G = generator(opt.nz).to(device)
        self.G.apply(weight_filler)
        self.D = discriminator().to(device)
        self.D.apply(weight_filler)


        # criteria
        self.criteria_DG = nn.BCELoss().to(device)

        # solver
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0, 0.9))
        self.G_scheduler = optim.lr_scheduler.ExponentialLR(self.G_optimizer, gamma=0.99)
        self.D_scheduler = optim.lr_scheduler.ExponentialLR(self.D_optimizer, gamma=0.99)


    def train(self):
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []

        opt = self.opt
        device = self.device
        n_epochs = opt.n_epochs
        s = subprocess.Popen(self.HTKcmd, shell=True, stdout=subprocess.PIPE)
        print('training start!')
        start_time = time.time()
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            last = False
            iter = 0
            while not last:
                size, dataloader, last = processDataUni(s, opt.batchsize, 16)
                for data in dataloader:
                    iter += 1
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

                    fake = self.G(noise)
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
                    if iter % opt.n_dis == 0:
                        self.G.zero_grad()
                        label_Gfake = torch.ones_like(label_real)  # fake labels are real for generator cost
                        outputD = self.D(fake)
                        errG = self.criteria_DG(outputD, label_Gfake)

                        errG.backward()
                        self.G_optimizer.step()


                        train_hist['D_losses'].append(errD.item())
                        train_hist['G_losses'].append(errG.item())


                    if iter % 20 == 0:
                        print(
                            '[%d/%d][iter: %d] Loss_D: %.4f Loss_G: %.4f  D(x): %.4f D(G(z)): %.4f '
                            % (
                            epoch, n_epochs, iter, errD.item(), errG.item(), D_X, D_G))

            self.D_scheduler.step()
            self.G_scheduler.step()
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
            if (epoch + 1) % 100 == 0:
                torch.save(self.G, '%s/checkpoints/netG_epoch_%d.pkl' % (opt.outf, epoch))
                torch.save(self.D, '%s/checkpoints/netD_epoch_%d.pkl' % (opt.outf, epoch))


        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)
        print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
            torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), n_epochs, total_ptime))
        print("Training finish!... save training results")
        torch.save(self.G, '%s/netG_.pkl' % opt.outf)
        torch.save(self.D, '%s/netD_.pkl' % opt.outf)


        with open('%s/train_hist.pkl' % opt.outf, 'wb') as f:
            pickle.dump(train_hist, f)

        s.kill()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train DCGAN model')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0,1,2, 0,2.')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--n_dis', type=int, default=1, help='discriminator critic iters')
    parser.add_argument('--nz', type=int, default=100, help='dimention of lantent noise')
    parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
    parser.add_argument('--map_size', default=[16, 40], help='size of feature map')
    parser.add_argument('--phone', default='aa', help='phone')
    parser.add_argument('--outf', default='outf/test', help="path to output files)")
    opt = parser.parse_args()

    print(opt)

    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs('%s/checkpoints' % opt.outf, exist_ok=True)
    #os.makedirs('%s/images' % opt.outf, exist_ok=True)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if torch.cuda.is_available():
        device = torch.device('cuda:%d' % opt.gpu_id)
        torch.cuda.manual_seed_all(opt.manualSeed)
    else:
        device = torch.device('cpu')
    #cudnn.benchmark = True

    CDCGAN_Classifier(_netG, _netD, opt, device).train()
