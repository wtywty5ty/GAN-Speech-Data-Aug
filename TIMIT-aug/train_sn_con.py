import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import pickle
import os,time
import subprocess
from utils import processTriData, triphoneMap  
from models.models_w16d40 import _netG, _netD
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_ids', default=[0, 1, 2, 3], help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--phone', default='m', help='phone')
parser.add_argument('--n_dis', type=int, default=2, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=100, help='dimension of latent noise')
parser.add_argument('--nclass', type=int, default=3, help='number of classes')
parser.add_argument('--batchsize', type=int, default=100, help='training batch size')
parser.add_argument('--map_size', default=[16, 40], help='size of feature map')
parser.add_argument('--outf', default='outf/log-m-divide4', help="path to output files)")
opt = parser.parse_args()

phoneMap = triphoneMap('slist.txt', opt.phone)
opt.nclass = phoneMap.nlabels()
print(opt)



os.makedirs(opt.outf, exist_ok=True)
os.makedirs('%s/checkpoints' % opt.outf, exist_ok=True)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.cuda.set_device(opt.gpu_ids[0])

cudnn.benchmark = True


def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# initialize netG and netD
n_dis = opt.n_dis
nz = opt.nz
G = _netG(nz, 1, 64, opt.nclass)
SND = _netD(1, 64, opt.nclass)
print(G)
print(SND)
G.apply(weight_filler)
SND.apply(weight_filler)

input = torch.FloatTensor(opt.batchsize, 1, 16, 40)
noise = torch.FloatTensor(opt.batchsize, nz)
label = torch.FloatTensor(opt.batchsize)
real_label = 1
fake_label = 0


criterion = nn.BCELoss()

# label preprocess
onehot = torch.zeros(opt.nclass, opt.nclass)
v = []
for i in range(opt.nclass):
    v.append(i)
onehot = onehot.scatter_(1, torch.LongTensor(v).view(opt.nclass,1), 1)
fill = torch.zeros([opt.nclass, opt.nclass, opt.map_size[0], opt.map_size[1]])
for i in range(opt.nclass):
    fill[i, i, :, :] = 1

if opt.cuda:
    G.cuda()
    SND.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()
    onehot, fill = onehot.cuda(), fill.cuda()

optimizerG = optim.Adam(G.parameters(), lr=opt.lr, betas=(0, 0.9))
optimizerSND = optim.Adam(SND.parameters(), lr=opt.lr, betas=(0, 0.9))
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['D_losses_epoch'] = []
train_hist['G_losses_epoch'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

n_epochs = opt.n_epochs
print('training start!')
start_time = time.time()

# read data from TIMIT
DIR = '/home/ty/tw472/triphone/temp.tri_Z/dnntrain'
HTKcmd = '%s/hmm0/HNTrainSGD -B -C %s/basic.cfg -C %s/finetune.cfg -S %s/lib/flists/train.scp -l LABEL -I %s/train.mlf -H %s/hmm0/MMF -M %s/hmm0 %s/hmms.mlist'%(DIR,DIR,DIR,DIR,DIR,DIR,DIR,DIR)
s = subprocess.Popen(HTKcmd, shell=True, stdout=subprocess.PIPE)
epoch = 1
step = 0

while True:
    size, dataloader, last = processTriData(s, opt.batchsize, phoneMap, 16)
    for data in dataloader:
        step += 1
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        SND.zero_grad()
        real_cpu, real_y = data
        batch_size = real_cpu.size(0)

        input.resize_(real_cpu.size()).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        y_fill_ = Variable(fill[real_y])
        output = SND(inputv, y_fill_)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, noise.size(1)).normal_(0, 1)
        noisev = Variable(noise)
        y_ = (torch.rand(batch_size, 1) * opt.nclass).type(torch.LongTensor).squeeze()
        y_fake_ = onehot[y_]
        y_fake_fill_ = fill[y_]
        y_fake_, y_fake_fill_ = Variable(y_fake_), Variable(y_fake_fill_)

        fake = G(noisev, y_fake_)
        labelv = Variable(label.fill_(fake_label))
        output = SND(fake.detach(), y_fake_fill_)
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake

        optimizerSND.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if step % n_dis == 0 or step ==1:
            G.zero_grad()
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output = SND(fake, y_fake_fill_)
            errG = criterion(output, labelv)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
        if step % 20 == 0:
            print('[%d/%d][step: %d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, n_epochs, step,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        train_hist['D_losses'].append(errD.item())
        train_hist['G_losses'].append(errG.item())


    if last:
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - start_time
        print('[%d/%d] - time cost: %.2f'%(epoch, n_epochs, per_epoch_ptime))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        d_loss_plot = train_hist['D_losses']
        train_hist['D_losses_epoch'].append(np.mean(d_loss_plot[epoch*step - step:epoch*step]))
        g_loss_plot = train_hist['G_losses']
        train_hist['G_losses_epoch'].append(np.mean(g_loss_plot[epoch*step - step:epoch*step]))

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

        plt.figure()
        x = np.linspace(1, epoch, epoch)
        plt.xticks(x)
        plt.plot(x, train_hist['D_losses_epoch'])
        plt.xlabel('Epochs')
        plt.ylabel('Discriminator\'s loss')
        plt.savefig('%s/d_loss_epoch.png' % opt.outf)

        plt.figure()
        plt.xticks(x)
        plt.plot(x, train_hist['G_losses_epoch'])
        plt.xlabel('Epochs')
        plt.ylabel('Generator\'s loss')
        plt.savefig('%s/g_loss_epoch.png' % opt.outf)
        plt.close('all')
        # do checkpointing
        if epoch % 4 == 0:
            torch.save(G, '%s/checkpoints/netG_epoch_%d.pkl' % (opt.outf, epoch))
            torch.save(SND, '%s/checkpoints/netD_epoch_%d.pkl' % (opt.outf, epoch))


        epoch += 1
        step = 0


    if epoch > n_epochs:
        break

s.kill()
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (total_ptime/n_epochs, n_epochs, total_ptime))
print("Training finish!... save training results")
torch.save(G, '%s/netG_.pkl' % opt.outf)
torch.save(SND, '%s/netD_.pkl' % opt.outf)

with open('%s/train_hist.pkl' % opt.outf, 'wb') as f:
    pickle.dump(train_hist, f)


























