import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import random
import argparse
import pickle
import os,time
from models.models_sngan_con import _netG, _netD

parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_ids', default=[0, 1, 2, 3], help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_dis', type=int, default=2, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=100, help='dimention of lantent noise')
parser.add_argument('--nclass', type=int, default=10, help='number of classes')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--map_size', default=[32, 32], help='size of feature map')
parser.add_argument('--outf', default='outf/sn-con-critic2', help="path to output files)")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.outf, exist_ok=True)
os.makedirs('%s/checkpoints'%opt.outf, exist_ok=True)
os.makedirs('%s/images'%opt.outf, exist_ok=True)

# dataset = datasets.ImageFolder(root='/home/chao/zero/datasets/cfp-dataset/Data/Images',
#                            transform=transforms.Compose([
#                                transforms.Scale(32),
#                                transforms.CenterCrop(32),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ])
#                                       )

dataset = datasets.MNIST(root='../data/mnist', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.map_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,
                                         shuffle=True)

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

n_dis = opt.n_dis
nz = opt.nz

G = _netG(nz, 1, 64, opt.nclass)
SND = _netD(1, 64, opt.nclass)
print(G)
print(SND)
G.apply(weight_filler)
SND.apply(weight_filler)

input = torch.FloatTensor(opt.batchsize, 1, 32, 32)
noise = torch.FloatTensor(opt.batchsize, nz)
label = torch.FloatTensor(opt.batchsize)
real_label = 1
fake_label = 0

# prepare fixed noise & label
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

criterion = nn.BCELoss()

# label preprocess
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1)
fill = torch.zeros([10, 10, opt.map_size[0], opt.map_size[1]])
for i in range(10):
    fill[i, i, :, :] = 1

if opt.cuda:
    G.cuda()
    SND.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()
    fixed_z_, fixed_y_label_ = fixed_z_.cuda(), fixed_y_label_.cuda()
    onehot, fill = onehot.cuda(), fill.cuda()

optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0, 0.9))
optimizerSND = optim.Adam(SND.parameters(), lr=0.0002, betas=(0, 0.9))

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

n_epochs = opt.n_epochs
print('training start!')
start_time = time.time()
for epoch in range(n_epochs):
    epoch_start_time = time.time()
    for i, data in enumerate(dataloader, 0):
        step = epoch * len(dataloader) + i
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
        y_ = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).squeeze()
        y_fake_ = Variable(onehot[y_])
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
        if step % n_dis == 0 or step == 1:
            G.zero_grad()
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output = SND(fake, y_fake_fill_)
            errG = criterion(output, labelv)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
        if i % 20 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, n_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/images/real_samples.png' % opt.outf,
                    normalize=True)
            fake = G(fixed_z_, fixed_y_label_)
            vutils.save_image(fake.data,
                    '%s/images/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    nrow=10, normalize=True)

        train_hist['D_losses'].append(errD.item())
        train_hist['G_losses'].append(errG.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f'%((epoch+1), n_epochs, per_epoch_ptime))
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
    # do checkpointing
    if epoch %30  ==0:
        torch.save(G, '%s/checkpoints/netG_epoch_%d.pkl' % (opt.outf, epoch))
        torch.save(SND, '%s/checkpoints/netD_epoch_%d.pkl' % (opt.outf, epoch))

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), n_epochs, total_ptime))
print("Training finish!... save training results")
torch.save(G, '%s/netG_.pkl' % opt.outf)
torch.save(SND, '%s/netD_.pkl' % opt.outf)

with open('%s/train_hist.pkl' % opt.outf, 'wb') as f:
    pickle.dump(train_hist, f)


























