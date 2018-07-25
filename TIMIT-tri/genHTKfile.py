import torch
import struct
import subprocess
import numpy
import os, signal
from utils.ProcessRawData import *
from utils.GenerateData import *
from utils.TestData import *



class genHTKfile(object):
    def __init__(self, phone, ID):
        #self.mode = 'prior'  # almost fixed file size
        self.mode = 'uniform_rej'
        # 'prior'
        self.sampPeriod = 100000
        self.sampSize = 2080
        self.parmKind = 9 + 0o004000
        self.phone = phone
        self.ID = ID
        G_PATH = 'outf/GAN_array/%s/netG_.pkl'%phone
        generator = torch.load(G_PATH, map_location=lambda storage, loc: storage).cuda().eval()
        self.generator = generator
        self.phoneMap = triphoneMap('slist.txt', phone)
        if self.mode.split('_')[0] == 'uniform':
            self.nSamples = 6000 - 6000 % self.phoneMap.nlabels()
            self.splitSize = self.nSamples//self.phoneMap.nlabels()
        """
        elif self.mode.split('_')[0] == 'prior':
            self.splitSize = {}
            self.nSamples = 0
            prior = {}
            with open('prior.txt', 'r') as f:
                for line in f.readlines():
                    state = int(line.split(' ')[0])
                    pr = float(line.split(' ')[1].strip('\n'))
                    prior[state] = pr
            totalsamp = 21600 # 18s x 48
            for fid in range(self.phoneMap.nlabels()):
                self.splitSize[fid] = int(totalsamp*prior[self.phoneMap.f2t[fid]])
                self.nSamples += self.splitSize[fid]
        """


    def genSamples(self):
        phoneMap = self.phoneMap
        nclass = phoneMap.nlabels()
        splitSizeSet = self.splitSize
        DIR = '/home/ty/tw472/triphone/FH7/dnntrain'
        HTKcmd = '%s/HNForward -C %s/basic.cfg -C %s/eval.cfg -H %s/hmm0/MMF %s/hmms.mlist' % (DIR, DIR, DIR, DIR, DIR)
        s = subprocess.Popen(HTKcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        binary = struct.pack('>iihh', self.nSamples, self.sampPeriod, self.sampSize, self.parmKind)
        print('Start generating %s samples:'%self.phone)
        for id in range(nclass):
            if self.mode.split('_')[0] == 'prior':
                splitSize = splitSizeSet[id]
            else:
                splitSize = splitSizeSet

            if self.mode.split('_')[1] != 'rej':
                buf, data = generateData(self.generator, splitSize, nclass, id)
                s.stdin.write(buf)
                s.stdin.flush()
                tid = phoneMap.f2t[id]
                index = dataFilter(s, tid)
                data = data[:, :13, :]
                data_f = data[index]

                while data_f.size(0) < splitSize:
                    buf, data = generateData(self.generator, splitSize, nclass, id)
                    s.stdin.write(buf)
                    s.stdin.flush()
                    tid = phoneMap.f2t[id]
                    index = dataFilter(s, tid)
                    data = data[:, :13, :].detach()
                    data_f_ = data[index]
                    data_f = torch.cat([data_f, data_f_], 0)

                data_f = data_f[:splitSize]

                flat_data = data_f.cpu().view(-1).detach().numpy()
                body_ = flat_data.astype('>f').tostring()
                binary = binary + body_

            else:
                noise = torch.randn(splitSize, 100).cuda()
                y = torch.zeros(splitSize, nclass).cuda()
                y[:, id] = 1
                gen_data = self.generator(torch.cat([noise, y], 1)).squeeze()
                gen_data = gen_data[:, :13, :]
                gen_data = 3 * gen_data
                samples = gen_data.cpu().view(-1).detach().numpy()
                body_ = samples.astype('>f').tostring()
                binary = binary + body_

            print('..'+str(id), end="")
        print('..finish!')
        end = struct.pack('i', 0)
        s.stdin.write(end)
        s.kill()
        return binary

    def genfbk(self):
        binary = self.genSamples()
        with open('HTKFILE/fbk/%s/%s_gan_%d_%s.fbk' % (self.mode, self.phone, self.ID, self.mode), 'wb') as f:
            f.write(binary)
        print('Generating fbk file successfully')

    def appendscp(self):
        with open('HTKFILE/flists/gan_%s.scp'%self.mode, 'a') as f:
            f.write('%s_gan_%d_%s.fbk' % (self.phone, self.ID, self.mode)
                    + '=/home/ty/tw472/master/FH5_w16d40_tri/HTKFILE/fbk/%s/%s_gan_%d_%s.fbk' % (self.mode, self.phone, self.ID, self.mode)
                    + '[0,%d]\n'%(self.nSamples-1))
        print('Appending scp file successfully')

    def appendmlf(self):
        smap = statemap('states.map')
        splitSizeSet = self.splitSize
        if not os.path.exists('HTKFILE/mlabs/gan_%s.mlf'%self.mode):
            with open('HTKFILE/mlabs/gan_%s.mlf'%self.mode, 'w') as f:
                f.write('#!MLF!#\n')
        with open('HTKFILE/mlabs/gan_%s.mlf'%self.mode, 'a') as f:
            f.write('"%s_gan_%d_%s.lab"\n'% (self.phone, self.ID, self.mode))
            for id in range(self.phoneMap.nlabels()):
                vstate = smap[self.phoneMap.f2states[id]]
                if self.mode.split('_')[0] == 'prior':
                    splitSize = splitSizeSet[id]
                else:
                    splitSize = splitSizeSet
                if id == 0:
                    start =0
                    end = start + splitSize*self.sampPeriod
                else:
                    start = end
                    end = start + splitSize*self.sampPeriod

                f.write('%d %d %s\n'% (start, end, vstate))

            f.write('.\n')
            print('Appending mlf file successfully')


if __name__ == '__main__':
    phone_list1 = ['b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't']
    phone_list2 = ['v', 'w', 'y', 'z', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'ch']
    phone_list3 = ['cl', 'dh', 'dx', 'eh', 'el', 'en', 'er', 'ey', 'hh', 'ih', 'ix', 'iy']
    phone_list4 = ['jh', 'ng', 'ow', 'oy', 'sh', 'th', 'uh', 'uw', 'zh', 'epi', 'sil', 'vcl']
    phone_list = phone_list1 + phone_list2 + phone_list3 + phone_list4
    for phone in [ 'sil']:
        for ID in range(4):
            task = genHTKfile(phone, ID)
            task.genfbk()
            task.appendscp()
            task.appendmlf()
