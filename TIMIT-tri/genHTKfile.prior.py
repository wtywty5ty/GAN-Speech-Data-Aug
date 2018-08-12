import torch
import struct
import subprocess
import numpy
import os, signal
import random
import time
from utils.ProcessRawData import *
from utils.GenerateData import *
from utils.TestData import *



class genHTKfile(object):
    def __init__(self, phone, ID):
        #self.mode = 'equal_phone'  # almost fixed file size
        self.mode = 'con_prior_u09'
        # 'prior'
        self.data = 'prior'
        self.sampPeriod = 100000
        self.sampSize = 2080
        self.parmKind = 9 + 0o004000
        self.phone = phone
        self.ID = ID
        G_PATH = 'outf/GAN_array/%s/netG_.pkl'%phone
        self.generator = torch.load(G_PATH, map_location=lambda storage, loc: storage).cuda().eval()
        self.phoneMap = triphoneMap('slist.txt', phone)
        self.total = 1080000 #3 hours
    
        if self.data == 'prior':
            self.splitSize = {}
            self.nSamples = 0
            prior = {}
            with open('prior.txt', 'r') as f:
                for line in f.readlines():
                    state = int(line.split(' ')[0])
                    pr = float(line.split(' ')[1].strip('\n'))
                    prior[state] = pr
            totalsamp = self.total 
            for fid in range(self.phoneMap.nlabels()):
                self.splitSize[fid] = int(totalsamp*prior[self.phoneMap.f2t[fid]])
                self.nSamples += self.splitSize[fid]


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
            splitSize = splitSizeSet[id]
            size = 0
            _size = 0
            while _size != splitSize:
                buf, data = generateData(self.generator, 2000, nclass, id)
                s.stdin.write(buf)
                s.stdin.flush()
                tid = phoneMap.f2t[id]
                index = dataFilter(s, tid, 0.9)
                data = data[:, :13, :].detach()
                data_f = data[index]
                size += data_f.size(0)
                if size > splitSize:
                    rem = splitSize - _size
                    data_f = data_f[:rem]
                    _size += data_f.size(0)
                else:
                    _size = size
                samples = data_f.cpu().view(-1).numpy()
                body_ = samples.astype('>f').tostring()
                binary = binary + body_

            print('%d %d'%(id, splitSize))
        print('..finish!')
        end = struct.pack('i', 0)
        s.stdin.write(end)
        s.kill()
        return binary

    def genfbk(self):
        binary = self.genSamples()
        os.makedirs('HTKFILE/fbk/%s' % (self.mode), exist_ok=True)
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
        if not os.path.exists('HTKFILE/mlabs/gan_%s.mlf' % self.mode):
            with open('HTKFILE/mlabs/gan_%s.mlf' % self.mode, 'w') as f:
                f.write('#!MLF!#\n')
        with open('HTKFILE/mlabs/gan_%s.mlf' % self.mode, 'a') as f:
            f.write('"%s_gan_%d_%s.lab"\n' % (self.phone, self.ID, self.mode))
            for id in range(self.phoneMap.nlabels()):
                vstate = smap[self.phoneMap.f2states[id]]
                splitSize = self.splitSize[id]

                if id == 0:
                    start = 0
                    end = start + splitSize * self.sampPeriod
                else:
                    start = end
                    end = start + splitSize * self.sampPeriod

                f.write('%d %d %s\n' % (start, end, vstate))

            f.write('.\n')
            print('Appending mlf file successfully')

if __name__ == '__main__':
    phone_list1 = ['b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't']
    phone_list2 = ['v', 'w', 'y', 'z', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'ch']
    phone_list3 = ['cl', 'dh', 'dx', 'eh', 'el', 'en', 'er', 'ey', 'hh', 'ih', 'ix', 'iy']
    phone_list4 = ['jh', 'ng', 'ow', 'oy', 'sh', 'th', 'uh', 'uw', 'zh', 'epi', 'sil', 'vcl']
    phone_list = phone_list1 + phone_list2 + phone_list3 + phone_list4
    ID = 0
    manualSeed = 2018 + ID
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manualSeed)
    start_time = time.time()

    for phone in phone_list:
        task = genHTKfile(phone, ID)
        task.genfbk()
        task.appendscp()
        task.appendmlf()
    end_time = time.time()
    total = end_time - start_time
    print('total time: %.3f'%(total/3600))