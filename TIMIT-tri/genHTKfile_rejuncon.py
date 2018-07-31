import torch
import struct
import subprocess
import numpy
import math
import os, time
from utils.ProcessRawData import *
from utils.GenerateData import *
from utils.TestData import *



class genHTKfile(object):
    def __init__(self, phone, ID):
        self.mode = 'uncon_prior' #reject correct samples
        #self.data = 'uniform'
        self.data = 'prior'
        self.ID = ID
       # 'uncon_rejw' reject wrong samples 
        # 'prior'
        self.sampPeriod = 100000
        self.sampSize = 2080
        self.parmKind = 9 + 0o004000
        self.phone = phone
        G_PATH = 'outf/GAN_array_uncon/%s/netG_.pkl'%phone
        generator = torch.load(G_PATH, map_location=lambda storage, loc: storage).cuda().eval()
        self.generator = generator
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
        DIR = '/home/ty/tw472/triphone/FH7/dnntrain'
        HTKcmd = '%s/HNForward -C %s/basic.cfg -C %s/eval.cfg -H %s/hmm0/MMF %s/hmms.mlist' % (DIR, DIR, DIR, DIR, DIR)
        s = subprocess.Popen(HTKcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        print('Start generating %s samples:'%self.phone)
        body = 0
        for fid in range(nclass):
            size = 0
            _size = 0
            while _size != self.splitSize[fid]:
                buf, data = generateDataUncon(self.generator, 8000)
                s.stdin.write(buf)
                s.stdin.flush()
                index = pickstate(s, phoneMap, fid)
                data = data[:, :13, :].detach()
                data_f = data[index]
                size += data_f.size(0)
                if size > self.splitSize[fid]:
                    rem = self.splitSize[fid] - _size
                    data_f = data_f[:rem]
                    _size = _size + data_f.size(0)
                else:
                    _size = size
               # print('%d: %d'%(self.splitSize[fid], _size))
                flat_data = data_f.cpu().view(-1).detach().numpy()
                if body == 0:
                    body = flat_data.astype('>f').tostring()
                else:
                    body = body + flat_data.astype('>f').tostring()
        
        header = struct.pack('>iihh', self.nSamples, self.sampPeriod, self.sampSize, self.parmKind)
        binary = header + body
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
    start_time = time.time()
    for phone in phone_list:
        task = genHTKfile(phone, ID)
        task.genfbk()
        task.appendscp()
        task.appendmlf()
    end_time = time.time()
    total = end_time - start_time
    print('total time: %.3f'%(total/3600))


        

