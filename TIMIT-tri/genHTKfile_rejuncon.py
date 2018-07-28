import torch
import struct
import subprocess
import numpy
import math
import os, signal
from utils.ProcessRawData import *
from utils.GenerateData import *
from utils.TestData import *



class genHTKfile(object):
    def __init__(self, phone, ID):
        self.mode = 'uncon_rejw' #reject wrong samples
        # 'prior'
        self.sampPeriod = 100000
        self.sampSize = 2080
        self.parmKind = 9 + 0o004000
        self.phone = phone
        G_PATH = 'outf/GAN_array_uncon/%s/netG_.pkl'%phone
        generator = torch.load(G_PATH, map_location=lambda storage, loc: storage).cuda().eval()
        self.generator = generator
        self.phoneMap = triphoneMap('slist.txt', phone)




    def genSamples(self):
        phoneMap = self.phoneMap
        DIR = '/home/ty/tw472/triphone/FH7/dnntrain'
        HTKcmd = '%s/HNForward -C %s/basic.cfg -C %s/eval.cfg -H %s/hmm0/MMF %s/hmms.mlist' % (DIR, DIR, DIR, DIR, DIR)
        s = subprocess.Popen(HTKcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        print('Start generating %s samples:'%self.phone)

        buf, data = generateDataUncon(self.generator, 6000)
        s.stdin.write(buf)
        s.stdin.flush()
        index, splitSizeSet = genLabelFilter(s, phoneMap)
        data = data[:, :13, :]
        data_f = data[index]
        nSamples = data_f.size(0)

        flat_data = data_f.cpu().view(-1).detach().numpy()
        body_ = flat_data.astype('>f').tostring()
        header = struct.pack('>iihh', nSamples, self.sampPeriod, self.sampSize, self.parmKind)
        binary = header + body_
        print('..finish!')
        end = struct.pack('i', 0)
        s.stdin.write(end)
        s.kill()

        return binary, nSamples, splitSizeSet

    def genfbk(self):
        totalsamp = 0
        id = 0
        while totalsamp < 60000:
            binary, nSamples, splitSizeSet = self.genSamples()
            with open('HTKFILE/fbk/%s/%s_gan_%d_%s.fbk' % (self.mode, self.phone, id, self.mode), 'wb') as f:
                f.write(binary)
            print('Generating fbk file successfully')
            self.appendscp(nSamples, id)
            self.appendmlf(splitSizeSet, id)
            totalsamp += nSamples
            id += 1


    def appendscp(self, nSamples, ID):
        with open('HTKFILE/flists/gan_%s.scp'%self.mode, 'a') as f:
            f.write('%s_gan_%d_%s.fbk' % (self.phone, self.ID, self.mode)
                    + '=/home/ty/tw472/master/FH5_w16d40_tri/HTKFILE/fbk/%s/%s_gan_%d_%s.fbk' % (self.mode, self.phone, ID, self.mode)
                    + '[0,%d]\n'%(nSamples-1))
        print('Appending scp file successfully')


    def appendmlf(self, splitSizeSet, ID):
        smap = statemap('states.map')
        if not os.path.exists('HTKFILE/mlabs/gan_%s.mlf' % self.mode):
            with open('HTKFILE/mlabs/gan_%s.mlf' % self.mode, 'w') as f:
                f.write('#!MLF!#\n')
        with open('HTKFILE/mlabs/gan_%s.mlf' % self.mode, 'a') as f:
            f.write('"%s_gan_%d_%s.lab"\n' % (self.phone, ID, self.mode))
            for id in range(self.phoneMap.nlabels()):
                vstate = smap[self.phoneMap.f2states[id]]
                splitSize = splitSizeSet[id]

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

    pprior = {}
    with open('phone_prior.txt', 'r') as f:
        for line in f.readlines():
            pprior[line.split(' ')[0]] = line.split(' ')[1].strip('\n')
    for phone in phone_list:
        task = genHTKfile(phone)
        task.genfbk()


        """
        total = 384
        for ID in range(math.ceil(total*pprior[phone])):
            task = genHTKfile(phone, ID)
            task.appendscp()

        """

