import torch
import struct
import numpy
import os
from utils import triphoneMap




class genHTKfile(object):
    def __init__(self, phone, ID):
        self.mode = 'average1'  # almost fixed file length
        # 'average2' mode: non fixed file length. Fixed tied-state length instead.
        self.nSamples = 18000
        self.sampPeriod = 100000
        self.sampSize = 2080
        self.parmKind = 9 + 0o004000
        self.phone = phone
        self.ID = ID
        G_PATH = 'outf/GAN_array/%s/netG_.pkl'%phone
        self.generator = torch.load(G_PATH).eval()
        self.phoneMap = triphoneMap('slist.txt', phone)
        if self.mode == 'average1':
            self.nSamples = 18000 - 18000 % self.phoneMap.nlabels()
            self.splitSize = self.nSamples//self.phoneMap.nlabels()


    def genSamples(self):
        if self.mode == 'average1':
            phoneMap = self.phoneMap
            nclass = phoneMap.nlabels()
            splitSize = self.splitSize
            print('Start generating samples:')
            for id in range(nclass):
                noise = torch.randn(splitSize, 100).cuda()
                y = torch.zeros(splitSize, nclass).cuda()
                y[:, id] = 1
                gen_data = self.generator(torch.cat([noise, y], 1)).squeeze()
                gen_data = gen_data[:, :13, :]
                gen_data = 3 * gen_data

                if id == 0:
                    gen_data_ = gen_data
                else:
                    gen_data_ = torch.cat([gen_data_, gen_data], 0)
                print('..'+str(id), end="")
                
            print('..finish!')
            return gen_data_.cpu().view(-1)


    def genfbk(self):
        samples = self.genSamples().detach().numpy()
        print(samples[:520])
        body = samples.tobytes()
        # body = struct.pack('>%df'%len(samples), *samples)
        header = struct.pack('>iihh', self.nSamples, self.sampPeriod, self.sampSize, self.parmKind) 
        with open('HTKFILE/fbk/%s_gan_%d.fbk' % (self.phone, self.ID), 'wb') as f:
            f.write(header+body)
        print('Generating fbk file successfully')

    def appendscp(self):
        with open('HTKFILE/flists/gan.scp', 'a') as f:
            f.write('%s_gan_%d.fbk'%(self.phone, self.ID)
                    + '=/home/ty/tw472/master/FH5_w16d40_tri/HTKFILE/fbk/%s_gan_%d.fbk' % (self.phone, self.ID)
                    + '[0, %d]\n'%(self.nSamples-1))
        print('Appending scp file successfully')

    def appendmlf(self):
        if not os.path.exists('HTKFILE/mlabs/gan.mlf'):
            with open('HTKFILE/mlabs/gan.mlf', 'w') as f:
                f.write('#!MLF!#\n')
        with open('HTKFILE/mlabs/gan.mlf', 'a') as f:
            f.write('"%s_gan_%d.lab"\n'% (self.phone, self.ID))
            for id in range(self.phoneMap.nlabels()):
                f.write('%d %d %s\n'% (id*self.splitSize*self.sampPeriod,
                                     (id+1)*self.splitSize*self.sampPeriod, self.phoneMap.f2states[id]))

            f.write('.\n')
            print('Appending mlf file successfully')

if __name__ == '__main__':
    task = genHTKfile('b', 110)
    task.genfbk()
    task.appendscp()
    task.appendmlf()
