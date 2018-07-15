import torch
import struct
import numpy
import os
from utils import triphoneMap


class genHTKfile(object):
    def __init__(self, phone, ID):
        self.mode = 'equal_phone'  # almost fixed file size
        # 'equal_state' mode: non fixed file length. Fixed split size instead
        # 'prior'
        self.nSamples = 18000
        self.sampPeriod = 100000
        self.sampSize = 2080
        self.parmKind = 9 + 0o004000
        self.phone = phone
        self.ID = ID
        G_PATH = 'outf/GAN_array/%s/netG_.pkl'%phone
        self.generator = torch.load(G_PATH).eval()
        self.phoneMap = triphoneMap('slist.txt', phone)
        if self.mode == 'equal_phone':
            self.nSamples = 18000 - 18000 % self.phoneMap.nlabels()
            self.splitSize = self.nSamples//self.phoneMap.nlabels()

    def genSamples(self):
        phoneMap = self.phoneMap
        nclass = phoneMap.nlabels()
        splitSize = self.splitSize
        print('Start generating %s samples:'%self.phone)
        for id in range(nclass):
            noise = torch.randn(splitSize, 100).cuda()
            y = torch.zeros(splitSize, nclass).cuda()
            y[:, id] = 1
            gen_data = self.generator(torch.cat([noise, y], 1)).squeeze()
            gen_data = gen_data[:, :13, :]
            gen_data = 3 * gen_data

            samples = gen_data.cpu().view(-1).detach().numpy()
            body_ = samples.astype('>f').tostring()
            if id == 0:
                body = body_
            else:
                body = body + body_
            print('..'+str(id), end="")

        print('..finish!')
        return body

    def genfbk(self):
        body = self.genSamples()
        header = struct.pack('>iihh', self.nSamples, self.sampPeriod, self.sampSize, self.parmKind) 
        with open('HTKFILE/fbk/%s_gan_%d_%s.fbk' % (self.phone, self.ID, self.mode), 'wb') as f:
            f.write(header+body)
        print('Generating fbk file successfully')

    def appendscp(self):
        with open('HTKFILE/flists/gan_%s.scp'%self.mode, 'a') as f:
            f.write('%s_gan_%d_%s.fbk' % (self.phone, self.ID, self.mode)
                    + '=/home/ty/tw472/master/FH5_w16d40_tri/HTKFILE/fbk/%s_gan_%d_%s.fbk' % (self.phone, self.ID, self.mode)
                    + '[0, %d]\n'%(self.nSamples-1))
        print('Appending scp file successfully')

    def appendmlf(self):
        if not os.path.exists('HTKFILE/mlabs/gan_%s.mlf'%self.mode):
            with open('HTKFILE/mlabs/gan_%s.mlf'%self.mode, 'w') as f:
                f.write('#!MLF!#\n')
        with open('HTKFILE/mlabs/gan_%s.mlf'%self.mode, 'a') as f:
            f.write('"%s_gan_%d_%s.lab"\n'% (self.phone, self.ID, self.mode))
            for id in range(self.phoneMap.nlabels()):
                f.write('%d %d %s\n'% (id*self.splitSize*self.sampPeriod,
                                     (id+1)*self.splitSize*self.sampPeriod, self.phoneMap.f2states[id]))

            f.write('.\n')
            print('Appending mlf file successfully')


if __name__ == '__main__':
    phone_list1 = ['b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't']
    phone_list2 = ['v', 'w', 'y', 'z', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'ch']
    phone_list3 = ['cl', 'dh', 'dx', 'eh', 'el', 'en', 'er', 'ey', 'hh', 'ih', 'ix', 'iy']
    phone_list4 = ['jh', 'ng', 'ow', 'oy', 'sh', 'th', 'uh', 'uw', 'zh', 'epi', 'sil', 'vcl']
    phone_list = phone_list1 + phone_list2 + phone_list3 + phone_list4
    for phone in phone_list:
        for ID in range(10):
            task = genHTKfile(phone, ID)
            task.genfbk()
            task.appendscp()
            task.appendmlf()
