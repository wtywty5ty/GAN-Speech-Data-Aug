"""
from ProcessRawData import statemap
import subprocess

smap = statemap('states.map')
for key in smap.keys():
    subprocess.run("sed -i 's/%s/%s/' HTKFILE/mlabs/gan_equal_phone.mlf"%(key, smap[key]), shell=True)

"""

"""
from genHTKfile import genHTKfile

phone_list1 = ['b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't']
phone_list2 = ['v', 'w', 'y', 'z', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'ch']
phone_list3 = ['cl', 'dh', 'dx', 'eh', 'el', 'en', 'er', 'ey', 'hh', 'ih', 'ix', 'iy']
phone_list4 = ['jh', 'ng', 'ow', 'oy', 'sh', 'th', 'uh', 'uw', 'zh', 'epi', 'sil', 'vcl']
phone_list = phone_list1 + phone_list2 + phone_list3 + phone_list4
for phone in phone_list:
    for ID in range(10):
        task = genHTKfile(phone, ID)
        task.appendmlf()



import subprocess
from utils.ProcessRawData import computeprior
DIR = '/home/ty/tw472/triphone/temp.tri_Z/dnntrain'
HTKcmd = '%s/hmm0/HNTrainSGD -B -C %s/basic.cfg -C %s/finetune.cfg -S %s/lib/flists/train.scp -l LABEL -I %s/train.mlf -H %s/hmm0/MMF -M %s/hmm0 %s/hmms.mlist' % (
        DIR, DIR, DIR, DIR, DIR, DIR, DIR, DIR)

s = subprocess.Popen(HTKcmd, shell=True, stdout=subprocess.PIPE)
plist = computeprior(s)
total = 0
with open('prior.txt', 'w') as f:
    for key in plist.keys():
        total += plist[key]
        f.write('%d %.6f\n' %(key, plist[key]))

print(total)


# HNTrainSGD
DIR = '/home/ty/tw472/triphone/temp.tri_Z/dnntrain'
HTKcmd = '%s/hmm0/HNTrainSGD -B -C %s/basic.cfg -C %s/finetune_gpu%d.cfg -S %s/lib/flists/train.scp -l LABEL -I %s/train.mlf -H %s/hmm0/MMF -M %s/hmm0 %s/hmms.mlist' % (
            DIR, DIR, DIR, opt.gpu_id, DIR, DIR, DIR, DIR, DIR)

s = subprocess.Popen(HTKcmd, shell=True, stdout=subprocess.PIPE)

#for ... loop
    l = s.stdout.read(4)
    l = struct.unpack("i", l)
    l = l[0]
    label = s.stdout.read(l * 4)
    label = struct.unpack("%di" % l, label)
    size = s.stdout.read(8)
    size = struct.unpack("ii", size)
    data_size = size[0] * size[1]
    data = s.stdout.read(data_size * 4)
    data = struct.unpack("%df" % data_size, data)
    # ...

s.kill()

# HNForward
DIR = '/home/ty/tw472/triphone/FH7/dnntrain'
HTKcmd = '%s/HNForward -C %s/basic.cfg -C %s/eval.cfg -H %s/hmm0/MMF %s/hmms.mlist'%(DIR,DIR,DIR,DIR,DIR)
s = subprocess.Popen(HTKcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
#for loop ...
    # pack data ...
    s.stdin.write(binary_data)
    rows = s.stdout.read(4)
    rows = struct.unpack('i', rows)[0]
    columns = s.stdout.read(4)
    columns = struct.unpack('i',columns)[0]
    results_list = []
    for i in range(rows):
        results = s.stdout.read(columns*4)
        results = struct.unpack('%df'% columns, results)
        results_list.append(results)
end = struct.pack('i', 0)
s.stdin.write(end)
s.kill()
"""
from utils.ProcessRawData import *
prior = {}
phonemap = triphoneMap('slist.txt', 'aa')
with open('prior.txt', 'r') as f:
    for line in f.readlines():
        tid = int(line.split(' ')[0])
        state = phonemap.id2state[tid]
        phone = state.split('_')[0]
        pr = float(line.split(' ')[1].strip('\n'))
        if phone in prior:
            prior[phone] += pr
        else:
            prior[phone] = pr

with open('phone_prior.txt', 'w') as f:
    for phone in prior.keys():
        f.write('%s %f\n'%(phone, prior[phone]))




