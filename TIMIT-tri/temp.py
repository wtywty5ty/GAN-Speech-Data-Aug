import pickle 
import numpy as np
import subprocess 
import struct
from utils.ProcessRawData import *
# read test data from TIMIT




DIR = '/home/ty/tw472/triphone/temp.tri_Z/dnntrain'
HTKcmd = '%s/hmm0/HNTrainSGD -B -C %s/basic.cfg -C %s/finetune.cfg -S %s/lib/flists/dnn.cv.scp -l LABEL -I %s/train.mlf -H %s/hmm0/MMF -M %s/hmm0 %s/hmms.mlist' % (
            DIR, DIR, DIR, DIR, DIR, DIR, DIR, DIR)

'''
DIR = '/home/ty/tw472/master/acoustic/GAN/dnntrain.3.uncon.prior'
HTKcmd = '%s/hmm0/HNTrainSGD -B -C %s/basic.cfg -C %s/finetune.cfg -S %s/DATA/flists/uncon.prior.3.scp -l LABEL -I %s/DATA/mlabs/uncon.prior.3.mlf -H %s/hmm0/MMF -M %s/hmm0 %s/hmms.mlist' % (
            DIR, DIR, DIR, DIR, DIR, DIR, DIR, DIR)
'''
'''
DIR = '/home/ty/tw472/triphone/temp.tri_Z/dnntrain'
HTKcmd = '%s/hmm0/HNTrainSGD -B -C %s/basic.cfg -C %s/finetune.cfg -S %s/lib/flists/test.scp -l LABEL -I %s/timit_test.mlf -H %s/hmm0/MMF -M %s/hmm0 %s/hmms.mlist'%(
    DIR,DIR,DIR,DIR,DIR,DIR,DIR,DIR)
'''
data = subprocess.Popen(HTKcmd, shell=True, stdout=subprocess.PIPE)

DIR = '/home/ty/tw472/triphone/FH7/dnntrain'
HTKcmd = '%s/HNForward -C %s/basic.cfg -C %s/eval.cfg -H %s/hmm0/MMF %s/hmms.mlist' % (DIR, DIR, DIR, DIR, DIR)
test = subprocess.Popen(HTKcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

xent_list = []
last_batch =False
while not last_batch:
    llabel = data.stdout.read(4)
    llabel = struct.unpack('i', llabel)[0]
    label = data.stdout.read(llabel*4)
    label = struct.unpack('%di'%llabel, label)
    buf = data.stdout.read(llabel*640*4+8)
    if llabel<10000:
        last_batch = True

    test.stdin.write(buf)
    test.stdin.flush()
    
    rows = test.stdout.read(4)
    rows = struct.unpack('i', rows)[0]
    columns = test.stdout.read(4)
    columns = struct.unpack('i',columns)[0]
    for i in range(rows):
        tid = label[i]
        results = test.stdout.read(columns*4)
        results = struct.unpack('%df'% columns, results)
        results = np.array(results)
        phonemap_temp = triphoneMap('slist.txt', 'aa')
        phone = phonemap_temp.id2states[tid].split('_')[0]
        phonemap = triphoneMap('slist.txt', phone)
        tid_pool = [phonemap.f2t[i] for i in range(phonemap.nlabels())]
        score_pool = [results[tid] for tid in tid_pool]
        cls_score = max(score_pool)
        cls_fid = score_pool.index(cls_score)
        if tid == tid_pool[cls_fid]:
            xent_list.append(1)
        else:
            xent_list.append(0)

#avgxent = {}
#for tid in rdict.keys():
#    avgxent[tid] = rdict[tid][0]/rdict[tid][1]
print(np.mean(xent_list))

data.kill()
test.kill()
