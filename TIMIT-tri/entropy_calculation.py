import pickle 
import numpy as np
import subprocess 
import struct
# read test data from TIMIT



'''
DIR = '/home/ty/tw472/triphone/temp.tri_Z/dnntrain'
HTKcmd = '%s/hmm0/HNTrainSGD -B -C %s/basic.cfg -C %s/finetune.cfg -S %s/lib/flists/dnn.train.scp -l LABEL -I %s/train.mlf -H %s/hmm0/MMF -M %s/hmm0 %s/hmms.mlist' % (
            DIR, DIR, DIR, DIR, DIR, DIR, DIR, DIR)
'''

DIR = '/home/ty/tw472/master/acoustic/GAN/dnntrain.3.uncon.prior'
HTKcmd = '%s/hmm0/HNTrainSGD -B -C %s/basic.cfg -C %s/finetune.cfg -S %s/DATA/flists/uncon.prior.3.scp -l LABEL -I %s/DATA/mlabs/uncon.prior.3.mlf -H %s/hmm0/MMF -M %s/hmm0 %s/hmms.mlist' % (
            DIR, DIR, DIR, DIR, DIR, DIR, DIR, DIR)

'''
DIR = '/home/ty/tw472/triphone/temp.tri_Z/dnntrain'
HTKcmd = '%s/hmm0/HNTrainSGD -B -C %s/basic.cfg -C %s/finetune.cfg -S %s/lib/flists/test.scp -l LABEL -I %s/timit_test.mlf -H %s/hmm0/MMF -M %s/hmm0 %s/hmms.mlist'%(
    DIR,DIR,DIR,DIR,DIR,DIR,DIR,DIR)
'''
data = subprocess.Popen(HTKcmd, shell=True, stdout=subprocess.PIPE)

DIR = '/home/ty/tw472/triphone/FH7/dnntrain'
HTKcmd = '%s/HNForward -C %s/basic.cfg -C %s/eval.cfg -H %s/hmm0/MMF %s/hmms.mlist' % (DIR, DIR, DIR, DIR, DIR)
test = subprocess.Popen(HTKcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

rdict = {}
last_batch =False
while not last_batch:
    llabel = data.stdout.read(4)
    llabel = struct.unpack('i', llabel)[0]
    label = data.stdout.read(llabel*4)
    label = struct.unpack('%di'%llabel, label)
    buf = data.stdout.read(llabel*520*4+8)
    if llabel<10000:
        last_batch = True

    test.stdin.write(buf)
    test.stdin.flush()
    
    rows = test.stdout.read(4)
    rows = struct.unpack('i', rows)[0]
    columns = test.stdout.read(4)
    columns = struct.unpack('i',columns)[0]
    results_list = []
    for i in range(rows):
        results = test.stdout.read(columns*4)
        results = struct.unpack('%df'% columns, results)
        results = np.array(results)
        tid = label[i]
        entropy = -np.sum(results*np.log(results))
        if tid in rdict:
            #rdict[tid][0] += -np.sum(results*np.log(results))
            #rdict[tid][1] += 1
            if entropy < rdict[tid][0]:
                rdict[tid][0] = entropy
            elif entropy > rdict[tid][1]:
                rdict[tid][1] = entropy

        else:
            rdict[tid] = {}
            #rdict[tid][0] = -np.sum(results*np.log(results)) 
            #rdict[tid][1] = 1
            rdict[tid][0] = entropy
            rdict[tid][1] = entropy

#avgxent = {}
#for tid in rdict.keys():
#    avgxent[tid] = rdict[tid][0]/rdict[tid][1]

with open('fake_ent_boundary.txt', 'w') as f:
    f.write('state min max\n')
    for tid in rdict.keys():
        f.write('%d %.4f %.4f\n'%(tid, rdict[tid][0], rdict[tid][1]))

data.kill()
test.kill()

    

