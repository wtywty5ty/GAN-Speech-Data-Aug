import argparse
import subprocess
import struct
import matplotlib.pyplot as plt
from utils import generateData, generateDataUncon, testResults, returnMeanStd, triTestsetAcc, triphoneMap


plt.switch_backend('agg')
parser = argparse.ArgumentParser()
parser.add_argument('--phone', default='aa', help="phone")
parser.add_argument('--batchsize', type=int, default=100000, help="Batch size")
opt = parser.parse_args()
print(opt)


phonemap = triphoneMap('slist.txt', opt.phone)
# read test data from TIMIT
DIR = '/home/ty/tw472/triphone/temp.tri_Z/dnntrain'
HTKcmd_test = '%s/hmm0/HNTrainSGD -B -C %s/basic.cfg -C %s/finetune_test.cfg -S %s/lib/flists/test.scp -l LABEL -I %s/timit_test.mlf -H %s/hmm0/MMF -M %s/hmm0 %s/hmms.mlist'%(DIR,DIR,DIR,DIR,DIR,DIR,DIR,DIR)
test_s = subprocess.Popen(HTKcmd_test, shell=True, stdout=subprocess.PIPE)
llabel = test_s.stdout.read(4)
llabel = struct.unpack('i', llabel)[0]
label = test_s.stdout.read(llabel*4)
label = struct.unpack('%di'%llabel, label)
buf = test_s.stdout.read(llabel*640*4+8)

DIR = '/home/ty/tw472/triphone/FH7/dnntrain'
HTKcmd = '%s/HNForward -C %s/basic.cfg -C %s/eval.cfg -H %s/hmm0/MMF %s/hmms.mlist'%(DIR,DIR,DIR,DIR,DIR)
s = subprocess.Popen(HTKcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
s.stdin.write(buf)
row, column, results = testResults(s)
top1 = triTestsetAcc(results, label, phonemap, 1)
top3 = triTestsetAcc(results, label, phonemap, 3)
top5 = triTestsetAcc(results, label, phonemap, 5)

for key in top1.keys():
    print('%s Classidication Acc Top1: %f, Top3: %f, Top5: %f \n'%(key, top1[key][2], top3[key][2], top5[key][2]))

