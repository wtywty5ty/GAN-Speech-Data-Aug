import argparse
import os
import torch
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from utils import generateData, generateDataUncon, testResults, returnMeanStd, correctness, correctness_, triphoneMap


plt.switch_backend('agg')
parser = argparse.ArgumentParser()
parser.add_argument('--phone', default='aa', help="phone")
parser.add_argument('--phoneID', type = int, default=16, help='phone level ID')
parser.add_argument('--tstate', default='aa_s2_1', help=" the state of output phone")
parser.add_argument('--batchsize', type=int, default=1000, help="Batch size")
parser.add_argument('--G', default='outf/log-aa-divide4/netG_.pkl', help="path to generator (to continue training)")
parser.add_argument('--uncon', action='store_true', help='enables uncon mode')
parser.add_argument('--evaluation', default='evaluation/con', help="path to output files)")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.evaluation, exist_ok=True)

if opt.G == '':
    raise IOError('Please enter the correct location')

generator = torch.load(opt.G)
phonemap = triphoneMap('slist.txt',opt.phone)
nclass = phonemap.nlabels()
class_id = phonemap.state2label(opt.tstate)

if opt.uncon:
    buf = generateDataUncon(generator, opt.batchsize)
else:
    buf = generateData(generator, opt.batchsize, nclass, class_id)

DIR = '/home/ty/tw472/triphone/FH7/dnntrain'
HTKcmd = '%s/HNForward -C %s/basic.cfg -C %s/eval.cfg -H %s/hmm0/MMF %s/hmms.mlist'%(DIR,DIR,DIR,DIR,DIR)
s = subprocess.Popen(HTKcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
s.stdin.write(buf)
row, column, results = testResults(s)
mean_list, std_list = returnMeanStd(results)
phone_id = phonemap.states[opt.tstate]
print('Classification correctness top1: %f' %correctness(results, phone_id, 1))
print('Classification correctness top3: %f' %correctness(results, phone_id, 3))
print('Classification correctness top5: %f' %correctness(results, phone_id, 5))

DIR = '/home/ty/tw472/MLSALT2/TIMIT/exp/FH5/dnntrain'
HTKcmd = '%s/example/HNForward -C %s/basic.cfg -C %s/eval.cfg -H %s/hmm0/MMF %s/hmms.mlist'%(DIR,DIR,DIR,DIR,DIR)
s = subprocess.Popen(HTKcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
s.stdin.write(buf)
row, column, results = testResults(s)
print('(phone level) Classification correctness top1: %f' %correctness_(results, opt.phoneID, 1))
print('(phone level) Classification correctness top3: %f' %correctness_(results, opt.phoneID, 3))
print('(phone level) Classification correctness top5: %f' %correctness_(results, opt.phoneID, 5))

#top10
top10 = mean_list.argsort()[-10:][::-1]
plt.figure()
N = 10
ind = np.arange(N)
plt.bar(ind, mean_list[top10])
plt.ylabel('Confidence Score', fontsize=13)
plt.xlabel('Phone', fontsize=13)
_phone_list = []
for i in top10:
    _phone_list.append(phonemap.id2states[i])
plt.xticks(ind, _phone_list, rotation=30, fontsize =10)
plt.title('Confidence Score Top10 (phone: %s)'%opt.phone, fontsize=14)
print('Saving Results ...')
plt.savefig('%s/%s_results_top10.png' % (opt.evaluation, opt.tstate))
plt.close()



s.kill()


