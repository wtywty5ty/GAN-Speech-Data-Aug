import argparse
import os
import torch
import subprocess
import matplotlib.pyplot as plt
from utils import * 


plt.switch_backend('agg')
parser = argparse.ArgumentParser()
parser.add_argument('--phone', default='g', help="phone")
parser.add_argument('--batchsize', type=int, default=500, help="Batch size")
parser.add_argument('--G', default='outf/GAN_array/g/netG_.pkl', help="path to generator (to continue training)")
parser.add_argument('--uncon', action='store_true', help='enables uncon mode')
parser.add_argument('--embd', action='store_true', help='enables embd mode')
parser.add_argument('--evaluation', default='evaluation/GAN_array', help="path to output files)")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.evaluation, exist_ok=True)

if opt.G == '':
    raise IOError('Please enter the correct location')

generator = torch.load(opt.G).eval()
phonemap = triphoneMap('slist.txt',opt.phone)
nclass = phonemap.nlabels()

if opt.uncon:
    buf = generateDataUncon(generator, opt.batchsize)
elif opt.embd:
    buf = generateDataEmbd_all(generator, opt.batchsize, nclass)
else:
    buf = generateData_all(generator, opt.batchsize, nclass)

DIR = '/home/ty/tw472/triphone/FH7/dnntrain'
HTKcmd = '%s/HNForward -C %s/basic.cfg -C %s/eval.cfg -H %s/hmm0/MMF %s/hmms.mlist'%(DIR,DIR,DIR,DIR,DIR)
s = subprocess.Popen(HTKcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
s.stdin.write(buf)
#s.stdin.flush()
row, column, results = testResults(s)
for i, key in enumerate(phonemap.states.keys()):
    phone_id = phonemap.states[key]
    top1 = correctness(results[i*opt.batchsize:(i+1)*opt.batchsize], phone_id, 1)
    top3 = correctness(results[i*opt.batchsize:(i+1)*opt.batchsize], phone_id, 3)
    top5 = correctness(results[i*opt.batchsize:(i+1)*opt.batchsize], phone_id, 5)
    print('%s Classidication Acc Top1: %f, Top3: %f, Top5: %f \n'%(key, top1, top3, top5)) 
    with open('%s/classification.txt'%opt.evaluation, 'a') as f:
        f.write('%s Classidication Acc Top1: %f, Top3: %f, Top5: %f \n'%(key, top1, top3, top5))

print('Saving Results ...')

s.kill()


