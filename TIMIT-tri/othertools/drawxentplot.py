import matplotlib.pyplot as plt
import numpy as np
from utils.ProcessRawData import triphoneMap

phonemap = triphoneMap('slist.txt', 'aa')
'''
xd= {}
with open ('/Users/tianyu/Desktop/train_xent.txt', 'r') as f:
    for line in f.readlines():
        if line.split(' ')[0] == 'state':
            continue
        tid = int(line.split(' ')[0])
        xent = float(line.strip('\n').split(' ')[1])
        xd[tid] = xent


train_list = [xd[tid] for tid in range(808)]
train_list = np.array(train_list)
'''
xd= {}
with open ('/Users/tianyu/Desktop/test_ent.txt', 'r') as f:
    for line in f.readlines():
        if line.split(' ')[0] == 'state':
            continue
        tid = int(line.split(' ')[0])
        xent = float(line.strip('\n').split(' ')[1])
        xd[tid] = xent


test_list = [xd[tid] for tid in range(808)]
test_list = np.array(test_list)

xd= {}
with open ('/Users/tianyu/Desktop/fake_ent.txt', 'r') as f:
    for line in f.readlines():
        if line.split(' ')[0] == 'state':
            continue
        tid = int(line.split(' ')[0])
        xent = float(line.strip('\n').split(' ')[1])
        xd[tid] = xent


fake_list = [xd[tid] for tid in range(808)]
fake_list = np.array(fake_list)


order = test_list.argsort()[::-1]
plt.figure(figsize=(200,100))
N = 808
ind = np.arange(N)
'''
total_width, n = 0.8, 2
width = total_width / n
ind = ind - (total_width - width) / 2
'''

#plt.bar(ind, train_list[order], label='training set')
plt.bar(ind, fake_list[order], label='test set')
plt.ylabel('Entropy', fontsize=100)
plt.xlabel('Triphone State', fontsize=100)
_phone_list = []
for i in order:
    _phone_list.append(phonemap.id2states[i])
plt.xticks(ind, _phone_list, rotation=80, fontsize = 10)
plt.yticks(fontsize=50)
plt.title('Average Cross Entroy for Each State (Synthesised Data)', fontsize=160)
#plt.legend(fontsize=150)
print('Saving Results ...')
plt.savefig('averageEnt_fake.png' )
plt.close()