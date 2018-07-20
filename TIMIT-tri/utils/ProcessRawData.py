import torch
import struct


def processData(s, batchsize, phone_id, stack_w):
    # For mono phone system
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
    last_batch = False
    if size[0]<1024:
        s.stdout.readline()
        s.stdout.readline()
        last_batch = True

    # arrange data into mini batch
    dataloader = []
    num = size[0]//batchsize
    rem = size[0]%batchsize
    for i in range(num):
        batch = []
        x = data[i * batchsize * size[1]:i * batchsize * size[1] + batchsize * size[1]]
        x = torch.FloatTensor(x).view(batchsize, 1, stack_w, -1)
        x = x/3
        batch.append(x)
        y = label[i*batchsize:i*batchsize+batchsize]
        #y = torch.tensor(y)
        y = torch.tensor(y)-phone_id
        batch.append(y)
        dataloader.append(batch)
    if rem != 0 and rem != 1:
        batch = []
        x = data[data_size-rem*size[1]: data_size]
        x = torch.FloatTensor(x).view(rem, 1, stack_w, -1)
        x = x/3
        batch.append(x)
        y = label[size[0]-rem : size[0]]
        #y = torch.tensor(y)
        y = torch.tensor(y)-phone_id
        batch.append(y)
        dataloader.append(batch)

    return size, dataloader, last_batch

def processDataUni(s, batchsize, stack_w):
    # For single GAN or Embedding system
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
    last_batch = False
    if size[0]<1024:
        s.stdout.readline()
        s.stdout.readline()
        last_batch = True

    # arrange data into mini batch
    dataloader = []
    num = size[0]//batchsize
    rem = size[0]%batchsize
    for i in range(num):
        batch = []
        x = data[i * batchsize * size[1]:i * batchsize * size[1] + batchsize * size[1]]
        x = torch.FloatTensor(x).view(batchsize, 1, stack_w, -1)
        x = x/3
        batch.append(x)
        y = label[i*batchsize:i*batchsize+batchsize]
        y = torch.tensor(y)
        batch.append(y)
        dataloader.append(batch)
    if rem != 0 and rem != 1:
        batch = []
        x = data[data_size-rem*size[1]: data_size]
        x = torch.FloatTensor(x).view(rem, 1, stack_w, -1)
        x = x/3
        batch.append(x)
        y = label[size[0]-rem : size[0]]
        y = torch.tensor(y)
        batch.append(y)
        dataloader.append(batch)

    return size, dataloader, last_batch

def processTriData(s, batchsize, phoneMap, stack_w):
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
    last_batch = False
    if size[0]<1024:
        s.stdout.readline()
        s.stdout.readline()
        last_batch = True

    # arrange data into mini batch
    dataloader = []
    num = size[0]//batchsize
    rem = size[0]%batchsize
    for i in range(num):
        batch = []
        x = data[i * batchsize * size[1]:i * batchsize * size[1] + batchsize * size[1]]
        x = torch.FloatTensor(x).view(batchsize, 1, stack_w, -1)
        x = x/3
        batch.append(x)
        y = label[i*batchsize:i*batchsize+batchsize]
        y = list(map(lambda x: phoneMap.labeltrans(x), y))
        y = torch.tensor(y)
        batch.append(y)
        dataloader.append(batch)
    if rem != 0 and rem != 1:
        batch = []
        x = data[data_size-rem*size[1]: data_size]
        x = torch.FloatTensor(x).view(rem, 1, stack_w, -1)
        x = x/3
        batch.append(x)
        y = label[size[0]-rem : size[0]]
        y = list(map(lambda x: phoneMap.labeltrans(x), y))
        y = torch.tensor(y)
        batch.append(y)
        dataloader.append(batch)

    return size, dataloader, last_batch

class triphoneMap(object):
    def __init__(self, slist, phone):
        dict0 = {}
        dict1 = {}
        dict2 = {}
        dict3 = {}
        i = 0
        with open(slist, 'r') as f:
            for line in f.readlines():
                trueID = int(line.split(' ')[1].strip('\n'))
                dict0[trueID] = line.split(' ')[0]
                if line.split('_')[0] == phone:
                    fakeID = i
                    dict1[line.split(' ')[0]] = trueID
                    dict2[trueID] = fakeID
                    dict3[fakeID] = line.split(' ')[0]
                    i += 1
        self.id2states = dict0
        self.states = dict1
        self.t2f = dict2
        self.f2states = dict3

    def labeltrans(self, truelabel):
        # transform actual label to new ordered label
        labelID = self.t2f[truelabel]
        return labelID

    def nlabels(self):
        # compute the number triphone states
        return len(self.states)

    def state2label(self, phonestate):
        # return fake label of the input triphone states
        return self.t2f[self.states[phonestate]]


def statemap(map):
    sdict = {}
    with open(map, 'r') as f:
        for line in f.readlines():
            temp = line.strip('\n')
            temp = temp.split(' ')
            if temp[1] not in sdict:
                sdict[temp[1]] = temp[0]
    sdict['sil_s2'] = 'sil[2]'
    sdict['sil_s3'] = 'sil[3]'
    sdict['sil_s4'] = 'sil[4]'
    return sdict


def computeprior(s):
    # For single GAN or Embedding system
    last_batch = False
    priorlist = {}
    quantity = 0
    while not last_batch:
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
        if size[0] < 1024:
            s.stdout.readline()
            s.stdout.readline()
            last_batch = True

        quantity += size[0]
        for lab in label:
            if lab not in priorlist:
                priorlist[lab] = 1
            else:
                priorlist[lab] += 1


    for key in priorlist.keys():
        priorlist[key] /= quantity

    return priorlist
