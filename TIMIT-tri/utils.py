import struct
import numpy as np
import torch
from embedding.train_embedding import embedNet


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
        #y = torch.tensor(y)
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
        #y = torch.tensor(y)
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


def generateData(generator, batchsize, nclass, class_id):
    noise = torch.randn(batchsize, 100).cuda()
    y = torch.zeros(batchsize, nclass).cuda()
    y[:, class_id] = 1

    gen_data = generator(torch.cat([noise,y], 1)).squeeze()
    gen_data = 3 * gen_data

    rows = gen_data.size(0)
    columns = gen_data.size(1) * gen_data.size(2)

    flat_data = gen_data.view(-1)

    buf = struct.pack('2i%df'%len(flat_data), rows, columns, *flat_data)

    return buf


def generateDataEmbd(generator, batchsize, class_id):
    noise = torch.randn(batchsize, 100).cuda()
    y = torch.zeros(batchsize).type(torch.LongTensor).cuda()
    y.fill_(class_id)
    emb = torch.load('embedding/out/embedNet.pkl').emb.cuda()

    y_ = emb(y)

    gen_data = generator(torch.cat([noise, y_], 1)).squeeze()
    gen_data = 3 * gen_data

    rows = gen_data.size(0)
    columns = gen_data.size(1) * gen_data.size(2)

    flat_data = gen_data.view(-1)

    buf = struct.pack('2i%sf'%len(flat_data), rows, columns, *flat_data)

    return buf


def generateData_all(generator, batchsize, nclass):
    noise_temp = torch.randn(batchsize, 100).cuda()
    for id in range(nclass):
        y_temp = torch.zeros(batchsize, nclass).cuda()
        y_temp[:, id] = 1
        if id == 0:
            y = y_temp
            noise = noise_temp
        else:
            y = torch.cat([y, y_temp], 0)
            noise = torch.cat([noise, noise_temp], 0)

    gen_data = generator(torch.cat([noise, y], 1)).squeeze()
    gen_data = 3 * gen_data

    rows = gen_data.size(0)
    columns = gen_data.size(1) * gen_data.size(2)

    flat_data = gen_data.view(-1)

    buf = struct.pack('2i%sf' % len(flat_data), rows, columns, *flat_data)
    
    return buf


def generateDataEmbd_all(generator, batchsize, nclass):
    noise_temp = torch.randn(batchsize, 100).cuda()
    for id in range(nclass):
        y_temp = torch.zeros(batchsize).type(torch.LongTensor).cuda()
        y_temp.fill_(id)
        if id == 0:
            y = y_temp
            noise = noise_temp
        else:
            y = torch.cat([y, y_temp], 0)
            noise = torch.cat([noise, noise_temp], 0)

    gen_data = generator(noise, y).squeeze()
    gen_data = 3 * gen_data

    rows = gen_data.size(0)
    columns = gen_data.size(1) * gen_data.size(2)

    flat_data = gen_data.view(-1)

    buf = struct.pack('2i%sf'%len(flat_data), rows, columns, *flat_data)

    return buf

def generateDataUncon(generator, batchsize):
    noise = torch.randn(batchsize, 100).cuda()

    gen_data = generator(noise).squeeze()
    gen_data = 3 * gen_data

    rows = gen_data.size(0)
    columns = gen_data.size(1) * gen_data.size(2)

    flat_data = gen_data.view(-1)

    buf = struct.pack('2i%sf'%len(flat_data), rows, columns, *flat_data)

    return buf



def testResults(s):
    rows = s.stdout.read(4)
    rows = struct.unpack('i', rows)[0]
    columns = s.stdout.read(4)
    columns = struct.unpack('i',columns)[0]
    results_list = []
    for i in range(rows):
        results = s.stdout.read(columns*4)
        results = struct.unpack('%df'% columns, results)
        results_list.append(results)
    return rows, columns, results_list


def returnMeanStd(results_list):
    mean_list = np.mean(results_list, 0)
    std_list = np.std(results_list, 0)
    return mean_list, std_list

def readPhoneID(list_file):
    phone_id = []
    with open(list_file, 'r') as f:
        for line in f.readlines():
            phone_id.append(line.split(' ')[0])

    return phone_id


def correctness(results, phone_id,topn):
    classification = np.argsort(results, 1)
    correct = 0
    for i in range(len(classification)):
        if phone_id in classification[i][-topn:]:
            correct += 1
    return correct/len(classification)


def correctness_(results, phone_id,topn):
    # used for phone level test
    classification = np.argsort(results, 1)//3
    correct = 0
    for i in range(len(classification)):
        if phone_id in classification[i][-topn:]:
            correct += 1
    return correct/len(classification)

def triTestsetAcc(results, labels, phonemap, topn):
    classification = np.argsort(results, 1)
    correct_dict = {}
    # initialise dict
    for label in phonemap.states.keys():
        correct_dict[label] = [0, 0]
    for i in range(len(classification)):
        correct_dict[phonemap.id2states[labels[i]]][1] +=1 
        if labels[i] in classification[i][-topn:]:
            correct_dict[phonemap.id2states[labels[i]]][0] += 1
    for key in correct_dict.keys():
        ele = correct_dict[key] 
        ele.append(ele[0]/ele[1])
    return correct_dict



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



