import struct
import numpy as np


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

def dataFilter(s, id):
    rows = s.stdout.read(4)
    rows = struct.unpack('i', rows)[0]
    columns = s.stdout.read(4)
    columns = struct.unpack('i', columns)[0]
    idx_list = []
    for i in range(rows):
        results = s.stdout.read(columns*4)
        results = struct.unpack('%df' % columns, results)
        if results.index(max(results)) != id:
            idx_list.append(i)
    return idx_list

def dataFilterPlevel(s, id, phonemap):
    state = phonemap.id2sates[id]
    targetphone = state.split('_')[0]
    rows = s.stdout.read(4)
    rows = struct.unpack('i', rows)[0]
    columns = s.stdout.read(4)
    columns = struct.unpack('i', columns)[0]
    idx_list = []
    for i in range(rows):
        results = s.stdout.read(columns * 4)
        results = struct.unpack('%df' % columns, results)
        cls = results.index(max(results))
        if phonemap.id2sates[cls].split('_')[0] != targetphone:
            idx_list.append(i)
    return idx_list

def genLabel(s, phonemap):
    idx = {}
    splitSize = {}
    for fid in range(phonemap.nlabels()):
        idx[fid] = []

    rows = s.stdout.read(4)
    rows = struct.unpack('i', rows)[0]
    columns = s.stdout.read(4)
    columns = struct.unpack('i', columns)[0]
    for row in range(rows):
        results = s.stdout.read(columns * 4)
        results = struct.unpack('%df' % columns, results)
        tid_pool = [phonemap.f2t[i] for i in range(phonemap.nlabels())]
        score_pool = [results[tid] for tid in tid_pool]
        cls_fid = score_pool.index(max(score_pool))
        idx[cls_fid].append(row)

    index = []
    for fid in range(phonemap.nlabels()):
        index += idx[fid]
        splitSize[fid] = len(idx[fid])
        if splitSize[fid] == 0:
            print('warning: one state gets 0 sample')

    return index, splitSize

def genLabelFilter(s, phonemap):
    idx = {}
    splitSize = {}
    for fid in range(phonemap.nlabels()):
        idx[fid] = []

    rows = s.stdout.read(4)
    rows = struct.unpack('i', rows)[0]
    columns = s.stdout.read(4)
    columns = struct.unpack('i', columns)[0]
    for row in range(rows):
        results = s.stdout.read(columns * 4)
        results = struct.unpack('%df' % columns, results)
        tid_pool = [phonemap.f2t[i] for i in range(phonemap.nlabels())]
        cls_tid = results.index(max(results))
        if cls_tid not in tid_pool:
            continue
        cls_fid = phonemap.t2f[cls_tid]
        idx[cls_fid].append(row)

    index = []
    for fid in range(phonemap.nlabels()):
        index += idx[fid]
        splitSize[fid] = len(idx[fid])
        if splitSize[fid] == 0:
            print('warning: one state gets 0 sample')

    return index, splitSize


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