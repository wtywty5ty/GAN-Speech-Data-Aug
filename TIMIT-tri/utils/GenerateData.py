import struct
import torch


def generateData(generator, batchsize, nclass, class_id):
    noise = torch.randn(batchsize, 100).cuda()
    y = torch.zeros(batchsize, nclass).cuda()
    y[:, class_id] = 1

    gen_data = generator(torch.cat([noise, y], 1)).squeeze()
    gen_data = 3 * gen_data

    rows = gen_data.size(0)
    columns = gen_data.size(1) * gen_data.size(2)

    flat_data = gen_data.cpu().view(-1).detach().numpy()
    body = flat_data.astype('f').tostring()

    header = struct.pack('2i', rows, columns)

    #buf = struct.pack('2i%df' % len(flat_data), rows, columns, *flat_data)

    return header+body, gen_data


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

    buf = struct.pack('2i%sf' % len(flat_data), rows, columns, *flat_data)

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

    buf = struct.pack('2i%sf' % len(flat_data), rows, columns, *flat_data)

    return buf


def generateDataUncon(generator, batchsize):
    noise = torch.randn(batchsize, 100).cuda()

    gen_data = generator(noise).squeeze()
    gen_data = 3 * gen_data

    rows = gen_data.size(0)
    columns = gen_data.size(1) * gen_data.size(2)

    flat_data = gen_data.cpu().view(-1).detach().numpy()
    body = flat_data.astype('f').tostring()

    header = struct.pack('2i', rows, columns)

    #buf = struct.pack('2i%sf' % len(flat_data), rows, columns, *flat_data)

    return header+body, gen_data


