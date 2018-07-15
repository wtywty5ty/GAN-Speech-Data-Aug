import torch
import torch.nn as nn
import torch.optim as optim

class embedNet(nn.Module):
    def __init__(self, nclasses):
        super(embedNet, self).__init__()
        self.emb = nn.Embedding(nclasses, 20)
        self.li = nn.Sequential(
            nn.Linear(20, nclasses),
        )

    def forward(self, input):
        output = self.emb(input)
        output = self.li(output)
        return output


class embedding(object):
    def __init__(self, net, nclasses, batchsize, nepochs):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.device = device
        self.batchsize = batchsize
        self.nepochs = nepochs
        self.nclasses = nclasses

        # data

        # nets
        self.net = net(nclasses).to(device)

        # criteria
        self.criteria = nn.CrossEntropyLoss().to(device)

        # solver
        self.optimizer = optim.Adam(self.net.parameters())

    def train(self):
        nepochs = self.nepochs
        batchsize = self.batchsize
        device = self.device
        for epoch in range(nepochs):
            x = (torch.rand(batchsize, 1) * self.nclasses).type(torch.long).squeeze().to(device)
            #y = x.squeeze()
            output = self.net(x)
            err = self.criteria(output, x)
            err.backward()
            self.optimizer.step()

            if epoch%100 == 0:
                print('[%d/%d] loss: %.4f'%(epoch, nepochs, err.item()))

        torch.save(self.net, 'out/embedNet.pkl')


if __name__ == '__main__':
    embedding(embedNet, 808, 1024, 2000).train()
    net = torch.load('out/embedNet.pkl')
    input = torch.LongTensor([1, 2, 3, 4, 5]).cuda()
    print(net.emb(input))
