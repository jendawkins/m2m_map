import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import argparse

class ExU(nn.Module):
    def __init__(self, size):
        super(ExU, self).__init__()
        self.w = nn.Parameter(Normal(0,0.5).sample([size]), requires_grad=True)

    def forward(self, x):
        return x * torch.exp(self.w)


class NAM(nn.Module):
    def __init__(self, hidden_sizes=[(1,32), (32,32),(32,16),(16,1)], N_mult = 1, L = 6, K=6):
        super(NAM, self).__init__()
        self.N_mult = N_mult
        self.L = L
        self.K = K

        l_list = []
        for h in hidden_sizes:
            l_list.append(nn.Linear(h[0], h[1], bias = True))
            if h[-1] != 1:
                l_list.append(nn.ReLU())
        sequential = nn.Sequential(*l_list)
        self.model = nn.ModuleList([nn.ModuleList([nn.ModuleList([sequential for p in
                                                                  np.arange(N_mult)]) for l in
                                                   np.arange(L)]) for k in np.arange(K)])

        # self.bias = nn.Parameter(Normal(0,1).sample([1,K]), requires_grad=True)
        # self.alpha = nn.Parameter(Normal(0,1).sample([L,K]), requires_grad=True)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean = 0, std = 0.5)
            # m.bias.data.fill_(0.01)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()


    def forward(self, x):
        # alpha = torch.sigmoid(self.alpha)
        out = torch.cat([torch.cat([torch.stack([self.model[k][l][p](x[:,l].unsqueeze(-1)) for
                                                                      p in np.arange(self.N_mult)],-1).sum(-1)
                                     for l in np.arange(self.L)],-1).sum(-1).unsqueeze(-1) for k in np.arange(self.K)],1)
        return out


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type",'--type', default = 'sigmoid', type = str)
    parser.add_argument("-p", '--p', default=1, type=int)
    parser.add_argument("-use_l1", '--use_l1', default=0, type=int)
    args = parser.parse_args()


    L = 1
    K = 1
    P = args.p
    hidden = [(1,32),(32,32),(32,16),(16,1)]
    iterations =5001
    lr = 0.0001

    path = 'NAM_results/RELUwBias_2hidden-' + args.type + '-p' + str(P)
    if not os.path.isdir(path):
        os.mkdir(path)

    torch.manual_seed(0)
    x = Normal(0,10000).sample([1000,L])
    x = (x - torch.mean(x)) / (torch.std(x - torch.mean(x)))
    if args.type == 'exp':
        temp = -torch.exp(x)
    if args.type == 'sin':
        temp = -torch.sin(x)
    if args.type == 'sigmoid':
        temp = -torch.sigmoid(x)
    if args.type == 'poly':
        temp = -((x).pow(4) + (x).pow(3))
    if args.type == 'linear':
        temp = x
    bias = Normal(0,0.0001).sample([1,K])
    # alpha = np.array([[0,1,0,0,0],
    #                   [1,0,0,0,0],
    #                   [0,0,0,1,0],
    #                   [0,0,1,0,1]])

    net = NAM(hidden_sizes=hidden, N_mult = P, L= L, K=K)
    optimizer = optim.RMSprop(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
    criterion = nn.MSELoss()

    y = bias + torch.stack([temp for p in np.arange(P)],-1).sum(-1) + Normal(0,0.0001).sample([1000,K])
    y = (y - torch.mean(y)) / (torch.std(y - torch.mean(y)))

    fig, ax = plt.subplots(L, K, figsize=(8 * K, 8 * L))
    # for l in np.arange(L):
        # for k in np.arange(K):
    ax.scatter(x.detach().numpy(), y.numpy(), c='r', label='True')
    ax.legend()
    fig.savefig( path + '/data_gen.pdf')

    lossvec = []
    for epoch in np.arange(iterations):
        optimizer.zero_grad()
        out = net(x)
        l1_parameters = []
        loss = criterion(out, y)
        if args.use_l1:
            for parameter in net.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = net.compute_l1_loss(torch.cat(l1_parameters))
            loss += l1

        lossvec.append(loss.item())
        loss.backward()

        optimizer.step()
        scheduler.step()

        if epoch % 1000 == 0:
            fig, ax = plt.subplots()
            ax.plot(lossvec)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Loss')
            fig.savefig(path + '/'+ str(epoch) + 'loss.pdf')
            plt.close(fig)

            fig, ax = plt.subplots(L,K, figsize = (8*K, 8*L))
            # for l in np.arange(L):
                # for k in np.arange(K):
            ax.scatter(x.detach().numpy(), y.numpy(), c='r', label = 'True')
            ax.scatter(x.detach().numpy(), out.detach().numpy(), c = 'b', label = 'Guess')
            ax.legend()

            fig.savefig(path + '/' + str(epoch) + 'x_v_y.pdf')
            plt.close(fig)