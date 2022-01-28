import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.uniform import Uniform
import torch.optim as optim
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from helper import *
from sklearn.model_selection import KFold
from scipy.special import logit
from plot_helper import *
from MAP_loss import *
from concrete import *
import argparse
import re
from data_gen import *
import copy
import datetime
import sys

class Model(nn.Module):
    def __init__(self, met_locs, microbe_locs, temp_scheduled = 1, num_local_clusters = 6, K = 2, beta_var = 16.,
                 seed = 0, tau_transformer = 1, meas_var = 1, L = 3, cluster_per_met_cluster = 1,
                 compute_loss_for = ['alpha','beta','w','z','mu_bug','r_bug','pi_bug','mu_met','r_met','pi_met'],
                 learn_num_met_clusters = False, learn_num_bug_clusters = False, linear = True):
        super(Model, self).__init__()
        self.num_local_clusters = num_local_clusters
        self.beta_var = beta_var
        self.mu_var_met = (4/met_locs.shape[1])*np.sum(np.var(met_locs, 0))
        self.mu_var_bug = (4/microbe_locs.shape[1])*np.sum(np.var(microbe_locs, 0))
        self.meas_var = meas_var
        self.compute_loss_for = compute_loss_for
        self.MAPloss = MAPloss(self)
        self.temp_scheduled = temp_scheduled
        self.met_locs = met_locs
        self.microbe_locs = microbe_locs
        self.embedding_dim = met_locs.shape[1]
        self.seed = seed
        self.tau_transformer = tau_transformer
        self.learn_num_met_clusters = learn_num_met_clusters
        self.learn_num_bug_clusters = learn_num_bug_clusters
        self.linear = linear

        self.L, self.K = L, K
        self.L_sm, self.K_sm = L, K
        if self.learn_num_met_clusters:
            self.K = met_locs.shape[0]-1
            self.K_sm = K
        if self.learn_num_bug_clusters:
            self.L = microbe_locs.shape[0] - 1
            self.L_sm = L

        if not self.linear:
            self.NAM = [[nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16,1)
            ) for l in np.arange(self.L)] for k in np.arange(self.K)]

        self.alpha_loc = 1/(self.L_sm*self.K_sm)
        range = np.array([np.max(self.met_locs[:, d]) - np.min(self.met_locs[:, d]) for d in np.arange(self.met_locs.shape[1])])
        self.r_scale_met = np.sqrt(np.sum(range**2)) / (self.K_sm * 2)

        range = np.array([np.max(self.microbe_locs[:, d]) - np.min(self.microbe_locs[:, d]) for d in np.arange(self.microbe_locs.shape[1])])
        self.r_scale_bug = np.sqrt(np.sum(range**2)) / (self.L_sm * 2)

        self.params = {}
        self.distributions = {}
        self.params['beta'] = {'mean': 0, 'scale': np.sqrt(self.beta_var)}
        self.params['alpha'] = {'loc': self.alpha_loc, 'temp':self.temp_scheduled}
        self.params['mu_met'] = {'mean': 0, 'var': self.mu_var_met}
        self.params['mu_bug'] = {'mean': 0, 'var': self.mu_var_bug}
        self.params['r_bug'] = {'dof': 2, 'scale': self.r_scale_bug}
        self.params['r_met'] = {'dof': 2, 'scale': self.r_scale_met}
        self.params['e_met'] = {'dof': 10, 'scale': 10*self.K}
        self.distributions['beta'] = Normal(self.params['beta']['mean'], self.params['beta']['scale'])
        self.distributions['alpha'] = BinaryConcrete(self.params['alpha']['loc'], self.params['alpha']['temp'])
        self.distributions['mu_met'] = MultivariateNormal(torch.zeros(self.embedding_dim), self.params['mu_met']['var']*torch.eye(self.embedding_dim))
        self.distributions['mu_bug'] = MultivariateNormal(torch.zeros(self.embedding_dim), self.params['mu_bug']['var']*torch.eye(self.embedding_dim))
        self.distributions['r_bug'] = Gamma(self.params['r_bug']['dof'], self.params['r_bug']['scale'])
        self.distributions['r_met'] = Gamma(self.params['r_met']['dof'], self.params['r_met']['scale'])


        self.params['pi_met'] = {'epsilon': [2]*self.K}
        # self.params['pi_bug'] = {'epsilon': [2]*self.L}
        self.distributions['pi_met'] = Dirichlet(torch.Tensor(self.params['pi_met']['epsilon']))
        self.distributions['e_met'] = Gamma(10, 10*self.K)
        # self.distributions['pi_bug'] = Dirichlet(torch.Tensor(self.params['pi_bug']['epsilon']))
        self.range_dict = {}
        for param, dist in self.distributions.items():
            sampler = dist.sample([100])
            if len(sampler.shape)>1:
                sampler = sampler[:,0]
            range = sampler.max() - sampler.min()
            self.range_dict[param] = (sampler.min() - range * 0.1, sampler.max() + range * 0.1)
            if 'r_met' in param:
                self.range_dict[param] = (-0.1, np.mean(np.sqrt(self.met_locs[:,0]**2 + self.met_locs[:,1]**2)))
            if 'r_bug' in param:
                self.range_dict[param] = (-0.1, np.mean(np.sqrt(self.microbe_locs[:,0]**2 + self.microbe_locs[:,1]**2)))

        self.range_dict['w'] = (-0.1,1.1)
        self.range_dict['z'] = (-0.1,1.1)
        self.range_dict['alpha'] = (-0.1, 1.1)
        self.range_dict['beta[1:,:]*alpha'] = self.range_dict['beta']
        self.initialize(self.seed)


    def initialize(self, seed):
        torch.manual_seed(seed)
        self.initializations = {}
        self.initializations['beta'] = Normal(0,4)
        self.initializations['alpha'] = Normal(0,1)
        self.initializations['mu_met'] = self.met_locs
        self.initializations['mu_bug'] = self.microbe_locs
        self.initializations['z'] = Normal(0,1)
        self.initializations['w'] = Normal(0,1)
        self.initializations['pi_met'] = torch.ones(self.K)/self.K
        self.beta = nn.Parameter(self.initializations['beta'].sample([self.L+1, self.K]), requires_grad=True)

        self.alpha = nn.Parameter(self.initializations['alpha'].sample([self.L, self.K]), requires_grad=True)
        self.alpha_act = torch.sigmoid(self.alpha/self.tau_transformer)

        ix = np.random.choice(range(len(self.initializations['mu_bug'])), self.L, replace = False)
        self.mu_bug = nn.Parameter(torch.Tensor(self.microbe_locs[ix,:]), requires_grad=True)
        r_temp = self.r_scale_bug*torch.ones(self.L)
        # if self.learn_num_bug_clusters:
        #     l_remove = self.L - self.L_sm
        #     ix_remove = np.random.choice(range(self.L), int(l_remove), replace=False)
        #     r_temp[ix_remove,:] = 1e-4

        self.r_bug = nn.Parameter(torch.log(r_temp.squeeze()), requires_grad=True)
        # self.w_act = torch.sigmoid(self.w/self.tau_transformer)

        ix = np.random.choice(range(len(self.initializations['mu_met'])), self.K, replace = False)
        self.mu_met = nn.Parameter(torch.Tensor(self.met_locs[ix,:]), requires_grad = True)
        r_temp = self.r_scale_met*torch.ones((self.K)).squeeze()

        if self.learn_num_met_clusters:
            # k_remove = self.K - self.K_sm
            # ix_remove = np.random.choice(range(self.K), int(k_remove), replace=False)
            # r_temp[ix_remove] = 1e-4

            self.e_met = nn.Parameter(torch.log(self.initializations['pi_met'].unsqueeze(0)), requires_grad=True)
            self.pi_met = nn.Parameter(Dirichlet(self.initializations['pi_met'].unsqueeze(0)).sample(), requires_grad=True)
        else:
            self.e_met = torch.log(self.initializations['pi_met'].unsqueeze(0))
            self.pi_met = nn.Parameter(self.initializations['pi_met'].unsqueeze(0), requires_grad=True)
        self.r_met = nn.Parameter(torch.log(r_temp), requires_grad = True)

        z_temp = self.initializations['z'].sample([self.met_locs.shape[0], self.K])
        # if self.learn_num_met_clusters:
        #     z_temp[:,ix_remove] = -10
        self.z = nn.Parameter(z_temp, requires_grad=True)
        self.z_act = torch.softmax(self.z/self.tau_transformer, 1)

        kappa = torch.stack([((self.mu_bug - torch.tensor(self.microbe_locs[m, :])).pow(2)).sum(-1) for m in
                             range(self.microbe_locs.shape[0])])
        self.w = torch.sigmoid((self.r_bug - kappa))

    #@profile
    def forward(self, x, y):
        kappa = torch.stack([torch.sqrt(((self.mu_bug - torch.tensor(self.microbe_locs[m,:])).pow(2)).sum(-1)) for m in
                             np.arange(self.microbe_locs.shape[0])])
        epsilon = self.temp_scheduled / 4
        self.w = (1-2*epsilon) * torch.sigmoid((torch.exp(self.r_bug) - kappa)/self.tau_transformer) + epsilon
        g = x@self.w.float()

        self.alpha_act = (1-2*epsilon)*torch.sigmoid(self.alpha/self.tau_transformer) + epsilon
        self.z_act = (1-2*epsilon)*torch.softmax(self.z / self.tau_transformer, 1) + epsilon

        if self.linear:
            out_clusters = self.beta[0,:] + torch.matmul(g, self.beta[1:,:]*self.alpha_act)
        else:
            out_clusters = self.beta[0,:] + torch.cat([torch.cat([self.alpha_act[l,k]*self.NAM[l][k](g[:,l:l+1])
                                                  for l in np.arange(self.L)],1).sum(1).unsqueeze(1) for k in np.arange(self.K)],1)
        net.meas_var = np.var(out_clusters.detach().numpy().flatten()/10)
        loss = self.MAPloss.compute_loss(out_clusters,y)
        return out_clusters, loss

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("-learn", "--learn", help="params to learn", type=str, nargs='+', default = 'all')
    parser.add_argument("-lr", "--lr", help="params to learn", type=float, default = 0.001)
    parser.add_argument("-fix", "--fix", help="params to fix", type=str, nargs='+')
    parser.add_argument("-priors", "--priors", help="priors to set", type=str, nargs='+', default = 'all')
    parser.add_argument("-case", "--case", help="case", type=str, default = datetime.date.today().strftime('%m %d %Y').replace(' ','-'))
    parser.add_argument("-N_met", "--N_met", help="N_met", type=int, default = 10)
    parser.add_argument("-N_bug", "--N_bug", help="N_bug", type=int, default = 10)
    parser.add_argument("-L", "--L", help="number of microbe rules", type=int, default = 2)
    parser.add_argument("-K", "--K", help="metab clusters", type=int, default = 2)
    parser.add_argument("-N_nuisance", "--N_nuisance", help="N_nuisance", type=int, default = 0)
    parser.add_argument("-meas_var", "--meas_var", help="measurment variance", type=float, default = 0.01)
    parser.add_argument("-prior_meas_var", "--prior_meas_var", help = "prior measurment variance", type = float, default = 1e6)
    parser.add_argument("-iterations", "--iterations", help="number of iterations", type=int,default = 20001)
    parser.add_argument("-seed", "--seed", help = "seed for random start", type = int, default = 5)
    parser.add_argument("-load", "--load", help="0 to not load model, 1 to load model", type=int, default = 0)
    parser.add_argument("-rep_clust", "--rep_clust", help = "whether or not bugs are in more than one cluster", default = 0)
    parser.add_argument("-lb", "--lb", help = "whether or not to learn bug clusters", type = int, default = 0)
    parser.add_argument("-lm", "--lm", help = "whether or not to learn metab clusters", type = int, default = 0)
    parser.add_argument("-N_samples", "--N_samples", help="num of samples", type=int,
                        default=1000)
    parser.add_argument("-linear", "--linear", type = int, default = 1)
    args = parser.parse_args()

    if 'none' in args.priors:
        args.priors = []

    params2learn = args.learn
    priors2set = args.priors
    temp_scheduled = 'scheduled'
    temp_transformer = 0.1
    path = 'results_MAP/'

    path = path + args.case.replace(' ','_')
    if not os.path.isdir(path):
        os.mkdir(path)

    if 'all' not in params2learn or 'all' not in priors2set:
        path = path + '/learn_' + '_'.join(params2learn) + '-priors_' + '_'.join(priors2set) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)

    info = 'lr' + str(args.lr) + '-linear' + str(args.linear) + '-lm' + str(args.lm) + '-lb' + str(args.lb) +'-N_bug' + str(args.N_bug) + \
           '-N_met' + str(args.N_met) + '-L' + str(args.L) + '-K' + str(args.K)
    path = path + '/' + info +'/'

    if not os.path.isdir(path):
        os.mkdir(path)

    if 'all' in priors2set:
        priors2set = ['z','alpha','beta','mu_bug','mu_met','r_bug','r_met','pi_met','e_met']
        if args.lm == 0:
            priors2set.remove('e_met')
        if args.fix:
            params2learn = priors2set.copy()
            for p in args.fix:
                priors2set.remove(p)
                params2learn.remove(p)

    # print(params2learn)
    # print(priors2set)
    x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
    mu_met, r_bug, r_met, gen_u = generate_synthetic_data(
        N_met = args.N_met, N_bug = args.N_bug, N_met_clusters = args.K,
        N_bug_clusters = args.L,meas_var = args.meas_var,
        repeat_clusters= args.rep_clust, N_samples=args.N_samples, linear = args.linear)
    plot_syn_data(path, x, y, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
                  r_bug, mu_met, r_met, gen_u)

    if args.lm:
        r_met = np.append(r_met, np.zeros(args.N_met-1-len(r_met)))
        gen_z = np.hstack((gen_z, np.zeros((args.N_met, args.N_met - 1 - args.K))))
        mu_met = np.vstack((mu_met, np.zeros((args.N_met - args.K - 1, mu_met.shape[1]))))
        gen_beta = np.hstack((gen_beta, np.zeros((gen_beta.shape[0], args.N_met - args.K - 1))))
        gen_alpha = np.hstack((gen_alpha, np.zeros((gen_alpha.shape[0], args.N_met - args.K - 1))))
    if args.lb:
        r_bug = np.append(r_bug, np.zeros(args.N_bug - 1 - len(r_bug)))
        gen_w = np.hstack((gen_w, np.zeros((args.N_bug, args.N_bug - 1 - args.L))))
        gen_u = np.hstack((gen_u, np.zeros((args.N_bug, args.N_bug - 1 - args.L))))
        mu_bug = np.vstack((mu_bug, np.zeros((args.N_bug - args.L - 1, mu_bug.shape[1]))))
        gen_beta = np.vstack((gen_beta, np.zeros((args.N_bug - args.L - 1, gen_beta.shape[1]))))
        gen_alpha = np.vstack((gen_alpha, np.zeros((args.N_bug - args.L - 1, gen_alpha.shape[1]))))

    true_vals = {'y':y, 'beta':gen_beta, 'alpha':gen_alpha, 'mu_bug': mu_bug,
                 'mu_met': mu_met, 'u': gen_u, 'beta[1:,:]*alpha': gen_beta[1:,:]*gen_alpha,
                 'r_bug': r_bug, 'r_met': r_met, 'z': gen_z, 'w': gen_w, 'pi_met':np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0),
                 'pi_bug':np.expand_dims(np.sum(gen_w,0)/np.sum(np.sum(gen_w)),0), 'bug_locs': gen_bug_locs, 'met_locs':gen_met_locs,
                 'e_met': np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0)}

    print(priors2set)
    net = Model(gen_met_locs, gen_bug_locs, K=args.K, L=args.L,
                tau_transformer=temp_transformer, meas_var = args.prior_meas_var, compute_loss_for=priors2set,
                learn_num_bug_clusters=args.lb,
                learn_num_met_clusters=args.lm, linear = args.linear==1)
    net.to(device)

    for param, dist in net.distributions.items():
        parameter_dict = net.params[param]
        plot_distribution(dist, param, true_val = true_vals[param], ptype = 'prior', path = path, **parameter_dict)

    fig_dict4, ax_dict4 = {},{}
    fig_dict5, ax_dict5 = {},{}
    param_dict = {}
    tau_logspace = np.logspace(-0.5, -4, args.iterations)

    net.temp_scheduled = tau_logspace[0]
    param_dict[args.seed] = {}

    net.initialize(args.seed)

    lr_list = []
    beta_range = np.abs(net.range_dict['beta'][-1] - net.range_dict['beta'][0])
    start = 0
    for name, parameter in net.named_parameters():
        if name not in params2learn and 'all' not in params2learn:
            setattr(net, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))
            parameter = nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False)
        elif name == 'z' or name == 'alpha':
            parameter = getattr(net, name + '_act')
        elif name == 'r_bug' or name == 'r_met' or name == 'e_met':
            parameter = torch.exp(parameter)
        elif name == 'pi_met':
            parameter = torch.softmax(parameter,1)
        param_dict[args.seed][name] = [parameter.clone().detach().numpy()]
        range = np.abs(net.range_dict[name][-1] - net.range_dict[name][0])
        lr_list.append({'params': parameter, 'lr': (args.lr/beta_range)*range})
    param_dict[args.seed]['w'] = [net.w.clone().detach().numpy()]
    param_dict[args.seed]['beta[1:,:]*alpha'] = [net.beta[1:,:].clone().detach().numpy()*net.alpha.clone().detach().numpy()]
    loss_vec = []
    train_out_vec = []

    optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
    # optimizer = optim.RMSprop(lr_list, lr=args.lr)

    files = os.listdir(path)
    epochs = re.findall('epoch\d+', ' '.join(os.listdir(path)))
    path_orig = path
    if len(epochs)>0:
        if os.path.isfile(path_orig + 'seed' + str(args.seed) + '.txt'):
            with open(path_orig + 'seed' + str(args.seed) + '.txt', 'r') as f:
                largest = int(f.readlines()[0])
        else:
            largest = max([int(num.split('epoch')[-1]) for num in epochs])
        foldername = path + 'epoch' + str(largest) + '/'
        if 'seed' + str(args.seed) + '_checkpoint.tar' in os.listdir(foldername) and args.load==1:
            checkpoint = torch.load(foldername + 'seed' + str(args.seed) + '_checkpoint.tar')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = int(checkpoint['epoch'] - 1)
            # tau_logspace = np.concatenate((tau_logspace,np.logspace(-2, -5, int(iterations / 100))))
            ix = int(checkpoint['epoch']-100)
            if ix >= len(tau_logspace):
                ix = -1
            net.temp_scheduled = tau_logspace[ix]
            if args.iterations <= start:
                print('training complete')
                sys.exit()
            print('model loaded')
        else:
            print('no model loaded')
    else:
        print('no model loaded')

    x = torch.Tensor(x).to(device)
    # cluster_targets = np.stack([y[:,np.where(gen_z[:,i]==1)[0][0]] for i in np.arange(gen_z.shape[1])]).T
    timer = []
    end_learning = False
    tau_vec = []
    alpha_tau_vec = []
    lowest_loss_vec = []
    loss_dict_vec = []
    ix = 0
    # grad_dict = {}
    for epoch in np.arange(start, args.iterations):
        if isinstance(temp_scheduled, str):
            if epoch>100:
                ix = int(epoch-100)
                if ix >= len(tau_logspace):
                    ix = -1
                net.temp_scheduled = tau_logspace[ix]
            tau_vec.append(net.temp_scheduled)
        optimizer.zero_grad()
        cluster_outputs, loss = net(x, torch.Tensor(y))

        train_out_vec.append(cluster_outputs)
        loss_vec.append(loss.item())
        loss_dict_vec.append(copy.copy(net.MAPloss.loss_dict))
        loss.backward()

        lr_list = []
        beta_range = np.abs(np.max(net.beta.grad.view(-1).numpy()) - np.min(net.beta.grad.view(-1).numpy()))
        if epoch == start:
            for name, parameter in net.named_parameters():
                if parameter.grad is not None:
                    range = np.abs(np.max(parameter.grad.view(-1).numpy()) - np.min(parameter.grad.view(-1).numpy()))
                    lr_list.append({'params': parameter, 'lr': (args.lr / beta_range) * range})
            optimizer = optim.RMSprop(lr_list, lr=args.lr)

        optimizer.step()

        lr_list = []
        beta_range = np.abs(np.max(net.beta.grad.view(-1).numpy()) - np.min(net.beta.grad.view(-1).numpy()))
        for name, parameter in net.named_parameters():
            # if parameter.grad is not None:
            #     if name not in grad_dict.keys():
            #         grad_dict[name] = []
            #     grad_dict[name].append(parameter.grad.view(-1).numpy())
            if name == 'z' or name == 'alpha':
                parameter = getattr(net, name + '_act')
            elif name == 'r_bug' or name == 'r_met' or name == 'e_met':
                parameter = torch.exp(parameter)
            elif name == 'pi_met':
                parameter = torch.softmax(parameter, 1)
            param_dict[args.seed][name].append(parameter.clone().detach().numpy())
        param_dict[args.seed]['w'].append(net.w.clone().detach().numpy())
        param_dict[args.seed]['beta[1:,:]*alpha'].append(
            net.beta[1:, :].clone().detach().numpy() * net.alpha.clone().detach().numpy())


        if (epoch%1000 == 0 and epoch != 0):
            print('Epoch ' + str(epoch) + ' Loss: ' + str(loss_vec[-1]))
            print('Tau: ' + str(net.temp_scheduled))
            print('')

            last_mod = -1
            mapping = {}
            mapping['bug'] = unmix_clusters(true_vals['mu_bug'], param_dict[args.seed]['mu_bug'][last_mod], param_dict[args.seed]['r_bug'][last_mod],
                                            true_vals['bug_locs'])
            mapping['bug'] = pd.Series(mapping['bug']).sort_index()
            mapping['met'] = unmix_clusters(true_vals['mu_met'], param_dict[args.seed]['mu_met'][last_mod], param_dict[args.seed]['r_met'][last_mod],
                                            true_vals['met_locs'])
            mapping['met'] = pd.Series(mapping['met']).sort_index()

            if 'epoch' not in path:
                path = path + 'epoch' + str(epoch) + '/'
            else:
                path = path.split('epoch')[0] + 'epoch' + str(epoch) + '/'
            if not os.path.isdir(path):
                os.mkdir(path)

            # for parameter in grad_dict.keys():
            #     fig, ax = plt.subplots()
            #     for i in range(len(grad_dict[parameter])):
            #         ax.scatter([i]*len(grad_dict[parameter][i]), grad_dict[parameter][i], c='g')
            #     ax.set_xlabel('Epochs')
            #     ax.set_ylabel('Parameter gradient')
            #     ax.set_title(parameter)
            #     fig.savefig(path + 'seed' + str(args.seed) + '-' + parameter + '_grad.pdf')
            #     plt.close(fig)
            #
            # grad_dict = {}
            plot_param_traces(path, param_dict[args.seed], params2learn, true_vals, net, args.seed, mapping)
            fig3, ax3 = plt.subplots(figsize=(8, 8))
            fig3, ax3 = plot_loss(fig3, ax3, args.seed, epoch + 1 - start, loss_vec, lowest_loss=None)
            fig3.tight_layout()
            fig3.savefig(path_orig + 'loss_seed_' + str(args.seed) + '.pdf')
            plt.close(fig3)

            # fig3, ax3 = plt.subplots(len(net.MAPloss.loss_dict.keys()
            #                              ), 1, figsize=(8, 8 * len(net.MAPloss.loss_dict.keys())))
            # it = 0
            # for key, loss_val in net.MAPloss.loss_dict.items():
            #     vals = [ldv[key].item() for ldv in loss_dict_vec]
            #     ax3[it].plot(range(start, epoch + 1), vals)
            #     ax3[it].set_title(key)
            #     ax3[it].set_xlabel('Epochs')
            #     ax3[it].set_ylabel('Loss')
            #     it += 1
            # fig3.tight_layout()
            # fig3.savefig(path_orig + 'lossdict_seed_' + str(args.seed) + '.pdf')
            # plt.close(fig3)

            plot_output_locations(path, net, last_mod, param_dict[args.seed], args.seed, gen_u, mapping, type='last_train')
            plot_output(path, path_orig, last_mod, train_out_vec, y, gen_z, param_dict[args.seed],
                                 args.seed, mapping, type = 'last_train')

            if 'beta' not in params2learn:
                mapping['met'] = {i:i for i in np.arange(net.z.shape[1])}
                mapping['met'] = pd.Series(mapping['met']).sort_index()
                mapping['bug'] = {i:i for i in np.arange(net.w.shape[1])}
                mapping['bug'] = pd.Series(mapping['bug']).sort_index()
            plot_xvy(path, net, x, train_out_vec, last_mod, param_dict, gen_bug_locs, args.seed, mapping)

            # if args.n_local_clusters > 1:
            #     plot_rules_detectors_tree(path, net, last_mod, param_dict[args.seed], gen_bug_locs, args.seed)
            if isinstance(temp_scheduled, str) and len(tau_vec) > 0:
                fig, ax = plt.subplots()
                ax.semilogy(np.arange(start, epoch+1), tau_vec)
                fig.savefig(path + 'seed' + str(args.seed) + '_tau_scheduler.pdf')
                plt.close(fig)

            with open(path_orig + 'seed' + str(args.seed) + '.txt', 'w') as f:
                f.writelines(str(epoch))

            torch.save({'model_state_dict':net.state_dict(),
                       'optimizer_state_dict':optimizer.state_dict(),
                       'epoch': epoch},
                       path + 'seed' + str(args.seed) + '_checkpoint.tar')

