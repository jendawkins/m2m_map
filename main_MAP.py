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

class Model(nn.Module):
    def __init__(self, met_locs, microbe_locs, temp_grouper = 1, temp_selector = 1, num_local_clusters = 6, K = 2, beta_var = 16.,
                 seed = 0, tau_transformer = 1, meas_var = 1, L = 3, cluster_per_met_cluster = 1,
                 compute_loss_for = ['alpha','beta','w','z','mu_bug','r_bug','pi_bug','mu_met','r_met','pi_met']):
        super(Model, self).__init__()
        self.num_local_clusters = num_local_clusters
        self.cluster_per_met_cluster = cluster_per_met_cluster
        self.K = K
        self.L = L
        self.beta_var = beta_var
        self.mu_var_met = (4/met_locs.shape[1])*np.sum(np.var(met_locs, 0))
        self.mu_var_bug = (4/microbe_locs.shape[1])*np.sum(np.var(microbe_locs, 0))
        self.meas_var = meas_var
        self.compute_loss_for = compute_loss_for
        self.MAPloss = MAPloss(self)
        self.temp_grouper = temp_grouper
        self.temp_selector = temp_selector
        self.met_locs = met_locs
        self.microbe_locs = microbe_locs
        self.embedding_dim = met_locs.shape[1]
        self.seed = seed
        self.alpha_loc = 1.
        self.tau_transformer = tau_transformer

        range = np.array([np.max(self.met_locs[:, d]) - np.min(self.met_locs[:, d]) for d in np.arange(self.met_locs.shape[1])])
        self.r_scale_met = np.sqrt(np.sum(range**2)) / (self.K * 2)

        range = np.array([np.max(self.microbe_locs[:, d]) - np.min(self.microbe_locs[:, d]) for d in np.arange(self.microbe_locs.shape[1])])
        self.r_scale_bug = np.sqrt(np.sum(range**2)) / (self.L * 2)

        self.params = {}
        self.distributions = {}
        self.params['beta'] = {'mean': 0, 'scale': np.sqrt(self.beta_var)}
        self.params['alpha'] = {'loc': self.alpha_loc, 'temp':self.temp_selector}
        self.params['mu_met'] = {'mean': 0, 'var': self.mu_var_met}
        self.params['mu_bug'] = {'mean': 0, 'var': self.mu_var_bug}
        self.params['r_bug'] = {'dof': 2, 'scale': self.r_scale_bug}
        self.params['r_met'] = {'dof': 2, 'scale': self.r_scale_met}
        self.params['thresh'] = {'low': 10, 'high':1000}
        self.distributions['beta'] = Normal(self.params['beta']['mean'], self.params['beta']['scale'])
        self.distributions['alpha'] = BinaryConcrete(self.params['alpha']['loc'], self.params['alpha']['temp'])
        self.distributions['mu_met'] = MultivariateNormal(torch.zeros(self.embedding_dim), self.params['mu_met']['var']*torch.eye(self.embedding_dim))
        self.distributions['mu_bug'] = MultivariateNormal(torch.zeros(self.embedding_dim), self.params['mu_bug']['var']*torch.eye(self.embedding_dim))
        self.distributions['r_bug'] = Gamma(self.params['r_bug']['dof'], self.params['r_bug']['scale'])
        self.distributions['r_met'] = Gamma(self.params['r_met']['dof'], self.params['r_met']['scale'])

        self.params['pi_met'] = {'epsilon': [2]*self.K}
        # self.params['pi_bug'] = {'epsilon': [2]*self.L}
        self.distributions['pi_met'] = Dirichlet(torch.Tensor(self.params['pi_met']['epsilon']))
        # self.distributions['pi_bug'] = Dirichlet(torch.Tensor(self.params['pi_bug']['epsilon']))
        self.range_dict = {}
        for param, dist in self.distributions.items():
            sampler = dist.sample([100])
            if len(sampler.shape)>1:
                sampler = sampler[:,0]
            range = sampler.max() - sampler.min()
            self.range_dict[param] = (sampler.min() - range * 0.1, sampler.max() + range * 0.1)
            if 'r_met' in param:
                self.range_dict[param] = (0, np.mean(np.sqrt(self.met_locs[:,0]**2 + self.met_locs[:,1]**2)))
            if 'r_bug' in param:
                self.range_dict[param] = (0, np.mean(np.sqrt(self.microbe_locs[:,0]**2 + self.microbe_locs[:,1]**2)))

        self.range_dict['w'] = (-0.1,1.1)
        self.range_dict['z'] = (-0.1,1.1)
        self.range_dict['alpha'] = (-0.1, 1.1)
        self.range_dict['u'] = (-0.1,1.1)
        self.initialize(self.seed)


    def initialize(self, seed):
        torch.manual_seed(seed)
        self.initializations = {}
        self.initializations['beta'] = self.distributions['beta']
        self.initializations['alpha'] = Normal(0,1)
        self.initializations['mu_met'] = self.met_locs
        self.initializations['mu_bug'] = self.microbe_locs
        # self.initializations['r_met'] = self.distributions['r_met']
        # self.initializations['r_bug'] = self.distributions['r_bug']
        # self.initializations['pi_bug'] = self.distributions['pi_bug']
        self.initializations['pi_met'] = self.distributions['pi_met']
        self.initializations['z'] = Normal(0,1)
        self.initializations['w'] = Normal(0,1)
        # beta_dist = Normal(0, np.sqrt(self.beta_var))
        self.beta = nn.Parameter(self.initializations['beta'].sample([self.L+1, self.K]), requires_grad=True)

        self.alpha = nn.Parameter(self.initializations['alpha'].sample([self.L, self.K]), requires_grad=True)
        self.alpha_act = torch.sigmoid(self.alpha / self.temp_selector)

        if self.cluster_per_met_cluster:
            ix = np.concatenate([np.random.choice(range(len(self.initializations['mu_bug'])), self.L, replace = False) for
                  ii in range(self.num_local_clusters*self.K)])
            self.mu_bug = nn.Parameter(torch.Tensor(self.microbe_locs[ix,:]).view(self.L, self.num_local_clusters, self.K, -1).squeeze())
            r_temp = self.r_scale_bug*torch.ones((self.L, self.num_local_clusters, self.K)).squeeze()
            self.r_bug = nn.Parameter(torch.log(1 / r_temp), requires_grad=True)
            if self.num_local_clusters <= 1:
                self.w = self.initializations['w'].sample([self.microbe_locs.shape[0], self.L, self.K])
            else:
                self.w = nn.Parameter(self.initializations['w'].sample([self.L, self.num_local_clusters, self.K]),
                                      requires_grad=True)
        else:
            ix = np.concatenate([np.random.choice(range(len(self.initializations['mu_bug'])), self.L, replace = False) for
                  ii in range(self.num_local_clusters)])
            self.mu_bug = nn.Parameter(torch.Tensor(self.microbe_locs[ix,:]).view(self.L, self.num_local_clusters, -1).squeeze())
            r_temp = self.r_scale_bug*torch.ones((self.L, self.num_local_clusters)).squeeze()
            self.r_bug = nn.Parameter(torch.log(r_temp), requires_grad=True)
            if self.num_local_clusters<=1:
                self.w = self.initializations['w'].sample([self.microbe_locs.shape[0], self.L])
            else:
                self.w = nn.Parameter(self.initializations['w'].sample([self.L, self.num_local_clusters]),
                                      requires_grad=True)

        self.w_act = torch.sigmoid(self.w / self.temp_grouper)

        ix = np.random.choice(range(len(self.initializations['mu_met'])), self.K, replace = False)
        self.mu_met = nn.Parameter(torch.Tensor(self.met_locs[ix,:]), requires_grad = True)
        r_temp = self.r_scale_met*torch.ones((self.K)).squeeze()
        self.r_met = nn.Parameter(torch.log(r_temp), requires_grad = True)

        self.pi_met = nn.Parameter(self.initializations['pi_met'].sample().unsqueeze(0), requires_grad=True)

        self.z = nn.Parameter(self.initializations['z'].sample([self.met_locs.shape[0], self.K]), requires_grad=True)
        self.z_act = torch.softmax(self.z/self.temp_grouper, 1)

        kappa = torch.stack([((self.mu_bug - torch.tensor(self.microbe_locs[m, :])).pow(2)).sum(-1) for m in
                             range(self.microbe_locs.shape[0])])
        self.u = torch.sigmoid((self.r_bug - kappa) / self.temp_grouper)

    #@profile
    def forward(self, x, y):
        kappa = torch.stack([torch.sqrt(((self.mu_bug - torch.tensor(self.microbe_locs[m,:])).pow(2)).sum(-1)) for m in range(self.microbe_locs.shape[0])])
        self.u = torch.sigmoid((torch.exp(self.r_bug) - kappa)/self.temp_grouper)
        epsilon = self.temp_selector / 4
        if len(self.u.shape)>2:
            g = torch.einsum('ij,jkl->ikl', x, self.u.float())
            if not self.cluster_per_met_cluster:
                self.w_act = (1 - 2 * epsilon) * torch.sigmoid(self.w / self.tau_transformer) + epsilon
                g = (g * self.w_act).sum(-1)
        else:
            g = x@self.u.float()

        self.alpha_act = (1-2*epsilon)*torch.sigmoid(self.alpha/self.tau_transformer) + epsilon
        self.z_act = (1-2*epsilon)*torch.softmax(self.z / self.tau_transformer, 1) + epsilon
        if self.cluster_per_met_cluster:
            out_clusters = self.beta[0, :] + (g* self.beta[1:, :] * self.alpha_act).sum(1)
        else:
            out_clusters = self.beta[0,:] + torch.matmul(g, self.beta[1:,:]*self.alpha_act)
        loss = self.MAPloss.compute_loss(out_clusters,y)
        return out_clusters, loss

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("-learn", "--learn", help="params to learn", type=str, nargs='+')
    parser.add_argument("-priors", "--priors", help="priors to set", type=str, nargs='+')
    parser.add_argument("-case", "--case", help="case", type=str)
    parser.add_argument("-N_met", "--N_met", help="N_met", type=int)
    parser.add_argument("-N_bug", "--N_bug", help="N_bug", type=int)
    parser.add_argument("-n_local_clusters", "--n_local_clusters", help="number of microbe detectors", type=int)
    parser.add_argument("-L", "--L", help="number of microbe rules", type=int)
    parser.add_argument("-K", "--K", help="metab clusters", type=int)
    parser.add_argument("-N_nuisance", "--N_nuisance", help="N_nuisance", type=int)
    parser.add_argument("-meas_var", "--meas_var", help="measurment variance", type=float)
    parser.add_argument("-prior_meas_var", "--prior_meas_var", help = "prior measurment variance", type = float)
    parser.add_argument("-iterations", "--iterations", help="number of iterations", type=int)
    parser.add_argument("-seed", "--seed", help = "seed for random start", type = int)
    parser.add_argument("-load", "--load", help="0 to not load model, 1 to load model", type=int)
    parser.add_argument("-cluster_per_met_cluster", "--cluster_per_met_cluster",
                        help="whether or not to have each metabolite cluster have it's own clustering paradigm", type=int)
    parser.add_argument("-rep_clust", "--rep_clust", help = "whether or not bugs are in more than one cluster")
    args = parser.parse_args()

    # Set default values
    K=2
    n_local_clusters = 1
    L = 2
    N_met, N_bug = 10,10
    params2learn = ['all']
    priors2set = ['all']
    n_nuisance = 0
    meas_var = 0.001
    prior_meas_var = 100000
    case = '1-6-22'
    if args.rep_clust:
        case = case + '_repclust' + str(args.rep_clust)
    iterations = 20001
    seed = 0
    load = 0
    cluster_per_met_cluster = 0
    repeat_clusters = 0

    if args.K is not None:
        K = args.K
    if args.L is not None:
        L = args.L
    if args.n_local_clusters is not None:
        n_local_clusters = args.n_local_clusters
    if args.learn is not None:
        params2learn = args.learn
    if args.priors is not None:
        if 'none' in args.priors:
            args.priors = []
        priors2set = args.priors
    if args.case is not None:
        case = args.case
    if args.N_met is not None:
        N_met = args.N_met
    if args.N_bug is not None:
        N_bug = args.N_bug
    if args.N_nuisance is not None:
        n_nuisance = args.N_nuisance
    if args.meas_var is not None:
        meas_var = args.meas_var
    if args.prior_meas_var is not None:
        prior_meas_var = args.prior_meas_var
    if args.iterations is not None:
        iterations = args.iterations
    if args.seed is not None:
        seed = args.seed
    if args.load is not None:
        load = args.load
    if args.cluster_per_met_cluster is not None:
        cluster_per_met_cluster = args.cluster_per_met_cluster
    if args.rep_clust is not None:
        repeat_clusters = args.rep_clust

    # n_splits = 2
    use_MAP = True
    lr = 0.001
    temp_grouper, temp_selector = 'scheduled', 'scheduled'
    temp_transformer = 0.1
    # info = 'meas_var' + str(meas_var).replace('.', 'd') + '-prior_mvar' + str(prior_meas_var).replace('.', 'd') + \
    #        '-lr' + str(lr).replace('.','d')
    info = 'cluster_per_met' + str(cluster_per_met_cluster)
    if use_MAP:
        path = 'results_MAP/'
    else:
        path = 'results_ML/'

    path = path + case.replace(' ','_')
    if not os.path.isdir(path):
        os.mkdir(path)

    if 'all' not in params2learn or 'all' not in priors2set:
        path = path + '/learn_' + '_'.join(params2learn) + '-priors_' + '_'.join(priors2set) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)

    path = path + '/' + info + '-N_bug' + str(N_bug) + '-N_met' + str(N_met) + '-n_local' + str(n_local_clusters) + '-L' + str(L) + '-K' + str(K) +'/'
    if not os.path.isdir(path):
        os.mkdir(path)

    if 'all' in priors2set:
        priors2set = ['z','w','alpha','beta','mu_bug','mu_met','r_bug','r_met','pi_met']
        if n_local_clusters<=1:
            priors2set.remove('w')
    x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
    mu_met, r_bug, r_met, gen_u = generate_synthetic_data(
        N_met = N_met, N_bug = N_bug, N_met_clusters = K, N_local_clusters = n_local_clusters,
        N_bug_clusters = L,meas_var = meas_var, cluster_per_met_cluster= cluster_per_met_cluster,
        repeat_clusters= repeat_clusters)

    true_vals = {'y':y, 'beta':gen_beta, 'alpha':gen_alpha, 'mu_bug': mu_bug,
                 'mu_met': mu_met, 'u': gen_u,
                 'r_bug': r_bug, 'r_met': r_met, 'z': gen_z, 'w': gen_w, 'pi_met':np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0),
                 'pi_bug':np.expand_dims(np.sum(gen_w,0)/np.sum(np.sum(gen_w)),0)}
    plot_syn_data(path, x, y, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
                  r_bug, mu_met, r_met, gen_u)

    net = Model(gen_met_locs, gen_bug_locs, K=gen_z.shape[1], L=L, num_local_clusters=n_local_clusters,
                tau_transformer=temp_transformer, meas_var = prior_meas_var, compute_loss_for=priors2set,
                cluster_per_met_cluster=cluster_per_met_cluster)
    net.to(device)

    # net_ = Model(gen_met_locs, gen_bug_locs, K = gen_z.shape[1], L= L, num_local_clusters=n_local_clusters,
    #              tau_transformer=temp_transformer,
    #              meas_var = prior_meas_var, compute_loss_for=priors2set, cluster_per_met_cluster = cluster_per_met_cluster)

    for param, dist in net.distributions.items():
        parameter_dict = net.params[param]
        plot_distribution(dist, param, true_val = true_vals[param], ptype = 'prior', path = path, **parameter_dict)
    # kfold = KFold(n_splits = n_splits, shuffle = True)

    fig_dict4, ax_dict4 = {},{}
    fig_dict5, ax_dict5 = {},{}
    param_dict = {}
    tau_logspace = np.logspace(-0.5, -6, int(iterations/100))
    net.temp_grouper, net.temp_selector = tau_logspace[0],tau_logspace[0]
    # net_.temp_grouper, net_.temp_selector = tau_logspace[0], tau_logspace[0]
    param_dict[seed] = {}

    net.initialize(seed)
    # net_.initialize(seed)

    start = 0
    for name, parameter in net.named_parameters():
        if name not in params2learn and 'all' not in params2learn:
            setattr(net, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))
            parameter = nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False)
        elif name == 'w' or name == 'z' or name == 'alpha':
            parameter = getattr(net, name + '_act')
        elif name == 'r_bug':
            if 'r_bug' in params2learn or 'all' in params2learn:
                parameter = torch.exp(parameter)
        elif name == 'r_met' or 'all' in params2learn:
            if 'r_met' in params2learn:
                parameter = torch.exp(parameter)
        param_dict[seed][name] = [parameter.clone().detach().numpy()]
    param_dict[seed]['u'] = [net.u.clone().detach().numpy()]
    loss_vec = []
    train_out_vec = []
    optimizer = optim.RMSprop(net.parameters(), lr=lr)

    files = os.listdir(path)
    epochs = re.findall('epoch\d+', ' '.join(os.listdir(path)))
    path_orig = path
    if len(epochs)>0:
        largest = max([int(num.split('epoch')[-1]) for num in epochs])
        foldername = path + 'epoch' + str(largest) + '/'
        if 'seed' + str(seed) + '_checkpoint.tar' in os.listdir(foldername) and load==1:
            checkpoint = torch.load(foldername + 'seed' + str(seed) + '_checkpoint.tar')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch'] - 1
            # tau_logspace = np.concatenate((tau_logspace,np.logspace(-2, -5, int(iterations / 100))))
            ix = int((checkpoint['epoch'] - 400) / 100)
            if ix >= len(tau_logspace):
                ix = -1
            net.temp_grouper, net.temp_selector = tau_logspace[ix], tau_logspace[ix]
            iterations = start + iterations
            print('model loaded')
        else:
            print('no model loaded')
    else:
        print('no model loaded')

    x = torch.Tensor(x).to(device)
    # for name, parameter in net_.named_parameters():
    #     if 'r' in name:
    #         setattr(net_, name, nn.Parameter(torch.log(torch.Tensor(true_vals[name])), requires_grad=False))
    #     elif 'pi' in name:
    #         val = torch.Tensor(true_vals[name])
    #         val = torch.log(val) + torch.log(torch.exp(val).sum())
    #         setattr(net_, name, nn.Parameter(val, requires_grad=False))
    #     else:
    #         setattr(net_, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))
    # net_.z_act, net_.w_act = torch.softmax(net_.z / 0.1, 1), torch.softmax(net_.w / 0.1, 1)
    # net_.alpha[net_.alpha == 0] = -1
    # net_.alpha_act = torch.sigmoid(net_.alpha / 0.1)
    # # lowest_loss = criterion_.compute_loss(torch.Tensor(targets), torch.Tensor(targets))
    # _, lowest_loss = net_(x, torch.Tensor(y))
    # print('Lowest Loss:' + str(lowest_loss.item()))
    cluster_targets = np.stack([y[:,np.where(gen_z[:,i]==1)[0][0]] for i in np.arange(gen_z.shape[1])]).T
    timer = []
    end_learning = False
    tau_vec = []
    alpha_tau_vec = []
    lowest_loss_vec = []
    loss_dict_vec = []
    ix = 0
    for epoch in range(start, iterations):
        if epoch == iterations:
            end_learning = True
        if isinstance(temp_grouper, str) and isinstance(temp_selector, str):
            if epoch%100==0 and epoch>400:
                ix = int((epoch-400)/100)
                if ix >= len(tau_logspace):
                    ix = -1
                net.temp_grouper, net.temp_selector = tau_logspace[ix],tau_logspace[ix]
                # net_.temp_grouper, net_.temp_selector = tau_logspace[ix], tau_logspace[ix]
                # _, lowest_loss = net_(x, torch.Tensor(y))
                # print('Lowest Loss:' + str(lowest_loss.item()))
                # print('tau:' + str(net.temp_grouper))
            # net.temp_grouper, net.temp_selector = 1/(epoch+1), 1/(epoch+1)
            # net_.temp_grouper, net_.temp_selector = 1 / (epoch + 1), 1 / (epoch + 1)
            tau_vec.append(net.temp_grouper)
        optimizer.zero_grad()
        cluster_outputs, loss = net(x, torch.Tensor(y))

        train_out_vec.append(cluster_outputs)
        loss_vec.append(loss.item())
        loss_dict_vec.append(copy.copy(net.MAPloss.loss_dict))
        loss.backward()
        optimizer.step()

        for name, parameter in net.named_parameters():
            if name == 'w' or name == 'z' or name == 'alpha':
                parameter = getattr(net, name + '_act')
            elif name == 'r_bug':
                if 'r_bug' in params2learn or 'all' in params2learn:
                    parameter = torch.exp(parameter)
            elif name == 'r_met':
                if 'r_met' in params2learn or 'all' in params2learn:
                    parameter = torch.exp(parameter)
            param_dict[seed][name].append(parameter.clone().detach().numpy())
        param_dict[seed]['u'].append(net.u.clone().detach().numpy())

        if epoch%1000==0 or end_learning:
            if epoch != 0:
                fig3, ax3 = plt.subplots(figsize=(8, 8))
                fig3, ax3 = plot_loss(fig3, ax3, seed, epoch+1 - start, loss_vec, lowest_loss=None)
                fig3.tight_layout()
                fig3.savefig(path_orig + 'loss_seed_' + str(seed) + '.pdf')
                plt.close(fig3)

                fig3, ax3 = plt.subplots(len(net.MAPloss.loss_dict.keys()
                                             ), 1, figsize = (8, 8*len(net.MAPloss.loss_dict.keys())))
                it = 0
                for key, loss_val in net.MAPloss.loss_dict.items():
                    vals = [ldv[key].item() for ldv in loss_dict_vec]
                    ax3[it].plot(range(start, epoch+1), vals)
                    ax3[it].set_title(key)
                    ax3[it].set_xlabel('Epochs')
                    ax3[it].set_ylabel('Loss')
                    it += 1
                fig3.tight_layout()
                fig3.savefig(path_orig + 'lossdict_seed_' + str(seed) + '.pdf')
                plt.close(fig3)


        if (epoch%1000 == 0 and epoch != 0) or end_learning:
            print('Epoch ' + str(epoch) + ' Loss: ' + str(loss_vec[-1]))
            print('Tau: ' + str(net.temp_grouper))
            print('')
            if 'epoch' not in path:
                path = path + 'epoch' + str(epoch) + '/'
            else:
                path = path.split('epoch')[0] + 'epoch' + str(epoch) + '/'
            if not os.path.isdir(path):
                os.mkdir(path)
            plot_param_traces(path, param_dict[seed], params2learn, true_vals, net, seed)

            # best_mod = np.argmin(loss_vec)
            # plot_output(path, best_mod, train_out_vec, y, gen_z, param_dict[seed],
            #                       seed, type = 'best_train')
            # plot_output_locations(path, net, best_mod, param_dict[seed], seed, type = 'best_train')
            last_mod = -1
            plot_output(path, path_orig, last_mod, train_out_vec, y, gen_z, param_dict[seed],
                                 seed, type = 'last_train')
            plot_output_locations(path, net, last_mod, param_dict[seed], seed, type = 'last_train')
            if n_local_clusters > 1:
                plot_rules_detectors_tree(path, net, last_mod, param_dict[seed], gen_bug_locs, seed)
            if isinstance(temp_grouper, str) and len(tau_vec) > 0:
                fig, ax = plt.subplots()
                ax.semilogy(range(start, epoch+1), tau_vec)
                fig.savefig(path + 'seed' + str(seed) + '_tau_scheduler.pdf')
                plt.close(fig)
            torch.save({'model_state_dict':net.state_dict(),
                       'optimizer_state_dict':optimizer.state_dict(),
                       'epoch': epoch},
                       path + 'seed' + str(seed) + '_checkpoint.tar')

