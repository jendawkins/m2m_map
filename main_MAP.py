import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta
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
                 seed = 0, alpha_temp = 1, omega_temp = 1, omega_temp_2 = 1, data_meas_var = 1, L = 3, cluster_per_met_cluster = 1,
                 compute_loss_for = ['alpha','beta','w','z','mu_bug','r_bug','pi_bug','mu_met','r_met','pi_met'],
                 learn_num_met_clusters = False, learn_num_bug_clusters = False, linear = True, mu_hyper = False, r_hyper = False,
                l1= 1, marginalize_z = True, pp = 1, learn_mvar = True, learn_w = True, sample_a = True, hard_w = True):
        super(Model, self).__init__()
        self.num_local_clusters = num_local_clusters
        self.sigma_beta = 1/(50*data_meas_var)
        self.sigma_alpha = self.sigma_beta/data_meas_var
        self.beta_var = beta_var
        self.mu_var_met = 10*(1/met_locs.shape[1])*np.sum(np.var(met_locs, 0))
        self.mu_var_bug = 10*(1/microbe_locs.shape[1])*np.sum(np.var(microbe_locs, 0))
        self.compute_loss_for = compute_loss_for
        self.MAPloss = MAPloss(self)
        self.temp_scheduled = temp_scheduled
        self.met_locs = met_locs
        self.microbe_locs = microbe_locs
        self.embedding_dim = met_locs.shape[1]
        self.seed = seed
        self.alpha_temp = alpha_temp
        self.omega_temp = omega_temp
        self.omega_temp_2 = omega_temp_2
        self.learn_num_met_clusters = learn_num_met_clusters
        self.learn_num_bug_clusters = learn_num_bug_clusters
        self.linear = linear
        self.mu_hyper = mu_hyper
        self.r_hyper = r_hyper
        self.N_met = self.met_locs.shape[0]
        self.N_bug = self.microbe_locs.shape[0]
        self.l1 = l1
        self.marginalize_z = marginalize_z
        self.learn_w = learn_w
        self.sample_a = sample_a
        self.pp = pp
        self.learn_mvar = learn_mvar
        self.w_loc = L/self.N_bug
        self.hard_w = hard_w

        self.L, self.K = L, K
        self.L_sm, self.K_sm = L, K
        if self.learn_num_met_clusters:
            self.K = met_locs.shape[0]-1
            self.K_sm = K
        if self.learn_num_bug_clusters:
            self.L = microbe_locs.shape[0] - 1
            self.L_sm = L

        if not self.linear:
            self.NAM = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Sequential(
                nn.Linear(1, 32, bias = True),
                nn.ELU(),
                nn.Linear(32, 32, bias = True),
                nn.ELU(),
                nn.Linear(32, 16, bias = True),
                nn.ELU(),
                nn.Linear(16,1, bias = True)
            ) for pp in np.arange(self.pp)]) for l in np.arange(self.L)]) for k in np.arange(self.K)])

        self.alpha_loc = 1/(self.L_sm*self.K_sm)
        self.met_range = np.array([np.max(self.met_locs[:, d]) - np.min(self.met_locs[:, d]) for d in np.arange(self.met_locs.shape[1])])
        self.Ro = torch.diag(torch.Tensor(self.met_range))
        self.r_scale_met = np.sqrt(np.sum(self.met_range**2)) / (2*self.K_sm)

        self.bug_range = np.array([np.max(self.microbe_locs[:, d]) - np.min(self.microbe_locs[:, d]) for d in np.arange(self.microbe_locs.shape[1])])
        self.r_scale_bug = np.sqrt(np.sum(self.bug_range**2)) / (2*self.L_sm)

        self.params = {}
        self.distributions = {}
        self.params['beta'] = {'mean': 0, 'scale': np.sqrt(self.beta_var)}
        self.params['alpha'] = {'loc': self.alpha_loc, 'temp':self.temp_scheduled}
        self.params['mu_met'] = {'mean': 0, 'var': self.mu_var_met}
        self.params['mu_bug'] = {'mean': 0, 'var': self.mu_var_bug}
        self.params['r_bug'] = {'dof': 2, 'scale': self.r_scale_bug}
        self.params['r_met'] = {'dof': 2, 'scale': self.r_scale_met}
        self.params['e_met'] = {'dof': 10, 'scale': 10*self.K}
        self.params['pi_met'] = {'epsilon': [2] * self.K}
        self.params['b'] = {'mean': 0, 'scale': 1}
        self.params['sigma'] = {'dof': 0.01, 'scale': 0.01}
        self.params['p'] = {'alpha': 1, 'beta': 5}
        self.distributions['beta'] = Normal(self.params['beta']['mean'], self.params['beta']['scale'])
        self.distributions['alpha'] = BinaryConcrete(self.params['alpha']['loc'], self.params['alpha']['temp'])
        self.distributions['mu_met'] = MultivariateNormal(torch.zeros(self.embedding_dim), 10*self.params['mu_met']['var']*torch.eye(self.embedding_dim))
        self.distributions['mu_bug'] = MultivariateNormal(torch.zeros(self.embedding_dim), 10*self.params['mu_bug']['var']*torch.eye(self.embedding_dim))
        self.distributions['r_bug'] = Gamma(self.params['r_bug']['dof'], 2*self.params['r_bug']['scale'])
        self.distributions['r_met'] = Gamma(self.params['r_met']['dof'], 2*self.params['r_met']['scale'])
        # self.params['pi_bug'] = {'epsilon': [2]*self.L}
        self.distributions['pi_met'] = Dirichlet(torch.Tensor(self.params['pi_met']['epsilon']))
        self.distributions['e_met'] = Gamma(10, 10*self.K)
        self.distributions['b'] = MultivariateNormal(torch.zeros(self.embedding_dim), torch.eye(self.embedding_dim))
        self.distributions['sigma'] = Gamma(0.1, 0.1)
        self.distributions['p'] = Beta(self.params['p']['alpha'],self.params['p']['beta'])
        # self.distributions['pi_bug'] = Dirichlet(torch.Tensor(self.params['pi_bug']['epsilon']))
        self.range_dict = {}
        self.lr_range = {}
        for param, dist in self.distributions.items():
            sampler = dist.sample([100])
            if len(sampler.shape)>1:
                sampler = sampler[:,0]
            if 'mu_bug' or 'mu_met' in param:
                sampler = sampler/10
            if 'r_met' in param:
                self.range_dict[param] = (-0.1,torch.tensor(np.sqrt(np.sum(self.met_range**2))/2).float())
            elif 'r_bug' in param:
                self.range_dict[param] = (-0.1,torch.tensor(np.sqrt(np.sum(self.bug_range**2))/2).float())
            elif 'sigma' in param:
                vals = np.log(1/self.distributions[param].sample([100]))
                self.range_dict[param] = (vals.min(), vals.max())
                self.lr_range[param] = torch.abs(vals.max() - vals.min())
            elif 'w' in param or 'z' in param or 'alpha' in param:
                self.range_dict[param] = (-0.1,1.1)
            elif param == 'pi' or param == 'p' or param == 'lambda_mu':
                log_sampler = torch.log(sampler)
                self.lr_range[param] = torch.abs(log_sampler.max() - log_sampler.min())
                range = sampler.max() - sampler.min()
                self.range_dict[param] = (sampler.min() - range * 0.1, sampler.max() + range * 0.1)
            else:
                range = sampler.max() - sampler.min()
                self.range_dict[param] = (sampler.min() - range * 0.1, sampler.max() + range * 0.1)
                self.lr_range[param] = torch.abs(sampler.max() - sampler.min())

        self.lr_range['r_bug'] = torch.log(torch.tensor(self.r_scale_bug).float())
        bug_range = 1.1*(np.max(self.microbe_locs,0) - np.min(self.microbe_locs,0))
        self.lr_range['mu_bug'] = torch.log(torch.tensor(np.sqrt(np.sum(bug_range**2))).float())

        self.lr_range['r_met'] = torch.log(torch.tensor(self.r_scale_met).float())
        met_range = 1.1*(np.max(self.met_locs,0) - np.min(self.met_locs,0))
        self.lr_range['mu_met'] = torch.tensor(np.sqrt(np.sum(met_range ** 2))).float()

        vals = Normal(0,500).sample([100])
        self.lr_range['w'] = torch.abs(vals.max()- vals.min())
        self.lr_range['z'] = torch.abs(vals.max() - vals.min())
        self.lr_range['alpha'] = torch.abs(vals.max() - vals.min())
        self.range_dict['beta[1:,:]*alpha'] = self.range_dict['beta']

        vals = torch.log(Gamma(0.5, 0.5).sample([100]))
        self.range_dict['lambda_mu'] = (vals.min(), vals.max())
        self.range_dict['C'] = self.range_dict['r_met']
        self.initialize(self.seed)

    def initialize(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.initializations = {}
        self.initializations['beta'] = Normal(0,np.sqrt(self.beta_var))
        self.initializations['alpha'] = Normal(0,0.00001)
        self.initializations['mu_met'] = MultivariateNormal(torch.zeros(self.embedding_dim),
                                                            (self.mu_var_met/5)*torch.eye(self.embedding_dim))
        # self.initializations['mu_bug'] = MultivariateNormal(torch.zeros(self.embedding_dim),
        #                                                     (self.mu_var_bug)*torch.eye(self.embedding_dim))
        self.initializations['z'] = Normal(0,1)
        self.initializations['w'] = Normal(0,1)

        # self.pw = nn.Parameter(torch.tensor(-2.3).float(), requires_grad=True)
        self.p = nn.Parameter(torch.tensor(-2.3).float(), requires_grad=True)

        if self.learn_mvar:
            self.sigma = nn.Parameter(torch.tensor(1.).float(), requires_grad=True)
        else:
            self.sigma = torch.tensor(1.).float()
        if self.linear:
            self.beta = nn.Parameter(self.initializations['beta'].sample([self.L+1, self.K]), requires_grad=True)
        else:
            self.beta = nn.Parameter(self.initializations['beta'].sample([self.K]), requires_grad=True)
        self.alpha = nn.Parameter(self.initializations['alpha'].sample([self.L, self.K]), requires_grad=True)
        self.alpha_act = torch.sigmoid(self.alpha)
        pi_init = (1/self.K)*torch.ones(self.K)

        # ixs = [np.random.choice(range(self.microbe_locs.shape[0]), 1, replace = False)[0]]
        # for l in np.arange(self.L-1):
        #     choose_from = np.concatenate([np.where(((self.microbe_locs - self.microbe_locs[ixs[i],:])**2).sum(1)>
        #                                            self.r_scale_bug)[0] for i in np.arange(len(ixs))])
        #     if len(choose_from == 0):
        #         choose_from = np.arange(self.microbe_locs.shape[0])
        #     ixs.append(np.random.choice(choose_from, 1, replace = False)[0])
        # ix = np.array(ixs)
        # self.mu_bug = nn.Parameter(torch.Tensor(self.microbe_locs[ix,:]), requires_grad=True)
        self.mu_bug = nn.Parameter(MultivariateNormal(torch.zeros(self.embedding_dim),
                                                      0.001*self.params['mu_bug']['var']*torch.eye(self.embedding_dim)
                                                      ).sample([self.L]), requires_grad=True)
        # self.mu_bug = nn.Parameter(self.initializations['mu_bug'].sample([self.L]), requires_grad=True)
        r_temp = self.L_sm*self.r_scale_bug*torch.ones(self.L)
        # if self.learn_num_bug_clusters:
        #     l_remove = self.L - self.L_sm
        #     ix_remove = np.random.choice(range(self.L), int(l_remove), replace=False)
        #     r_temp[ix_remove,:] = 1e-4

        # self.r_bug = nn.Parameter(torch.log(self.L_sm*r_temp.squeeze()), requires_grad=True)
        self.r_bug = nn.Parameter(torch.log(r_temp.squeeze()), requires_grad=True)
        # self.w_act = torch.sigmoid(self.w/self.tau_transformer)

        ix = np.random.choice(range(self.met_locs.shape[0]), self.K, replace = False)
        self.mu_met = nn.Parameter(torch.Tensor(self.met_locs[ix,:]), requires_grad = True)
        # self.mu_met = nn.Parameter(self.initializations['mu_met'].sample([self.K]), requires_grad=True)
        r_temp = self.r_scale_met*torch.ones((self.K)).squeeze()

        if self.learn_num_met_clusters:
            self.e_met = nn.Parameter(torch.log(pi_init.unsqueeze(0)), requires_grad=True)
            self.pi_met = nn.Parameter(Dirichlet(pi_init.unsqueeze(0)).sample(), requires_grad=True)
            if self.mu_hyper:
                self.lambda_mu = nn.Parameter(torch.log(Gamma(0.5, 0.5).sample([self.embedding_dim])), requires_grad=True)
                Lambda = torch.diag(torch.sqrt(torch.exp(self.lambda_mu)))
                Bo = Lambda @ self.Ro @ Lambda
                self.b = nn.Parameter(
                    MultivariateNormal(torch.zeros(self.embedding_dim), torch.eye(self.embedding_dim)).sample([self.K]), requires_grad=True)
            if self.r_hyper:
                self.C = nn.Parameter(torch.Tensor(np.log(self.r_scale_met) * np.ones(self.K)), requires_grad=True)
                self.c = 1.25 + (self.embedding_dim - 1) / 4
                self.g = 0.25 + (self.embedding_dim - 1) / 4
                self.G = self.c / (50 * self.g) * np.sqrt(np.sum(self.met_range ** 2))
        else:
            self.e_met = torch.log(pi_init.unsqueeze(0))
            self.pi_met = nn.Parameter(pi_init.unsqueeze(0), requires_grad=True)
        self.r_met = nn.Parameter(torch.log(r_temp), requires_grad = True)

        if not self.marginalize_z:
            z_temp = self.initializations['z'].sample([self.met_locs.shape[0], self.K])

            self.z = nn.Parameter(z_temp, requires_grad=True)
            self.z_act = torch.softmax(self.z/self.omega_temp, 1)
        else:
            self.z_act = torch.zeros((self.N_met, self.K))

        kappa = torch.stack([((self.mu_bug - torch.tensor(self.microbe_locs[m, :])).pow(2)).sum(-1) for m in
                             range(self.microbe_locs.shape[0])])

        if not self.learn_w:
            self.w_act = torch.sigmoid((self.r_bug - kappa))
        elif self.learn_w == 1:
            w_temp = self.initializations['z'].sample([self.microbe_locs.shape[0], self.L])
            self.w = nn.Parameter(w_temp, requires_grad=True)
            if self.hard_w:
                self.w_soft = torch.sigmoid(self.w/self.omega_temp)
                self.w_act = torch.round(self.w_soft)
            else:
                self.w_act = torch.sigmoid(self.w/self.omega_temp)
        elif self.learn_w > 1:
            self.w = self.initializations['z'].sample([self.microbe_locs.shape[0], self.L])
            if self.hard_w:
                self.w_soft = torch.sigmoid(self.w/self.omega_temp)
                self.w_act = torch.round(self.w_soft)
            else:
                self.w_act = torch.sigmoid(self.w/self.omega_temp)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def get_w_probs(self):
        eye = torch.eye(self.embedding_dim).unsqueeze(0).expand(self.L, -1, -1)
        var = torch.exp(self.r_bug).unsqueeze(-1).unsqueeze(-1).expand(-1,self.embedding_dim,
                                                                           self.embedding_dim)*eye
        w_log_probs = Bernoulli(0.1).log_prob(torch.tensor(1).float()) + MultivariateNormal(self.mu_bug.unsqueeze(1).
                                                                                     expand(-1,self.N_bug,-1),var.unsqueeze(1).expand(
                -1,self.N_bug,-1,-1)).log_prob(
            torch.Tensor(self.microbe_locs))
        w_log_probs[w_log_probs>0] = 0
        temp = torch.exp(w_log_probs.detach()).T
        norm = torch.stack([temp.index_fill(-1,i,0).sum(1) for i in torch.arange(temp.shape[1])])
        # norm = torch.logsumexp(w_log_probs,1)
        prob =temp/norm.T
        prob[prob > 1] = 1-(1e-4)
        return prob

    # def get_alpha_logits(self):


    def forward(self, x, y):
        alpha_epsilon = self.alpha_temp / 4
        omega_epsilon = self.omega_temp / 4
        if not self.learn_w:
            kappa = torch.stack(
                [torch.sqrt(((self.mu_bug - torch.tensor(self.microbe_locs[m, :])).pow(2)).sum(-1)) for m in
                 np.arange(self.microbe_locs.shape[0])])
            self.w_act = torch.sigmoid((torch.exp(self.r_bug) - kappa)/self.omega_temp)
        elif self.learn_w == 1:
            if self.hard_w:
                self.w_soft, self.w_act = gumbel_sigmoid(self.w, self.omega_temp, omega_epsilon)
            else:
                self.w_act = (1-2*omega_epsilon) * torch.sigmoid(self.w/self.omega_temp) + omega_epsilon
        else:
            kappa = torch.stack(
                [torch.sqrt(((self.mu_bug - torch.tensor(self.microbe_locs[m, :])).pow(2)).sum(-1)) for m in
                 np.arange(self.microbe_locs.shape[0])])
            if self.learn_w == 2:
                gamma = torch.sigmoid(kappa/torch.exp(self.r_bug))
                self.w_soft, self.w_act = gumbel_sigmoid(torch.log(gamma), self.omega_temp, omega_epsilon)
            if self.learn_w == 3:
                uu = torch.sigmoid((torch.exp(self.r_bug) - kappa)/self.omega_temp_2)
                gamma = uu*0.99 + (1-uu)*(1-0.99)
                self.w_soft, self.w_act = gumbel_sigmoid(torch.log(gamma), self.omega_temp, omega_epsilon)

        g = x@self.w_act.float()

        if not self.sample_a:
            self.alpha_act = (1-2*alpha_epsilon)*torch.sigmoid(self.alpha/self.alpha_temp) + alpha_epsilon
        else:
            self.alpha_soft, self.alpha_act = gumbel_sigmoid(self.alpha, self.alpha_temp, alpha_epsilon)
        if not self.marginalize_z:
            self.z_act = (1-2*alpha_epsilon)*torch.softmax(self.z / self.alpha_temp, 1) + alpha_epsilon

        if self.linear:
            out_clusters = self.beta[0,:] + torch.matmul(g, self.beta[1:,:]*self.alpha_act)
        else:
            out_clusters = self.beta + torch.cat([torch.cat([self.alpha_act[l,k]*torch.stack([
                self.NAM[k][l][p](g[:,l:l+1]) for p in np.arange(self.pp)],-1).sum(-1)
                                                  for l in np.arange(self.L)],1).sum(1).unsqueeze(1)
                                                  for k in np.arange(self.K)],1)


        loss = self.MAPloss.compute_loss(out_clusters,y)

        if not self.linear and self.l1:
            l1_parameters = []
            for parameter in self.NAM.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = self.compute_l1_loss(torch.cat(l1_parameters))
            loss += l1
        return out_clusters, loss


def run_learner(args, device):
    lr_min = args.lr/50
    if args.linear == 1:
        args.l1 = 0

    # T_0= 100
    # T_mult = 1
    T_0= 10
    T_mult = 2

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


    if 'all' in priors2set:
        priors2set = ['z','w','alpha','beta','mu_bug','mu_met','r_bug','r_met','pi_met','e_met','sigma','p']
    if args.lb == 0 and 'p' in priors2set:
        priors2set.remove('p')
    if args.lm == 0 and 'e_met' in priors2set:
        priors2set.remove('e_met')
    if args.linear == 0 and 'beta' in priors2set:
        priors2set.remove('beta')
    if args.learn_mvar == 0 and 'sigma' in priors2set:
        priors2set.remove('sigma')
    if args.mz and 'z' in priors2set:
        priors2set.remove('z')
    if args.lw == 0 and 'w' in priors2set:
        priors2set.remove('w')
    if args.lw ==0 and ('w' in params2learn or params2learn=='all'):
        params2learn.remove('w')
    if args.fix:
        for p in args.fix:
            if p in priors2set:
                priors2set.remove(p)
            if p in params2learn:
                params2learn.remove(p)

    if 'all' not in params2learn or 'all' not in priors2set:
        path = path + '/learn_' + '_'.join(params2learn) + '-priors_' + '_'.join(priors2set) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)


    info = '-d' + str(args.dim) + '-lr' + str(args.lr) + '-linear'*(args.linear) + '-rep_clust'*args.rep_clust + \
           '-learn_w'+str(args.lw) +'-adj_lr'*args.adjust_lr + '-samp_mu'*args.sample_mu + \
           '-adj_mvar'*args.adjust_mvar + '-l1'*(args.l1) + '-'*(1-args.linear) +args.nltype*(1-args.linear) + \
           '-lm'*args.lm + str(args.lm)*args.lm + '-lb'*args.lb + str(args.lb)*args.lb + \
           '-sched_lr'*args.schedule_lr + '-dist_var_perc' + str(args.dist_var_perc).replace('.', '_') + \
           '-learn_mvar'*args.learn_mvar +'-N_bug' + str(args.N_bug) + \
           '-N_met' + str(args.N_met) + '-L' + str(args.L) + '-K' + str(args.K) + '-atau' + str(args.a_tau).replace('.','_') + \
            '-wtau' + str(args.w_tau).replace('.', '_') + '-wtau2' + str(args.w_tau2).replace('.', '_') + 'rep_clust'*(args.rep_clust!=0)

    path = path + '/' + info +'/'

    if not os.path.isdir(path):
        os.mkdir(path)

    # print(params2learn)
    # print(priors2set)
    x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
    mu_met, r_bug, r_met, gen_u = generate_synthetic_data(
        N_met = args.N_met, N_bug = args.N_bug, N_met_clusters = args.K,
        N_bug_clusters = args.L,meas_var = args.meas_var,
        repeat_clusters= args.rep_clust, N_samples=args.N_samples, linear = args.linear,
        nl_type = args.nltype, dist_var_perc=args.dist_var_perc, embedding_dim=args.dim)

    # y = (y - np.mean(y, 0)) / np.std((y - np.mean(y)), 0)
    if not args.linear:
        gen_beta = gen_beta[0,:]
    plot_syn_data(path, x, y, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
                  r_bug, mu_met, r_met, gen_u)

    if args.lm:
        r_met = np.append(r_met, np.zeros(args.N_met-1-len(r_met)))
        gen_z = np.hstack((gen_z, np.zeros((args.N_met, args.N_met - 1 - args.K))))
        mu_met = np.vstack((mu_met, np.zeros((args.N_met - args.K - 1, mu_met.shape[1]))))
        if args.linear:
            gen_beta = np.hstack((gen_beta, np.zeros((gen_beta.shape[0], args.N_met - args.K - 1))))
        gen_alpha = np.hstack((gen_alpha, np.zeros((gen_alpha.shape[0], args.N_met - args.K - 1))))
    if args.lb:
        r_bug = np.append(r_bug, np.zeros(args.N_bug - 1 - len(r_bug)))
        gen_w = np.hstack((gen_w, np.zeros((args.N_bug, args.N_bug - 1 - args.L))))
        gen_u = np.hstack((gen_u, np.zeros((args.N_bug, args.N_bug - 1 - args.L))))
        mu_bug = np.vstack((mu_bug, np.zeros((args.N_bug - args.L - 1, mu_bug.shape[1]))))
        if args.linear:
            gen_beta = np.vstack((gen_beta, np.zeros((args.N_bug - args.L - 1, gen_beta.shape[1]))))
        gen_alpha = np.vstack((gen_alpha, np.zeros((args.N_bug - args.L - 1, gen_alpha.shape[1]))))

    true_vals = {'y':y, 'beta':gen_beta, 'alpha':gen_alpha, 'mu_bug': mu_bug,
                 'mu_met': mu_met, 'u': gen_u,'w_soft': gen_w,
                 'r_bug':1.2*r_bug, 'r_met': 1.2*r_met, 'z': gen_z, 'w': gen_w, 'pi_met':np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0),
                 'pi_bug':np.expand_dims(np.sum(gen_w,0)/np.sum(np.sum(gen_w)),0), 'bug_locs': gen_bug_locs, 'met_locs':gen_met_locs,
                 'e_met': np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0),'b': mu_met, 'sigma': args.meas_var,
                 'p': args.p}

    plot_interactions(path, 0, true_vals, '_InitialInteractions')
    if args.linear:
        true_vals['beta[1:,:]*alpha'] = gen_beta[1:,:]*sigmoid(gen_alpha)

    print(priors2set)
    net = Model(gen_met_locs, gen_bug_locs, K=args.K, L=args.L,
                compute_loss_for=priors2set,
                learn_num_bug_clusters=args.lb,learn_num_met_clusters=args.lm, linear = args.linear==1,
                mu_hyper = args.hyper_mu == 1, r_hyper = args.hyper_r == 1, marginalize_z=args.mz==1,
                pp = args.p_num, learn_w=args.lw, hard_w = args.hard, data_meas_var = args.meas_var)
    net.initialize(args.seed)
    net.to(device)

    for param, dist in net.distributions.items():
        parameter_dict = net.params[param]
        plot_distribution(dist, param, true_val = true_vals[param], ptype = 'prior', path = path, **parameter_dict)


    param_dict = {}
    alpha_tau_logspace = np.logspace(args.a_tau[0], args.a_tau[1], args.iterations)
    omega_tau_logspace = np.logspace(args.w_tau[0], args.w_tau[1], args.iterations)
    omega_tau_2_logspace = np.logspace(args.w_tau2[0], args.w_tau[1], args.iterations)

    net.alpha_temp = alpha_tau_logspace[0]
    net.omega_temp = omega_tau_logspace[0]
    net.omega_temp_2 = omega_tau_2_logspace[0]
    param_dict[args.seed] = {}


    start = 0
    for name, parameter in net.named_parameters():
        if 'NAM' in name or 'lambda_mu' in name or name=='b' or name == 'C':
            continue
        if name not in params2learn and 'all' not in params2learn:
            if name == 'r_bug' or name == 'r_met' or name == 'e_met' or name == 'sigma' or name == 'p' or name == 'pi_met':
                setattr(net, name, nn.Parameter(torch.tensor(np.log(true_vals[name])).float(), requires_grad=False))
            else:
                setattr(net, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))
        if name == 'z' or name == 'alpha' or name == 'w':
            parameter = getattr(net, name + '_act')
        if name == 'r_bug' or name == 'r_met' or name == 'e_met' or name == 'sigma' or name == 'p':
            parameter = np.exp(parameter.clone().detach().numpy())
        if name == 'pi_met':
            parameter = torch.softmax(parameter.clone().detach(),1).numpy()

        if torch.is_tensor(parameter):
            param_dict[args.seed][name] = [parameter.clone().detach().numpy()]
        else:
            param_dict[args.seed][name] = [parameter]
    if args.mz:
        param_dict[args.seed]['z'] = [net.z_act.clone().numpy()]
    if 'w' not in param_dict[args.seed].keys():
        param_dict[args.seed]['w'] = [net.w_act.clone().detach().numpy()]
    if net.linear:
        param_dict[args.seed]['beta[1:,:]*alpha'] = [net.beta[1:,:].clone().detach().numpy()*net.alpha_act.clone().detach().numpy()]
    if args.lw:
        if args.hard:
            param_dict[args.seed]['w_soft'] = [net.w_soft.clone().detach().numpy()]
        param_dict[args.seed]['w_learned'] = [net.w.clone().detach().numpy()]
    loss_vec = []
    train_out_vec = []

    lr_dict = {}
    if args.adjust_lr:
        matching_dict = {}
        lr_list = []
        # beta_range = net.lr_range['beta']
        ii = 0
        for name, parameter in net.named_parameters():
            if name in params2learn or 'all' in params2learn or 'NAM' in name:
                # range = np.abs(np.max(parameter.detach().view(-1).numpy()) - np.min(parameter.detach().view(-1).numpy()))
                if name not in net.lr_range.keys():
                    range = np.abs(np.max(parameter.detach().view(-1).numpy()) - np.min(parameter.detach().view(-1).numpy()))
                else:
                    range = net.lr_range[name]

                # if name == 'mu_bug' or name == 'r_bug' or 'w' in name:
                #     range = range*10
                matching_dict[name] = ii
                ii+= 1
                # if 'bug' in name:
                #     lr_out = 100*args.lr
                # else:
                #     lr_out = args.lr
                # lr_list.append({'params': parameter, 'lr': lr_out})
                # lr_dict[name] = [lr_out]

                lr_list.append({'params': parameter, 'lr': (args.lr / net.lr_range['beta']) * range})
                lr_dict[name] = [(args.lr / net.lr_range['beta'].item()) * range.item()]
        optimizer = optim.RMSprop(lr_list, lr=args.lr)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr)

    pd.DataFrame(lr_dict).to_csv(path + 'per_param_lr.csv')
    if args.schedule_lr:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    # print(lr_list)
    # optimizer = optim.RMSprop(lr_list, lr=args.lr)

    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)

    files = os.listdir(path)
    epochs = re.findall('epoch\d+', ' '.join(os.listdir(path)))
    path_orig = path
    mvar = []
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
            if args.schedule_lr:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start = int(checkpoint['epoch'] - 1)
            # tau_logspace = np.concatenate((tau_logspace,np.logspace(-2, -5, int(iterations / 100))))
            ix = int(checkpoint['epoch'])-1000
            if ix >= len(alpha_tau_logspace):
                ix = -1
            if ix > -1:
                net.alpha_temp = alpha_tau_logspace[ix]
                net.omega_temp = omega_tau_logspace[ix]
                net.omega_temp_2 = omega_tau_2_logspace[ix]
            else:
                net.alpha_temp = alpha_tau_logspace[0]
                net.omega_temp = omega_tau_logspace[0]
                net.omega_temp_2 = omega_tau_2_logspace[0]
            if args.iterations <= start:
                print('training complete')
                sys.exit()
            print('model loaded')
        else:
            print('no model loaded')
    else:
        print('no model loaded')

    if not os.path.isdir(path + '/epoch0'):
        os.mkdir(path + '/epoch0')
    plot_syn_data(path + '/epoch0/seed' + str(args.seed), x, y, gen_z, gen_bug_locs, gen_met_locs, net.mu_bug.detach().numpy(),
                  torch.exp(net.r_bug.detach()).numpy(), net.mu_met.detach().numpy(), torch.exp(net.r_met.detach()).numpy(),
                  gen_w)

    x = torch.Tensor(x).to(device)
    # cluster_targets = np.stack([y[:,np.where(gen_z[:,i]==1)[0][0]] for i in np.arange(gen_z.shape[1])]).T
    loss_dict_vec = {}
    ix = 0
    # grad_dict = {}
    T_i = T_0
    stime = time.time()
    last_epoch = 0
    tracking_name = list(lr_dict.keys())[0]
    lr_tracking = lr_dict[tracking_name]
    for epoch in np.arange(start, args.iterations):
        if isinstance(temp_scheduled, str):
            net.alpha_temp = alpha_tau_logspace[ix]
            net.omega_temp = omega_tau_logspace[ix]
            net.omega_temp_2 = omega_tau_2_logspace[ix]
            # if epoch>1000:
            #     ix = int(epoch)-1000
            #     if ix >= len(tau_logspace):
            #         ix = -1
        optimizer.zero_grad()
        cluster_outputs, loss = net(x, torch.Tensor(y))
        train_out_vec.append(cluster_outputs)
        loss.backward()
        loss_vec.append(loss.item())

        for param in net.MAPloss.loss_dict:
            if param not in loss_dict_vec.keys():
                loss_dict_vec[param] = [net.MAPloss.loss_dict[param].detach().item()]
            else:
                loss_dict_vec[param].append(net.MAPloss.loss_dict[param].detach().item())
            # if epoch%100==0 and epoch > 0 and ('mu_bug' in params2learn or 'all' in params2learn) and args.sample_mu:
            #     if (loss_vec[-1] >= np.array(loss_vec[-10:-1])).all():
            #         val = net.initializations['mu_bug'].sample([net.L])
            #         setattr(net, 'mu_bug', nn.Parameter(val, requires_grad=True))

        optimizer.step()
        if args.schedule_lr:
            scheduler.step()

            ixx = matching_dict[tracking_name]
            try:
                lr_curr = scheduler.get_last_lr()[ixx].item()
            except:
                lr_curr = scheduler.get_last_lr()[ixx]
            if lr_curr > lr_tracking[-1]:
                if epoch > 2000 + last_epoch or epoch == args.iterations - 1:
                    last_epoch = epoch
            lr_tracking.append(lr_curr)
        else:
            last_epoch = args.iterations-1

        # lr_list = []
        # if epoch%10 == 0:
        for name, parameter in net.named_parameters():
            # if name in lr_dict.keys():
            #     ixx = matching_dict[name]
            #     lr_dict[name].append(scheduler.get_last_lr()[ixx].item())
            if 'NAM' in name or 'lambda_mu' in name or name=='b' or name == 'C':
                continue
            if name == 'z' or name == 'alpha' or name == 'w':
                parameter = getattr(net, name + '_act')
            elif name == 'r_bug' or name == 'r_met' or name == 'e_met' or name == 'sigma' or name == 'p':
                parameter = np.exp(parameter.clone().detach().numpy())
            elif name == 'pi_met':
                parameter = torch.softmax(parameter.clone().detach(), 1).numpy()
            if torch.is_tensor(parameter):
                param_dict[args.seed][name].append(parameter.clone().detach().numpy())
            else:
                param_dict[args.seed][name].append(parameter)
        if 'w' not in net.named_parameters():
            param_dict[args.seed]['w'].append(net.w_act.clone().detach().numpy())
        if args.mz:
            param_dict[args.seed]['z'].append(net.z_act.clone().numpy())
        if net.linear:
            param_dict[args.seed]['beta[1:,:]*alpha'].append(
                net.beta[1:, :].clone().detach().numpy() * net.alpha_act.clone().detach().numpy())
        if args.lw:
            if args.hard:
                param_dict[args.seed]['w_soft'].append(net.w_soft.clone().detach().numpy())
            param_dict[args.seed]['w_learned'].append(net.w.clone().detach().numpy())


        # if (epoch%1000 == 0 and epoch != 0):
        if epoch == last_epoch:
            print('Epoch ' + str(epoch) + ' Loss: ' + str(loss_vec[-1]))
            if args.schedule_lr:
                print('LR: ' + str(lr_tracking[-2]))
            print('')

            if 'epoch' not in path:
                path = path + 'epoch' + str(epoch) + '/'
            else:
                path = path.split('epoch')[0] + 'epoch' + str(epoch) + '/'
            if not os.path.isdir(path):
                os.mkdir(path)
            if net.embedding_dim == 2:
                plot_output_locations(path, net, -1, param_dict[args.seed], args.seed, gen_u,
                                      type='best_train', plot_zeros=False)
                plot_output_locations(path, net, -1, param_dict[args.seed], args.seed, gen_u,
                                      type='best_train', plot_zeros=True)
            print('Epoch ' + str(epoch))

            if epoch >= 1:
                best_mod = np.argmin(loss_vec)
                mapping = {}
                # mapping['bug'] = unmix_clusters(true_vals['mu_bug'], param_dict[args.seed]['mu_bug'][last_mod], param_dict[args.seed]['r_bug'][last_mod],
                #                                 true_vals['bug_locs'])
                # mapping['bug'] = pd.Series(mapping['bug']).sort_index()
                # mapping['met'] = unmix_clusters(true_vals['mu_met'], param_dict[args.seed]['mu_met'][last_mod], param_dict[args.seed]['r_met'][last_mod],
                #                                 true_vals['met_locs'])
                # mapping['met'] = pd.Series(mapping['met']).sort_index()
                # mapping['met'] = {i:i for i in np.arange(net.z_act.shape[1])}
                # mapping['met'] = pd.Series(mapping['met']).sort_index()
                # mapping['bug'] = {i:i for i in np.arange(net.w_act.shape[1])}
                # mapping['bug'] = pd.Series(mapping['bug']).sort_index()

                if 'epoch' not in path:
                    path = path + 'epoch' + str(epoch) + '/'
                else:
                    path = path.split('epoch')[0] + 'epoch' + str(epoch) + '/'
                if not os.path.isdir(path):
                    os.mkdir(path)

                if not os.path.isfile(path + 'Loss.txt'):
                    with open(path + 'Loss.txt', 'w') as f:
                        f.writelines('Seed ' + str(args.seed) + ', Lowest Loss: ' + str(np.min(loss_vec)) + '\n')
                else:
                    with open(path + 'Loss.txt', 'a') as f:
                        f.writelines('Seed ' + str(args.seed) + ', Lowest Loss: ' + str(np.min(loss_vec))+ '\n')

                plot_loss_dict(path, args.seed, loss_dict_vec)
                plot_interactions(path, best_mod, param_dict[args.seed], args.seed)
                plot_xvy(path, net, x, train_out_vec, best_mod, y, gen_z, gen_w, param_dict, args.seed)
                plot_param_traces(path, param_dict[args.seed], params2learn, true_vals, net, args.seed)
                fig3, ax3 = plt.subplots(figsize=(8, 8))
                fig3, ax3 = plot_loss(fig3, ax3, args.seed, np.arange(start, epoch + 1), loss_vec, lowest_loss=None)
                fig3.tight_layout()
                fig3.savefig(path_orig + 'loss_seed_' + str(args.seed) + '.pdf')
                plt.close(fig3)

                if args.schedule_lr:
                    fig4, ax4 = plt.subplots(1,1, figsize = (8, 8))
                    ax4.plot(np.arange(len(lr_tracking)), lr_tracking)
                    ax4.set_title(lr_tracking)
                    ax4.set_xlabel('iterations')
                    ax4.set_ylabel('learning rate')
                    fig4.savefig(path_orig + 'lr_seed_' + str(args.seed) + '.pdf')
                    plt.close(fig4)

                # if net.embedding_dim == 2:
                #     plot_output_locations(path, net, best_mod, param_dict[args.seed], args.seed, gen_u,
                #                           type='best_train', plot_zeros=False)
                #     plot_output_locations(path, net, best_mod, param_dict[args.seed], args.seed, gen_u,
                #                           type='best_train', plot_zeros=True)
                plot_output(path, path_orig, best_mod, train_out_vec, y, gen_z, param_dict[args.seed],
                                     args.seed, type = 'best_train')

                with open(path_orig + 'seed' + str(args.seed) + '.txt', 'w') as f:
                    f.writelines(str(epoch))

                etime= time.time()
                with open(path_orig + 'seed' + str(args.seed) + '_min_per_epoch.txt', 'w') as f:
                    f.writelines(str(epoch) + ': ' + str(np.round((etime - stime)/60, 3)) + ' minutes')


                save_dict = {'model_state_dict':net.state_dict(),
                           'optimizer_state_dict':optimizer.state_dict(),
                           'epoch': epoch}
                if args.schedule_lr:
                    save_dict['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(save_dict,
                           path + 'seed' + str(args.seed) + '_checkpoint.tar')
    etime = time.time()
    print('total time:' + str(etime - stime))
    print('delta loss:' + str(loss_vec[-1] - loss_vec[0]))

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("-learn", "--learn", help="params to learn", type=str, nargs='+', default = ['mu_bug','r_bug'])
    parser.add_argument("-lr", "--lr", help="params to learn", type=float, default = 0.001)
    parser.add_argument("-fix", "--fix", help="params to fix", type=str, nargs='+')
    parser.add_argument("-priors", "--priors", help="priors to set", type=str, nargs='+', default = ['mu_bug','r_bug'])
    parser.add_argument("-case", "--case", help="case", type=str, default = datetime.date.today().strftime('%m %d %Y').replace(' ','-'))
    # parser.add_argument("-case", "--case", help="case", type=str, default = 'base_case_2-16')
    parser.add_argument("-N_met", "--N_met", help="N_met", type=int, default = 20)
    parser.add_argument("-N_bug", "--N_bug", help="N_bug", type=int, default = 20)
    parser.add_argument("-L", "--L", help="number of microbe rules", type=int, default = 3)
    parser.add_argument("-K", "--K", help="metab clusters", type=int, default = 3)
    parser.add_argument("-meas_var", "--meas_var", help="measurment variance", type=float, default = 0.1)
    parser.add_argument("-iterations", "--iterations", help="number of iterations", type=int,default = 20500)
    parser.add_argument("-seed", "--seed", help = "seed for random start", type = int, default = 2)
    parser.add_argument("-load", "--load", help="0 to not load model, 1 to load model", type=int, default = 0)
    parser.add_argument("-rep_clust", "--rep_clust", help = "whether or not bugs are in more than one cluster", default = 0, type = int)
    parser.add_argument("-lb", "--lb", help = "whether or not to learn bug clusters", type = int, default = 0)
    parser.add_argument("-lm", "--lm", help = "whether or not to learn metab clusters", type = int, default = 0)
    parser.add_argument("-N_samples", "--N_samples", help="num of samples", type=int, default=1000)
    parser.add_argument("-linear", "--linear", type = int, default = 1)
    parser.add_argument("-nltype", "--nltype", type = str, default = "exp")
    parser.add_argument("-hyper_r", "--hyper_r", type=int, default=0)
    parser.add_argument("-hyper_mu","--hyper_mu", type = int, default = 0)
    parser.add_argument("-adjust_lr", "--adjust_lr", type=int, default=1)
    parser.add_argument("-l1", "--l1", type=int, default=0)
    parser.add_argument("-mz","--mz", type = int, default = 1)
    parser.add_argument("-lw", "--lw", type=int, default=1)
    parser.add_argument("-adjust_mvar", "--adjust_mvar", type=int, default=0)
    parser.add_argument("-sample_mu", "--sample_mu", type=int, default=0)
    parser.add_argument("-dist_var_perc", "--dist_var_perc", type=float, default=0.5)
    parser.add_argument("-schedule_lr", "--schedule_lr", type=int, default=1)
    parser.add_argument("-p_num", "--p_num", type=int, default=0)
    parser.add_argument("-p", "--p", type=float, default=0.001)
    parser.add_argument("-learn_mvar", "--learn_mvar", type=int, default=1)
    parser.add_argument("-dim", "--dim", type=int, default=2)
    parser.add_argument("-hard", "--hard", type=int, default=0)
    parser.add_argument("-a_tau", "--a_tau", type=float, nargs = '+', default=[-0.5, -3])
    parser.add_argument("-w_tau", "--w_tau", type=float, nargs='+', default=[-0.01, -1])
    parser.add_argument("-w_tau2", "--w_tau2", type=float, nargs='+', default=[-0.01, -1])
    args = parser.parse_args()

    # try:
    run_learner(args, device)
    # except:
    #     print('FAILURE')
    #     print(args)
    #     print('')