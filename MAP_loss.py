import scipy.stats as st
from collections import defaultdict
import pickle as pkl
from datetime import datetime
from statsmodels.stats.multitest import multipletests
import random
import torch
import math
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.binomial import Binomial
import torch.nn as nn
import time
import numpy as np
from concrete import *
from helper import *

class MAPloss():
    def __init__(self,net):
        self.net = net
        self.loss_dict = {}
        self.compute_loss_for = net.compute_loss_for

    def compute_loss(self, outputs, true):
        if self.net.marginalize_z:
            self.loss_dict['y'] = self.marginalized_loss(outputs, true)
        else:
            temp_dist = Normal(outputs.T, torch.sqrt(self.net.meas_var))
            log_probs = torch.stack([temp_dist.log_prob(true[:,j]) for j in np.arange(true.shape[1])])
            self.loss_dict['y'] = -torch.log((self.net.z_act.unsqueeze(-1).repeat(1,1,log_probs.shape[-1])*
                                              torch.exp(log_probs)).sum(1)).sum()
        total_loss = 0
        for param in self.compute_loss_for:
            if len(param)>0:
                fun = getattr(self, param + '_loss')
                fun()
                total_loss += self.loss_dict[param]
        total_loss += self.loss_dict['y']
        return total_loss

    def marginalized_loss(self, outputs, true):
        c_hat = torch.tensor(np.array(list(itertools.product(np.arange(self.net.K),repeat = self.net.N_met))))
        mu = self.net.mu_met[c_hat,:]
        eye = torch.eye(self.net.embedding_dim).unsqueeze(0).unsqueeze(0).expand(c_hat.shape[0], c_hat.shape[1], -1,-1)
        r = self.net.r_met[c_hat].unsqueeze(-1).unsqueeze(-1).expand(-1,-1,self.net.embedding_dim,self.net.embedding_dim)*eye
        z_hat = nn.functional.one_hot(c_hat, self.net.K)

        loss = -torch.logsumexp((MultivariateNormal(mu, r).log_prob(torch.Tensor(self.net.met_locs)) +
                    (Categorical(self.net.pi_met).log_prob(z_hat)*z_hat).sum(-1) +
                    Normal(outputs.T[c_hat, :],
                           self.net.meas_var.unsqueeze(0).unsqueeze(0).expand(
                               c_hat.shape[-1], outputs.shape[0])).log_prob(true.T).sum(-1)).sum(1), 0)

        return loss



    def alpha_loss(self):
        self.loss_dict['alpha'] = ((self.net.temp_scheduled + 1) * torch.log(self.net.alpha_act*
                                                                             self.net.temp_scheduled*self.net.alpha_loc)
                                   + (self.net.temp_scheduled + 1) * torch.log(1 - self.net.alpha_act) +
                              2 * torch.log(self.net.alpha_act.pow(-self.net.temp_scheduled)*self.net.alpha_loc +
                                            (1-self.net.alpha_act).pow(-self.net.temp_scheduled))).sum()
        if self.net.learn_num_bug_clusters:
            L_active = self.net.alpha_act.sum(0)
            nb = -NegativeBinomial(self.net.L_sm, 0.5).log_prob(L_active).sum()
            self.loss_dict['alpha'] += nb

    def beta_loss(self):
        temp_dist = self.net.distributions['beta']
        self.loss_dict['beta'] = -temp_dist.log_prob(self.net.beta).sum()

    def mu_bug_loss(self):
        temp_dist = self.net.distributions['mu_bug']
        self.loss_dict['mu_bug'] = -temp_dist.log_prob(self.net.mu_bug).sum()

    def r_bug_loss(self):
        gamma = self.net.distributions['r_bug']
        val = 1 / torch.exp(self.net.r_bug)
        val = torch.clamp(val, min=1e-10)
        self.loss_dict['r_bug'] = -gamma.log_prob(val).sum()
        # if torch.isnan(self.loss_dict['r_bug']).any() or torch.isinf(self.loss_dict['r_bug']).any():
        #     print('debug')

    # def pi_bug_loss(self):
    #     self.loss_dict['pi_bug'] = (torch.Tensor(1 - np.array(self.net.params['pi_bug']['epsilon'])) * torch.softmax(self.net.pi_bug,1)).sum()
    def z_loss(self):
        # probability of all K clusters for all M metabolites
        con = Concrete(torch.softmax(self.net.pi_met, 1), self.net.temp_scheduled)
        self.loss_dict['z'] = -torch.stack(
            [torch.log(torch.stack([self.net.z_act[m,k]*torch.exp(MultivariateNormal(self.net.mu_met[k, :], (torch.eye(self.net.mu_met.shape[1]) *
                                                                torch.exp(self.net.r_met[k])).float()).log_prob(
                torch.FloatTensor(self.net.met_locs[m, :]))) for k in np.arange(self.net.mu_met.shape[0])]).sum()) +
             torch.log(con.pdf(self.net.z_act[m,:])) for m in
             np.arange(self.net.met_locs.shape[0])]).sum()
        # con = Concrete(torch.softmax(self.net.pi_met, 1), self.net.temp_scheduled)
        # probs2 = con.pdf(self.net.z_act[m,:])
        # self.loss_dict['z'] = self.net.temp_scheduled*(torch.stack([(-torch.log(torch.softmax(self.net.pi_met,1)) +
        #                                                            (self.net.temp_scheduled + 1) * torch.log(self.net.z_act[m, :]) +
        #                                torch.log((torch.softmax(self.net.pi_met,1) *
        #                                           (self.net.z_act[m, :].pow(-self.net.temp_scheduled))).sum())).sum() -
        #                               torch.log(torch.matmul(self.net.z_act[m, :],probs[m,:]))
        #                               for m in np.arange(self.net.met_locs.shape[0])]).sum(0))

    # def z_loss(self):
    #     mvn = [MultivariateNormal(self.net.mu_met[k,:], (torch.eye(self.net.mu_met.shape[1]) *
    #                                                            torch.exp(self.net.r_met[k])).float()) for k in np.arange(self.net.mu_met.shape[0])]
    #     con = Concrete(torch.softmax(self.net.pi_met,1), self.net.temp_scheduled)
    #     multi = MultDist(con, mvn)
    #     log_probs = torch.stack([-torch.log(multi.pdf(self.net.z_act[m,:], torch.Tensor(self.net.met_locs[m,:])))
    #                              for m in np.arange(self.net.met_locs.shape[0])]).sum()
    #     self.loss_dict['z'] = log_probs
    def mu_met_loss(self):
        if self.net.mu_hyper:
            Lambda = torch.diag(self.net.lambda_mu)
            Bo = Lambda @ self.net.Ro @ Lambda
            mvn = MultivariateNormal(self.net.b, Bo)
            self.loss_dict['mu_met'] = -mvn.log_prob(self.net.mu_met).sum()

            gamma = Gamma(0.5, 0.5)
            self.loss_dict['mu_met'] += -gamma.log_prob(torch.exp(self.net.lambda_mu)).sum()

            mvn = MultivariateNormal(torch.zeros(self.net.embedding_dim), torch.eye(self.net.embedding_dim, self.net.embedding_dim))
            self.loss_dict['mu_met'] += -mvn.log_prob(self.net.b).sum()
        else:
            temp_dist = self.net.distributions['mu_met']
            self.loss_dict['mu_met'] = -temp_dist.log_prob(self.net.mu_met).sum()

    def r_met_loss(self):
        if self.net.r_hyper:
            gamma = Gamma(self.net.c, self.net.C)
            self.loss_dict['r_met'] = -gamma.log_prob(1 / torch.exp(self.net.r_met)).sum()

            gamma = Gamma(self.net.g, self.net.G)
            self.loss_dict['r_met'] += -gamma.log_prob(1 / torch.exp(self.net.C)).sum()
        else:
            val = 1 / torch.exp(self.net.r_met)
            val = torch.clamp(val, min=1e-20)
            gamma = self.net.distributions['r_met']
            self.loss_dict['r_met'] = -gamma.log_prob(val).sum()


    def pi_met_loss(self):
        epsilon = torch.exp(self.net.e_met)
        self.loss_dict['pi_met'] = (torch.Tensor(1 - epsilon) * torch.log(torch.softmax(self.net.pi_met,1))).sum()

    def e_met_loss(self):
        val = torch.exp(self.net.e_met)
        gamma = self.net.distributions['e_met']
        self.loss_dict['e_met'] = -gamma.log_prob(val).sum()

    def rad_mu_loss(self):
        kappa = torch.stack([torch.sqrt(((self.net.mu_bug - torch.tensor(
            self.net.microbe_locs[m,:])).pow(2)).sum(-1)) for m in range(self.net.microbe_locs.shape[0])])
        num_gzero = [len(torch.where((torch.exp(self.net.r_bug[l]) - kappa[:,l])>0)[0]) for l in range(len(self.net.r_bug))]
        binom = Binomial(self.net.microbe_locs.shape[0], (1/len(self.net.r_bug)))
        log_prob = -binom.log_prob(torch.Tensor(np.array(num_gzero))).sum()
        self.loss_dict['rad_mu'] = log_prob

