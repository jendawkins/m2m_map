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
        self.meas_var = net.meas_var
        self.loss_dict = {}
        self.compute_loss_for = net.compute_loss_for

    #@profile
    def compute_loss(self, outputs, true):
        temp_dist = Normal(outputs.T, np.sqrt(self.meas_var))
        log_probs = torch.stack([temp_dist.log_prob(true[:,j]) for j in range(true.shape[1])])
        log_probs = torch.clamp(log_probs, min = -103, max = 103)
        self.loss_dict['y'] = -torch.log((self.net.z_act.unsqueeze(-1).repeat(1,1,log_probs.shape[-1])*torch.exp(log_probs)).sum(1)).sum()
        total_loss = 0
        for param in self.compute_loss_for:
            fun = getattr(self, param + '_loss')
            fun()
            total_loss += self.loss_dict[param]
        total_loss += self.loss_dict['y']
        return total_loss

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
        mvn = [MultivariateNormal(self.net.mu_met[k,:], (torch.eye(self.net.mu_met.shape[1]) *
                                                               torch.exp(self.net.r_met[k])).float()) for k in np.arange(self.net.mu_met.shape[0])]
        con = Concrete(torch.softmax(self.net.pi_met,1), self.net.temp_scheduled)
        multi = MultDist(con, mvn)
        log_probs = torch.stack([-torch.log(multi.pdf(self.net.z_act[m,:], torch.Tensor(self.net.met_locs[m,:])))
                                 for m in np.arange(self.net.met_locs.shape[0])]).sum()
        self.loss_dict['z'] = log_probs

    def mu_met_loss(self):
        temp_dist = self.net.distributions['mu_met']
        self.loss_dict['mu_met'] = -temp_dist.log_prob(self.net.mu_met).sum()

    def r_met_loss(self):
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

