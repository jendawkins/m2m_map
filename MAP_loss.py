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
from torch.distributions.beta import Beta
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
            temp_dist = Normal(outputs.T, torch.sqrt(torch.exp(self.net.sigma)))
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
        if torch.isnan(total_loss):
            print('total loss is nan')
        return total_loss

    def marginalized_loss(self, outputs, true):

        eye = torch.eye(self.net.embedding_dim).unsqueeze(0).expand(self.net.K, -1, -1)
        var = torch.exp(self.net.r_met).unsqueeze(-1).unsqueeze(-1).expand(-1,self.net.embedding_dim,
                                                                           self.net.embedding_dim)*eye
        eps = 1e-10
        temp = (1-2*eps)*torch.softmax(self.net.pi_met,1) + eps
        z_log_probs = torch.log(temp.T).unsqueeze(1) + MultivariateNormal(
            self.net.mu_met.unsqueeze(1).expand(-1,self.net.N_met,-1),var.unsqueeze(1).expand(
                -1,self.net.N_met,-1,-1)).log_prob(
            torch.Tensor(self.net.met_locs)).unsqueeze(1) + \
               Normal(outputs.T.unsqueeze(-1).expand(-1,-1,self.net.N_met), torch.sqrt(torch.exp(self.net.sigma))).log_prob(true)
        self.net.z_act = nn.functional.one_hot(torch.argmax(z_log_probs.sum(1),0),self.net.K)
        loss = -torch.logsumexp(z_log_probs, 0).sum(1).sum()
        return loss

    def alpha_loss(self):
        self.loss_dict['alpha'] = -BinaryConcrete(self.net.alpha_loc, self.net.alpha_temp).log_prob(self.net.alpha_soft).sum().sum()
        if self.net.learn_num_bug_clusters:
            L_active = self.net.alpha_act.sum(0)
            nb = -NegativeBinomial(self.net.L, torch.exp(self.net.p)).log_prob(L_active).sum()
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

    def mu_met_loss(self):
        temp_dist = self.net.distributions['mu_met']
        self.loss_dict['mu_met'] = -temp_dist.log_prob(self.net.mu_met).sum()

    def r_met_loss(self):
        val = 1 / torch.exp(self.net.r_met)
        gamma = self.net.distributions['r_met']
        self.loss_dict['r_met'] = -gamma.log_prob(val).sum()

    def pi_met_loss(self):
        epsilon = torch.exp(self.net.e_met)
        eps = 1e-10
        temp = (1-2*eps)*torch.softmax(self.net.pi_met,1) + eps
        self.loss_dict['pi_met'] = (torch.Tensor(1 - epsilon) * torch.log(temp)).sum()

    def e_met_loss(self):
        val = torch.exp(self.net.e_met)
        gamma = self.net.distributions['e_met']
        self.loss_dict['e_met'] = -gamma.log_prob(val).sum()

    def sigma_loss(self):
        gamma_log_prob = Gamma(1,1).log_prob(1/torch.exp(self.net.sigma))
        self.loss_dict['sigma'] = -gamma_log_prob

