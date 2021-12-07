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
import torch.nn as nn
import time
import numpy as np

class MAPloss():
    def __init__(self,net, meas_var=1,
                 compute_loss_for = ['alpha','beta','w','z','mu_bug','r_bug','pi_bug','mu_met','r_met','pi_met']):
        self.net = net
        self.meas_var = meas_var
        self.loss_dict = {}
        self.compute_loss_for = compute_loss_for

    def compute_loss(self, outputs, true):
        s1, s2 = outputs.shape[1], true.shape[1]
        temp_dist = Normal(outputs.T, np.sqrt(self.meas_var))
        log_probs = torch.stack([temp_dist.log_prob(true[:,j]) for j in range(true.shape[1])])
        log_probs[np.where(log_probs < -103)] = -103
        self.loss_dict['y'] = -torch.log((self.net.z_act.unsqueeze(-1).repeat(1,1,log_probs.shape[-1])*torch.exp(log_probs)).sum(1)).sum()
        # criterion2 = nn.L1Loss()
        # self.loss_dict['y'] = criterion2(outputs, true)
        total_loss = 0
        for param in self.compute_loss_for:
            fun = getattr(self, param + '_loss')
            fun()
            total_loss += self.loss_dict[param]
        total_loss += self.loss_dict['y']
        return total_loss

    def alpha_loss(self):
        self.loss_dict['alpha'] = ((self.net.temp_selector + 1) * torch.log(self.net.alpha_act) + (self.net.temp_selector + 1) * torch.log(
            1 - self.net.alpha_act) - \
                              2 * self.net.temp_selector * torch.log(self.net.alpha_act) - 2 * self.net.temp_selector * torch.log(
                    1 - self.net.alpha_act) + \
                              torch.log(torch.tensor(self.net.alpha_loc))).sum()


    def beta_loss(self):
        temp_dist = self.net.distributions['beta']
        self.loss_dict['beta'] = -temp_dist.log_prob(self.net.beta).sum()

    def w_loss(self):
        temp_dist = [MultivariateNormal(self.net.mu_bug[k,:], (torch.eye(self.net.mu_bug.shape[1]) *
                                                               torch.exp(self.net.r_bug[k])).float()) for k in np.arange(self.net.mu_bug.shape[0])]

        probs = torch.stack([torch.exp(torch.stack([temp_dist[k].log_prob(torch.FloatTensor(self.net.microbe_locs[m, :])) for
                                        k in np.arange(self.net.mu_bug.shape[0])])) for m in np.arange(self.net.microbe_locs.shape[0])])
        if probs.shape[1]< self.net.w_act.shape[1]:
            probs = torch.cat((probs, np.median(probs)*torch.ones((probs.shape[0], 1))), 1)
        probs[probs<1e-44] = 1e-44
        self.loss_dict['w'] = self.net.temp_grouper*(torch.stack([(-torch.log(torch.softmax(self.net.pi_bug,1)) +
                                                                   (self.net.temp_grouper + 1) * torch.log(self.net.w_act[m, :]) +
                                       torch.log((torch.softmax(self.net.pi_bug,1) *
                                                  (self.net.w_act[m, :].pow(-self.net.temp_grouper))).sum())).sum() -
                                      torch.log(torch.matmul(self.net.w_act[m, :],probs[m,:]))
                                      for m in np.arange(self.net.microbe_locs.shape[0])]).sum(0))
    def mu_bug_loss(self):
        temp_dist = self.net.distributions['mu_bug']
        self.loss_dict['mu_bug'] = -temp_dist.log_prob(self.net.mu_bug).sum()

    def r_bug_loss(self):
        gamma = self.net.distributions['r_bug']
        val = 1 / torch.exp(self.net.r_bug)
        val = torch.clamp(val, min=1e-20)
        self.loss_dict['r_bug'] = -gamma.log_prob(val).sum()
        # if torch.isnan(self.loss_dict['r_bug']).any() or torch.isinf(self.loss_dict['r_bug']).any():
        #     print('debug')

    def pi_bug_loss(self):
        self.loss_dict['pi_bug'] = (torch.Tensor(1 - np.array(self.net.params['pi_bug']['epsilon'])) * torch.softmax(self.net.pi_bug,1)).sum()

    def z_loss(self):
        temp_dist = [MultivariateNormal(self.net.mu_met[k,:], (torch.eye(self.net.mu_met.shape[1]) *
                                                               torch.exp(self.net.r_met[k])).float()) for k in np.arange(self.net.mu_met.shape[0])]

        probs = torch.stack([torch.exp(torch.stack([temp_dist[k].log_prob(torch.FloatTensor(self.net.met_locs[m, :])) for
                                        k in np.arange(self.net.mu_met.shape[0])])) for m in np.arange(self.net.met_locs.shape[0])])
        probs[probs<1e-44] = 1e-44
        self.loss_dict['z'] = self.net.temp_grouper*(torch.stack([(-torch.log(torch.softmax(self.net.pi_met,1)) +
                                                                   (self.net.temp_grouper + 1) * torch.log(self.net.z_act[m, :]) +
                                       torch.log((torch.softmax(self.net.pi_met,1) *
                                                  (self.net.z_act[m, :].pow(-self.net.temp_grouper))).sum())).sum() -
                                      torch.log(torch.matmul(self.net.z_act[m, :],probs[m,:]))
                                      for m in np.arange(self.net.met_locs.shape[0])]).sum(0))

    def mu_met_loss(self):
        temp_dist = self.net.distributions['mu_met']
        self.loss_dict['mu_met'] = -temp_dist.log_prob(self.net.mu_met).sum()

    def r_met_loss(self):
        val = 1 / torch.exp(self.net.r_met)
        val = torch.clamp(val, min=1e-20)
        gamma = self.net.distributions['r_met']
        self.loss_dict['r_met'] = -gamma.log_prob(val).sum()
        # if torch.isnan(self.loss_dict['r_met']).any() or torch.isinf(self.loss_dict['r_met']).any():
        #     print('debug')
        # loss_dict['r_met'] = ((net.r_scale_met*net.r0_bug)/(2*net.r_met) +
        #              (1 + net.r_scale_met/2)*torch.log(net.r_met)).sum()
    def pi_met_loss(self):
        self.loss_dict['pi_met'] = (torch.Tensor(1 - np.array(self.net.params['pi_met']['epsilon'])) * torch.log(torch.softmax(self.net.pi_met,1))).sum()

