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

        # if self.net.marginalize_w and ('mu_bug' in self.compute_loss_for or 'r_bug' in self.compute_loss_for):
        #     self.loss_dict['w'] = self.w_loss()
        #     total_loss += self.loss_dict['w']
        if torch.isnan(total_loss):
            print('total loss is nan')
        return total_loss

    def marginalized_loss(self, outputs, true):

        eye = torch.eye(self.net.embedding_dim).unsqueeze(0).expand(self.net.K, -1, -1)
        var = torch.exp(self.net.r_met).unsqueeze(-1).unsqueeze(-1).expand(-1,self.net.embedding_dim,
                                                                           self.net.embedding_dim)*eye

        z_log_probs = torch.log(torch.softmax(self.net.pi_met, 1).T).unsqueeze(1) + MultivariateNormal(
            self.net.mu_met.unsqueeze(1).expand(-1,self.net.N_met,-1),var.unsqueeze(1).expand(
                -1,self.net.N_met,-1,-1)).log_prob(
            torch.Tensor(self.net.met_locs)).unsqueeze(1) + \
               Normal(outputs.T.unsqueeze(-1).expand(-1,-1,self.net.N_met), torch.sqrt(torch.exp(self.net.sigma))).log_prob(true)

        # if 'r_met' in self.net.compute_loss_for or 'mu_met' in self.net.compute_loss_for or 'pi_met' in self.net.compute_loss_for:
        self.net.z_act = nn.functional.one_hot(torch.argmax(z_log_probs.sum(1),0),self.net.K)
        loss = -torch.logsumexp(z_log_probs, 0).sum(1).sum()

        return loss

    def w_loss(self):
        eye = torch.eye(self.net.embedding_dim).unsqueeze(0).expand(self.net.L, -1, -1)
        var = torch.exp(self.net.r_bug).unsqueeze(-1).unsqueeze(-1).expand(-1,self.net.embedding_dim,
                                                                           self.net.embedding_dim)*eye

        if self.net.learn_w!=0:
            temp = MultivariateNormal(self.net.mu_bug.unsqueeze(1).expand(
                -1,self.net.N_bug,-1),var.unsqueeze(1).expand(-1,self.net.N_bug,-1,-1)
                          ).log_prob(torch.Tensor(self.net.microbe_locs)).T

            if self.net.hard_w:
                loss =-(BinaryConcrete(self.net.w_loc, self.net.omega_temp).log_prob(self.net.w_soft).sum(1) + \
                  torch.log((torch.exp(temp)*self.net.w_act).sum(1) + 1e-10)).sum()
            else:
                loss = -(BinaryConcrete(self.net.w_loc, self.net.omega_temp).log_prob(self.net.w_act).sum(1) + \
                         torch.log((torch.exp(temp) * self.net.w_act).sum(1) + 1e-10)).sum()

        # if self.net.learn_w==2:
            # w_log_probs = Bernoulli(0.1).log_prob(torch.tensor(1).float()) + MultivariateNormal(self.net.mu_bug.unsqueeze(1).
            #                                                                              expand(-1,self.net.N_bug,-1),var.unsqueeze(1).expand(
            #         -1,self.net.N_bug,-1,-1)).log_prob(
            #     torch.Tensor(self.net.microbe_locs))
            # w_log_probs[w_log_probs>0] = 0
            # temp = torch.exp(w_log_probs.detach()).T
            # norm = torch.stack([torch.logsumexp(w_log_probs.index_fill(-1,i,0),1) for i in torch.arange(w_log_probs.shape[1])])
            # loss = -torch.logsumexp(w_log_probs,1).sum()
            self.loss_dict['w'] = loss


    def p_loss(self):
        self.loss_dict['p'] = -Beta(0.01,10).log_prob(torch.exp(self.net.p))

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
            gamma = self.net.distributions['r_met']
            self.loss_dict['r_met'] = -gamma.log_prob(val).sum()


    def pi_met_loss(self):
        epsilon = torch.exp(self.net.e_met)
        self.loss_dict['pi_met'] = (torch.Tensor(1 - epsilon) * torch.log(torch.softmax(self.net.pi_met,1))).sum()

    def e_met_loss(self):
        val = torch.exp(self.net.e_met)
        gamma = self.net.distributions['e_met']
        self.loss_dict['e_met'] = -gamma.log_prob(val).sum()


    def sigma_loss(self):
        gamma_log_prob = Gamma(1,1).log_prob(1/torch.exp(self.net.sigma))
        self.loss_dict['sigma'] = -gamma_log_prob

