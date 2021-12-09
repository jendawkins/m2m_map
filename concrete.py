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
import pandas as pd

class BinaryConcrete():
    def __init__(self, loc, tau):
        self.loc = loc
        self.tau = tau
        if not torch.is_tensor(self.loc):
            self.loc = torch.Tensor([self.loc])
        if not torch.is_tensor(self.tau):
            self.tau = torch.Tensor([self.tau])

    def sample(self, size=[1]):
        L = st.logistic(0,1).rvs(size)
        return 1/(1 + torch.exp(-(torch.log(self.loc) + L)/self.tau))

    def pdf(self, x):
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                x = torch.Tensor([x])
        top = self.tau*self.loc*(x.pow(-self.tau-1))*((1-x).pow(-self.tau-1))
        bottom = (self.loc*(x.pow(-self.tau)) + (1-x).pow(-self.tau)).pow(2)
        return top / bottom


class Concrete():
    def __init__(self, loc, tau):
        self.loc = loc
        self.tau = tau
        if not torch.is_tensor(self.loc):
            self.loc = torch.Tensor(self.loc)
        if not torch.is_tensor(self.tau):
            self.tau = torch.Tensor([self.tau])

    def sample(self, size = 1):
        G = torch.distributions.gumbel.Gumbel(0,10).sample([size, self.loc.shape[1]])
        top = torch.exp((torch.log(self.loc.squeeze()) + G)/self.tau)
        bottom = (torch.exp((torch.log(self.loc.squeeze()) + G)/self.tau)).sum(1)
        return torch.div(top.T, bottom).T

    def pdf(self, x):
        if not torch.is_tensor(x):
            x = torch.Tensor(x)
        n = len(x)
        C = math.factorial(n-1)*self.tau.pow(n-1)
        tot = ((self.loc.squeeze()*x.pow(-self.tau-1))/(self.loc.squeeze()*x.pow(-self.tau)).sum(0)).prod(0)
        return C*tot

class MultDist():
    def __init__(self, concrete_dist, mvn):
        self.concrete = concrete_dist
        self.mvn = mvn

    def pdf(self, zw, a):
        first = self.concrete.pdf(zw)
        second = first*torch.stack([zw[l]*torch.exp(self.mvn[l].log_prob(a)) for l in range(len(self.mvn))]).sum(0)
        return second

    def make_pdf_table(self, a_samples = None, size = 100):
        zw_samples = self.concrete.sample(size = size)
        if a_samples is None:
            a_samples = torch.cat([self.mvn[l].sample([size]) for l in range(len(self.mvn))])
        a_samp_tuples = [tuple(a_samp) for a_samp in a_samples]
        z_samp_tuples = [tuple(z_samp) for z_samp in zw_samples]
        self.prob_table = pd.DataFrame(np.zeros((len(a_samp_tuples),size)), index = a_samp_tuples,
                                       columns = z_samp_tuples)
        for zsamp in self.prob_table.columns.values:
            for asamp in self.prob_table.index.values:
                self.prob_table.loc[asamp, zsamp] = self.pdf(torch.stack(zsamp), torch.stack(asamp))

    # def sample(self, size = [1]):
    #     zw_samples = self.concrete.sample(size = [100])
    #     a_samples = self.mvn.sample(size = [100])
    #     feature_table = np.zeros((100,100))
    #     prob_table = np.zeros((100,100))
    #     for i in range(len(zw_samples)):
    #         for j in range(len(a_samples)):
    #             feature_table[i,j] = (zw_samples[i], a_samples[j])
    #             prob_table[i,j] = self.pdf(zw_samples[i], a_samples[j])

