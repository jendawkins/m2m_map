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

    def sample(self, size = [1]):
        G = torch.distributions.gumbel.Gumbel(0,1).sample(size)
        return torch.exp((torch.log(self.loc) + G)/self.tau
                         )/(torch.exp((torch.log(self.loc) + G)/self.tau)).sum(0)

    def pdf(self, x):
        if not torch.is_tensor(x):
            x = torch.Tensor(x)
        n = len(x)
        C = math.factorial(n-1)*self.tau.pow(n-1)
        tot = ((self.loc*x.pow(-self.tau-1))/(self.loc*x.pow(-self.tau)).sum(0)).prod(0)
        return C*tot


