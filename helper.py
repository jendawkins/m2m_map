import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from copy import copy, deepcopy
from sklearn.model_selection import StratifiedKFold
import os
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

def get_one_hot(x,l=None):
    if l is None:
        l = len(np.unique(x))
    vec = np.zeros(l)
    vec[x] = 1
    return vec

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def isclose(a, b, tol=1e-03):
    return (abs(a-b) <= tol).all()

def concrete_sampler(a, T, dim, loc = 0, scale = 1):
    G = st.gumbel_r(loc, scale).rvs(size = a.shape)
    return torch.softmax(T*(torch.log(torch.Tensor(a)) + torch.Tensor(G)), dim = dim)

# def concrete_logpdf(a, x, T):
#     n = len(a)
#     p1 = np.log(math.factorial(n-1))
#     p2 = (n-1)*np.log(T)
#     p3 =


def jarvis(points):
      """
      Jarvis Convex Hull algorithm.
      points is a list of CGAL.Point_2 points
      """
      points_arr = np.array(points)
      min_x_ix = np.argmin(points_arr[:,0])
      r0 = points[min_x_ix]
      hull = [r0]
      startPoint = points_arr[min_x_ix, :]
      remainingPoints = [x for x in points if x not in hull]
      while remainingPoints:
            endPoint = random.choice(remainingPoints)
            endPoint_diff = startPoint - np.array(endPoint)
            if endPoint_diff[0] == 0:
                endPoint_ang = np.pi / 2
            else:
                endPoint_ang = np.arctan(endPoint_diff[1] / endPoint_diff[0])
            for i,t in enumerate(remainingPoints):
                if t != endPoint and t!= tuple(startPoint):
                    diff = startPoint - points_arr[i,:]
                    if diff[0] == 0:
                        ang = np.pi/2
                    else:
                        ang = np.arctan(diff[1]/diff[0])
                    if ang > endPoint_ang:
                        endPoint = t
                        endPoint_ang = copy(ang)
                    print(t)
                    print(ang)
                    print(endPoint)
                    print('')
            endVec = np.array(endPoint) - np.array(r0)
            if endVec[0] == 0:
                endAngle = np.pi/2
            else:
                endAngle = np.arctan(endVec[1]/endVec[0])
            print(endPoint)
            if endAngle < endPoint_ang:
                break
            hull.append(endPoint)
            startPoint = endPoint
            remainingPoints = [x for x in points if x not in hull]
      return hull

def get_epsilon(data):
    vals = np.array(data).flatten()
    vals[np.where(vals == 0)[0]] = 1
    epsilon = 0.1 * np.min(np.abs(vals))
    return epsilon

def filter_by_train_set(x_train, x_test, meas_key, key = 'metabs', log_transform = True, standardize_data = True):
    if x_train.shape[1]>10:
        filt1 = filter_by_pt(x_train, perc=meas_key['pt_perc'][key], pt_thresh=meas_key['pt_tmpts'][key],
                             meas_thresh=meas_key['meas_thresh'][key], weeks=None)
        filt1_test = x_test[filt1.columns.values]
    else:
        filt1 = x_train
        filt1_test = x_test

    epsilon = get_epsilon(filt1)
    if '16s' in key:
        filt1_test = np.divide(filt1_test.T, np.sum(filt1_test,1)).T
        epsilon = get_epsilon(filt1_test)
        geom_means = np.exp(np.mean(np.log(filt1 + epsilon), 1))
        temp = np.divide(filt1.T, geom_means).T
        epsilon = get_epsilon(temp)
    xout = []
    for x in [filt1, filt1_test]:
        if '16s' not in key:
            if log_transform:
                transformed = np.log(x+ epsilon)
            else:
                transformed = x
        else:
            if log_transform:
                x = np.divide(x.T, np.sum(x,1)).T
                geom_means = np.exp(np.mean(np.log(x + epsilon), 1))
                transformed = np.divide(x.T, geom_means).T
                transformed = np.log(transformed + epsilon)
            else:
                transformed = x
        xout.append(transformed)
    xtr, xtst = xout[0], xout[1]

    if x_train.shape[1]>10:
        filt2 = filter_vars(xtr, perc=meas_key['var_perc'][key], weeks = None)
    else:
        filt2 = xtr
    filt2_test = xtst[filt2.columns.values]
    if standardize_data:
        dem = np.std(filt2,0)
        if (dem == 0).any():
            dem = np.where(np.std(filt2, 0) == 0, 1, np.std(filt2, 0))
        x_train_out = (filt2 - np.mean(filt2, 0))/dem
        x_test_out = (filt2_test - np.mean(filt2,0))/dem
    else:
        x_train_out = filt2
        x_test_out = filt2_test
    return x_train_out, x_test_out

def filter_vars(data, perc=5, weeks = [0,1,2]):

    if weeks:
        tmpt = [float(x.split('-')[1]) for x in data.index.values]
        rm2 = []
        for week in weeks:
            t_ix = np.where(np.array(tmpt)==week)[0]
            dat_in = data.iloc[t_ix,:]
            variances = np.std(dat_in, 0) / np.abs(np.mean(dat_in, 0))
            rm = np.where(variances > np.percentile(variances, perc))[0].tolist()
            rm2.extend(rm)
        rm2 = np.unique(rm2)
    else:
        variances = np.std(data, 0) / np.abs(np.mean(data, 0))
        rm2 = np.where(variances > np.percentile(variances, perc))[0].tolist()

    temp = data.iloc[:,rm2]
    # import pdb; pdb.set_trace()
    # if len(np.where(np.sum(temp,0)==0)[0]) > 0:
    #     import pdb; pdb.set_trace()
    return data.iloc[:,list(rm2)]

def MAPloss(outputs, true, net, meas_var=10, compute_loss_for = ['y', 'alpha','beta','w','z','mu_bug','r_bug','pi_bug','mu_met','r_met','pi_met']):
    loss_dict = {}
    # loss_dict['y'] = -torch.log(torch.matmul((1/np.sqrt(2*np.pi*meas_var**2))*torch.exp(
    #     (-1/(2*meas_var**2))*(true-outputs)**2),net.z_act).sum(1)).sum()
    # temp = (1/np.sqrt(2*np.pi*meas_var**2))*torch.exp((-1/(2*meas_var**2))*(true-outputs)**2)
    # y_loss_2 = -torch.log(torch.stack([temp*net.z_act[:,k] for k in range(net.K)]).sum(0)).sum()

    # criterion = nn.L1Loss()
    temp_dist = Normal(outputs, np.sqrt(meas_var))
    log_probs = temp_dist.log_prob(true)
    log_probs[np.where(log_probs < -103)] = -103
    loss_dict['y'] = (-torch.log(torch.matmul(torch.exp(log_probs), net.z_act))).sum()
    # loss_dict['y'] = criterion(outputs, true)
    loss_dict['alpha'] = ((net.temp_selector + 1)*torch.log(net.alpha_act) + (net.temp_selector + 1)*torch.log(1-net.alpha_act) - \
                 2*net.temp_selector*torch.log(net.alpha_act) - 2*net.temp_selector*torch.log(1-net.alpha_act) + \
                         torch.log(torch.tensor(net.alpha_loc))).sum()

    temp_dist = Normal(torch.zeros(net.beta.shape), np.sqrt(net.beta_var))
    loss_dict['beta'] = -temp_dist.log_prob(net.beta).sum()
    # loss_dict['beta'] = ((1/(2*net.beta_var))*(net.beta**2)).sum()

    # loss_dict['w'] = torch.stack([(-torch.log(net.pi_bug) + (net.temp_grouper + 1)*torch.log(net.w_act[m,:]) +
    #         torch.log((net.pi_bug*(net.w_act[m,:].pow(-net.temp_grouper))).sum())).sum() +
    #         torch.matmul(net.w_act[m,:], torch.exp((-1/2)*torch.matmul(torch.matmul(torch.tensor(net.microbe_locs)[m,:] - net.mu_bug.T,
    #                                    torch.eye(net.microbe_locs.shape[1])*(1/net.r_bug)),
    #                                    torch.tensor(net.microbe_locs)[m,:] -
    #                                    net.mu_bug)/torch.sqrt(((2*np.pi)**net.microbe_locs.shape[1]
    #                                                            )*torch.prod(net.r_bug))).type(torch.FloatTensor))
    #         for m in np.arange(net.microbe_locs.shape[0])]).sum(0)

    temp_dist = MultivariateNormal(net.mu_bug, (torch.eye(net.microbe_locs.shape[1])*torch.exp(net.r_bug)).float())
    loss_dict['w'] = torch.stack([(-torch.log(net.pi_bug) + (net.temp_grouper + 1)*torch.log(net.w_act[m,:]) +
            torch.log((net.pi_bug*(net.w_act[m,:].pow(-net.temp_grouper))).sum())).sum() -
            torch.matmul(net.w_act[m,:],
                         torch.exp(temp_dist.log_prob(torch.FloatTensor(net.microbe_locs[m,:]))))
            for m in np.arange(net.microbe_locs.shape[0])]).sum(0)
    # loss_dict['mu_bug'] = torch.matmul(torch.matmul(net.mu_bug.T, torch.eye(net.microbe_locs.shape[1])*net.mu_var_bug),
    #                                    net.mu_bug).sum()
    temp_dist = MultivariateNormal(torch.zeros(net.mu_bug.shape),torch.eye(net.microbe_locs.shape[1])*net.mu_var_bug)
    loss_dict['mu_bug'] = -temp_dist.log_prob(net.mu_bug).sum()
    gamma = Gamma(net.r_scale_bug / 2, torch.tensor((net.r_scale_bug * (net.r0_bug))) / 2)
    loss_dict['r_bug'] = (-gamma.log_prob(1/torch.exp(net.r_bug))).sum()
    # loss_dict['r_bug'] = (-(net.r_scale_bug*net.r0_bug)/(2*net.r_bug) -
    #              (1 + net.r_scale_bug/2)*torch.log(net.r_bug)).sum()
    loss_dict['pi_bug'] = (torch.Tensor(1-np.array(net.e_bug))*torch.log(net.pi_bug)).sum()
    #
    # loss_dict['z'] = torch.stack([(-torch.log(net.pi_met) + (net.temp_grouper + 1)*torch.log(net.z_act[m,:]) +
    #         torch.log((net.pi_met*(net.z_act[m,:].pow(-net.temp_grouper))).sum())).sum() +
    #         torch.matmul(net.z_act[m,:], torch.exp((-1/2)*torch.matmul(torch.matmul(torch.tensor(net.met_locs)[m,:] - net.mu_met.T,
    #                                    torch.eye(net.met_locs.shape[1])*(1/net.r_met)),
    #                                    torch.tensor(net.met_locs)[m,:] -
    #                                    net.mu_met)/torch.sqrt(((2*np.pi)**net.met_locs.shape[1])*torch.prod(net.r_met)
    #                                                           )).type(torch.FloatTensor))
    #         for m in np.arange(net.met_locs.shape[0])]).sum(0)

    temp_dist = MultivariateNormal(net.mu_met, (torch.eye(net.met_locs.shape[1])*torch.exp(net.r_met)).float())
    loss_dict['z'] = torch.stack([(-torch.log(net.pi_met) + (net.temp_grouper + 1)*torch.log(net.z_act[m,:]) +
            torch.log((net.pi_met*(net.z_act[m,:].pow(-net.temp_grouper))).sum())).sum() -
            torch.matmul(net.z_act[m,:],
                         torch.exp(temp_dist.log_prob(torch.FloatTensor(net.met_locs[m,:]))))
            for m in np.arange(net.met_locs.shape[0])]).sum(0)

    loss_dict['mu_met'] = torch.matmul(torch.matmul(net.mu_met.T, torch.eye(net.met_locs.shape[1])*net.mu_var_met),
                                       net.mu_met).sum()
    gamma = Gamma(net.r_scale_met / 2, torch.tensor((net.r_scale_met * (net.r0_met))) / 2)
    loss_dict['r_met'] = (-gamma.log_prob(1/torch.exp(net.r_met))).sum()
    # loss_dict['r_met'] = ((net.r_scale_met*net.r0_bug)/(2*net.r_met) +
    #              (1 + net.r_scale_met/2)*torch.log(net.r_met)).sum()
    loss_dict['pi_met'] = (torch.Tensor(1-np.array(net.e_bug))*torch.log(net.pi_met)).sum()
    # loss_dict['y'] = y_loss.copy()
    y_loss = 0
    for param in compute_loss_for:
        y_loss += loss_dict[param]
        if torch.isnan(loss_dict[param]) or torch.isinf(loss_dict[param]):
            print('debug')
    return y_loss, loss_dict



def filter_by_pt(dataset, targets=None, perc = .15, pt_thresh = 1, meas_thresh = 10, weeks = [0,1,2]):
    # tmpts = [float(x.split('-')[1]) for x in dataset.index.values if x.replace('.','').isnumeric()]
    # mets is dataset with ones where data is present, zeros where it is not
    mets = np.zeros(dataset.shape)
    mets[np.abs(dataset) > meas_thresh] = 1


    if weeks is not None:
        df_drop = [x for x in dataset.index.values if not x.split('-')[1].replace('.', '').isnumeric()]
        dataset = dataset.drop(df_drop)
        mets = np.zeros(dataset.shape)
        mets[np.abs(dataset) > meas_thresh] = 1
        pts = [x.split('-')[0] for x in dataset.index.values]
        ixs = dataset.index.values
        ix_add = [i for i in range(len(ixs)) if float(ixs[i].split('-')[1]) in weeks]
        oh = np.zeros(len(pts))
        oh[np.array(ix_add)] = 1
        index = pd.MultiIndex.from_tuples(list(zip(*[pts, oh.tolist()])), names = ['pts', 'add'])
        df = pd.DataFrame(mets, index = index)
        df2 = df.xs(1, level = 'add')
        df2 = df2.groupby(level=0).sum()
        mets = np.zeros(df2.shape)
        mets[np.abs(df2)>0] = 1

    # if measurement of a microbe/metabolite only exists in less than pt_thresh timepoints, set that measurement to zero
    if pt_thresh > 1:
        pts = [x.split('-')[0] for x in dataset.index.values]
        for pt in pts:
            ixs = np.where(np.array(pts) == pt)[0]
            mets_pt = mets[ixs,:]
            # tmpts_pt = np.array(tmpts)[ixs]
            mets_counts = np.sum(mets_pt, 0).astype('int')
            met_rm_ixs = np.where(mets_counts < pt_thresh)[0]
            for ix in ixs:
                mets[ix, met_rm_ixs] = 0

    mets_all_keep = []
    # For each class, count how many measurements exist within that class and keep only measurements in X perc in each class
    if targets is not None:
        if sum(['-' in x for x in targets.index.values]) > 1:
            labels = targets[dataset.index.values]
        else:
            labels = targets[np.array(pts)]
        for lab_cat in np.unique(labels):
            mets_1 = mets[np.where(labels == lab_cat)[0], :]
            met_counts = np.sum(mets_1, 0)
            met_keep_ixs = np.where(met_counts >= np.round(perc * mets_1.shape[0]))[0]
            mets_all_keep.extend(met_keep_ixs)
    else:
        met_counts = np.sum(mets, 0)
        mets_all_keep = np.where(met_counts >= np.round(perc * mets.shape[0]))[0]
    return dataset.iloc[:,np.unique(mets_all_keep)]
# points = [(0,1),(2,4),(3,2),(5,8),(0,2),(4,0),(3,7),(8,4)]
# plt.scatter(np.array(points)[:,0], np.array(points)[:,1])
# plt.show()
# hull = jarvis(points)