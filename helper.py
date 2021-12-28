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

def euc_dist(x,y):
    return np.sqrt(np.sum([(x[i] - y[i])**2 for i in range(len(x))]))

def pairwise_eval(guess, true):
    guess_dict = {i:guess[i] for i in range(len(guess))}
    true_dict = {i: true[i] for i in range(len(guess))}
    pairs = list(itertools.combinations(range(len(guess)),2))
    tp_fp = np.sum([math.comb(np.sum(guess==i),2) for i in np.unique(guess)])
    tp = len([i for i in range(len(pairs)) if guess_dict[pairs[i][0]]==guess_dict[pairs[i][1]] and
              true_dict[pairs[i][0]]==true_dict[pairs[i][1]]])
    fp = tp_fp - tp
    tn = len([i for i in range(len(pairs)) if guess_dict[pairs[i][0]]!=guess_dict[pairs[i][1]] and
              true_dict[pairs[i][0]]!=true_dict[pairs[i][1]]])
    tn_fn = math.comb(len(guess),2) - tp_fp
    fn = tn_fn - tn
    ri = (tp + tn)/(tp + fp + tn + fn)
    return tp, fp, tn, fn, ri

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