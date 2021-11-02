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


# points = [(0,1),(2,4),(3,2),(5,8),(0,2),(4,0),(3,7),(8,4)]
# plt.scatter(np.array(points)[:,0], np.array(points)[:,1])
# plt.show()
# hull = jarvis(points)