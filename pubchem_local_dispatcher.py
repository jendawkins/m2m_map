import numpy as np
import os
import argparse
import subprocess
import re
from dataLoader import *
import time
import warnings
import pubchempy as pcp

warnings.filterwarnings("ignore")

my_str = '''python3 ./run_pubchem.py -name {0} -path {1}'''

max_load = 50
curr = '/Users/jendawk/Dropbox (MIT)/Microbes to Metabolomes/'
dont_open = ['gut16s.csv', 'bloodMetabolites.csv','bloodMetabolites_exerciseSubStudy.csv']
dat = {}
for folder in os.listdir(curr + '/Datasets'):
    if '.DS' in folder:
        continue
    for file in os.listdir(curr + '/Datasets/' + folder):
        if '.csv' in file and file not in dont_open:
            with open(curr + '/Datasets/' + folder + '/' + file, 'rb') as f:
                if folder == 'taylor et al':
                    if folder not in dat.keys():
                        dat[folder] = {}
                    dat[folder][file.split('_')[0]] = pd.read_csv(f)
                else:
                    dat[folder] = pd.read_csv(f)


pid_list = []
res_path = 'pubchem/yang_metabolites/'
if not os.path.isdir(res_path):
    os.mkdir(res_path)
yang_metabolites = dat['yang et al'].iloc[2:,0].values
met_dict = {}
for met in yang_metabolites:
    if not os.path.isfile(res_path + met + '.pkl'):
        cmd = my_str.format(met, res_path)
        args2 = cmd.split(' ')
        pid = subprocess.Popen(args2)
        pid_list.append(pid)
    while sum([x.poll() is None for x in pid_list]) >= max_load:
        time.sleep(2)

met_dict = {}
for met in yang_metabolites:
    if os.path.isfile(res_path + met + '.pkl'):
        with open(res_path + met + '.pkl', 'rb') as f:
            doc = pkl.load(f)
        met_dict[met] = doc

with open(res_path + 'full_dict.pkl', 'wb') as f:
    pkl.dump(met_dict, f)