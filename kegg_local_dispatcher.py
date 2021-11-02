import numpy as np
import os
import argparse
import subprocess
import re
from dataLoader import *
dl = dataLoader()
import time
import warnings
warnings.filterwarnings("ignore")

my_str = '''python3 ./run_kegg.py -info {0} -path {1}'''

max_load = 50

path_cmpds = 'kegg/compounds/'
if not os.path.isdir('kegg'):
    os.mkdir('kegg')

edges = []
edge_dict = {}
cmpd_dict = {}
ix = 0
feature_data = dl.cdiff_data_dict['featureMetadata']
feature_data = feature_data.set_index('BIOCHEMICAL')
if not os.path.isdir(path_cmpds):
    os.mkdir(path_cmpds)
pid_list = []
cnt = 0
print('# metabolites: ' + str(len(feature_data.index.values)))
for metabolite in feature_data.index.values:
    cnt += 1
    if isinstance(feature_data.loc[metabolite]['KEGG'], str):
        kegg_num = feature_data.loc[metabolite]['KEGG']
        if os.path.isfile(path_cmpds + '/' + kegg_num + '.pkl'):
            continue
        cmd = my_str.format(kegg_num, path_cmpds)
        args2 = cmd.split(' ')
        pid = subprocess.Popen(args2)
        pid_list.append(pid)
        while sum([x.poll() is None for x in pid_list]) >= max_load:
            time.sleep(30)
    if cnt%100==0:
        print(cnt)

for file in os.listdir(path_cmpds):
    kegg_num = file.split('.pkl')[0]
    with open(path_cmpds + file, 'rb') as f:
        doc = pkl.load(f)
    if 'PATHWAY' in doc:
        pattern = re.compile(r'map ?[0-9][^\s]+')
        pathways = pattern.findall(doc)
        cmpd_dict[kegg_num] = pathways

path_pathways = 'kegg/pathways/'
if not os.path.isdir(path_pathways):
    os.mkdir(path_pathways)
unique_pathways = np.unique(np.concatenate(list(cmpd_dict.values())))
print('# Pathways: ' + str(len(unique_pathways)))

pid_list = []
path_pathways = 'kegg/pathways/'
if not os.path.isdir(path_pathways):
    os.mkdir(path_pathways)
path_modules = 'kegg/modules/'
if not os.path.isdir(path_modules):
    os.mkdir(path_modules)
edge_dict = {}
for pathway in unique_pathways:
    ix += 1
    if not os.path.isfile(path_pathways + pathway + '.pkl'):
        cmd = my_str.format(pathway, path_pathways)
        args2 = cmd.split(' ')
        pid = subprocess.Popen(args2)
        pid_list.append(pid)
        while sum([x.poll() is None for x in pid_list]) >= max_load:
            time.sleep(30)

    ii = 0
    while not os.path.isfile(path_pathways + pathway + '.pkl'):
        time.sleep(0.1)
        ii += 1
        if ii > 20:
            continue
    with open(path_pathways + pathway + '.pkl', 'rb') as f:
        doc = pkl.load(f)

    edge_dict[pathway] = []

    pattern = re.compile(r'[\s] M ?[0-9][^\s]+')
    modules = pattern.findall(doc)
    for module in modules:
        module = module.replace(' ', '')

        if not os.path.isfile(path_modules + module + '.pkl'):
            cmd = my_str.format(module, path_modules)
            args2 = cmd.split(' ')
            pid = subprocess.Popen(args2)
            pid_list.append(pid)
            while sum([x.poll() is None for x in pid_list]) >= max_load:
                time.sleep(30)
        ii = 0
        while not os.path.isfile(path_modules + module + '.pkl'):
            time.sleep(0.1)
            ii += 1
            if ii>20:
                continue
        with open(path_modules + module + '.pkl', 'rb') as f:
            doc2 = pkl.load(f)

        pattern2 = re.compile(r'(C[0-9]+ \+ )*(C[0-9]+ \-\> C[0-9]+)+( \+ C[0-9]+)*')
        rxns = pattern2.findall(doc2)
        for rxn in rxns:
            rx = ''.join(rxn)
            ba = rx.split(' -> ')
            before = ba[0].split(' + ')
            after = ba[1].split(' + ')
            for b in before:
                for a in after:
                    edges.append((b, a))
                    edge_dict[pathway].append((b, a))

    if ix % 100 == 0:
        print(ix)

with open('kegg/edge_dict.pkl', 'wb') as f:
    pkl.dump(edge_dict, f)

