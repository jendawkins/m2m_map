# use environment M2M_CodeBase!!!
import cirpy
import pubchempy as pcp
import subprocess
import re

from dataLoader import *
dl = dataLoader()

feature_data = dl.cdiff_data_dict['featureMetadata']
feature_data = feature_data.set_index('BIOCHEMICAL')
pubchem_nums = feature_data.PUBCHEM.dropna().values
metabolites = feature_data.index.values

my_str = '''python3 ./run_pubchem.py -path {0} -key {1} -val {2}'''
path = 'pubchem_isomeric'
max_load = 50
smiles_dict = {}
fl = 0
pid_list = []
print('# w/ pubchem: ' + str(len(pubchem_nums)))
cnt = 0
for ic, pubchem_num in enumerate(pubchem_nums):
    if isinstance(pubchem_num, int) or isinstance(pubchem_num, str):
        if isinstance(pubchem_num, str) and ';' in pubchem_num:
            nums = pubchem_num.split(';')
        else:
            nums = [pubchem_num]
        for num in nums:
            if os.path.isfile(path + '/cid/' + str(num) + '.pkl'):
                continue
            cmd = my_str.format(path, 'cid', int(num))
            args2 = cmd.split(' ')
            pid = subprocess.Popen(args2)
            pid_list.append(pid)
            cnt += 1
        while sum([x.poll() is None for x in pid_list]) >= max_load:
            time.sleep(30)

        if cnt % 100 == 0:
            print(cnt)


met_dict = {}
for ic, metabolite in enumerate(metabolites):
    orig_met = metabolite
    metabolite = metabolite.replace('*','')
    if ' or ' in metabolite:
        if '(' in metabolite:
            metabolite = metabolite.split('(')[1].split(')')
            if re.search('[a-zA-Z]', metabolite[1]):
                metabolite_1 = ''.join([metabolite[0].split('or')[0],metabolite[1]]).replace(' ','')
                metabolite_2 = ''.join([metabolite[0].split('or')[1], metabolite[1]]).replace(' ','')
                for met in [metabolite_1, metabolite_2]:
                    met_dict[met] = orig_met
                    if os.path.isfile(path + '/' + 'name/' + met +'.pkl'):
                        continue
                    cmd = my_str.format(path, 'name', met)
                    args2 = cmd.split(' ')
                    pid = subprocess.Popen(args2)
                    pid_list.append(pid)
                    while sum([x.poll() is None for x in pid_list]) >= max_load:
                        time.sleep(30)

            else:
                continue
        else:
            metabolite = metabolite.split('[')[0].replace(' ','')
    else:
        if os.path.isfile(path + '/' + 'name/' + metabolite + '.pkl'):
            continue
        met_dict[metabolite] = orig_met
        cmd = my_str.format(path, 'name', metabolite)
        args2 = cmd.split(' ')
        pid = subprocess.Popen(args2)
        pid_list.append(pid)
        while sum([x.poll() is None for x in pid_list]) >= max_load:
            time.sleep(30)

with open('pubchem/met_dict.pkl', 'wb') as f:
    pkl.dump(met_dict, f)