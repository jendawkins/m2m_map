import subprocess
from main_MAPv0 import *
my_str_orig = '''
#!/bin/bash
#BSUB -J m2m
#BSUB -o case{9}_L{4}_K{5}.out
#BSUB -e case{9}_L{4}_K{5}.err

# This is a sample script with specific resource requirements for the
# **bigmemory** queue with 64GB memory requirement and memory
# limit settings, which are both needed for reservations of
# more than 40GB.
# Copy this script and then submit job as follows:
# ---
# cd ~/lsf
# cp templates/bsub/example_8CPU_bigmulti_64GB.lsf .
# bsub < example_bigmulti_8CPU_64GB.lsf
# ---
# Then look in the ~/lsf/output folder for the script log
# that matches the job ID number

# Please make a copy of this script for your own modifications

#BSUB -q gpu

# Some important variables to check (Can be removed later)
echo '---PROCESS RESOURCE LIMITS---'
ulimit -a
echo '---SHARED LIBRARY PATH---'
echo $LD_LIBRARY_PATH
echo '---APPLICATION SEARCH PATH:---'
echo $PATH
echo '---LSF Parameters:---'
printenv | grep '^LSF'
echo '---LSB Parameters:---'
printenv | grep '^LSB'
echo '---LOADED MODULES:---'
module list
echo '---SHELL:---'
echo $SHELL
echo '---HOSTNAME:---'
hostname
echo '---GROUP MEMBERSHIP (files are created in the first group listed):---'
groups
echo '---DEFAULT FILE PERMISSIONS (UMASK):---'
umask
echo '---CURRENT WORKING DIRECTORY:---'
pwd
echo '---DISK SPACE QUOTA---'
df .
echo '---TEMPORARY SCRATCH FOLDER ($TMPDIR):---'
echo $TMPDIR

# Add your job command here
# source activate dispatcher
# module load Anaconda3/5.2.0
cd /PHShome/jjd65/m2m_map
'''
# python3 ./main_MAP.py -learn {0} -priors {1} -N_met {2} -N_bug {3} -L {4} -K {5} -meas_var {6} -seed {7} -load {8} -case {9} -iterations {10}

parser = argparse.ArgumentParser()
parser.add_argument("-load", "--load", help="load or not", type=int)
parser.add_argument("-case", "--case", help="case", type=str)
parser.add_argument("-iter", "--iter", help="iter", type=int)
args = parser.parse_args()

# prior_meas_var = 1e6
# meas_var = 0.01
# local = 1
# learn_num_clusters = 1
# learn_list = ['beta','alpha',('r_met', 'mu_met', 'z'),('r_bug','mu_bug'),('pi_met','z')]
learn_list = [('beta','alpha','r_met','mu_met','z','pi_met','e_met'),('beta','alpha','r_met','mu_met','z','pi_met','e_met')]
priors_list = [('beta','alpha','r_met','mu_met','z','pi_met','e_met'),('beta','alpha','r_met','mu_met','z','pi_met')]
param_dict1 = {'N_met': 20, 'N_bug': 20, ('L', 'K'): [(2,2),(3,3)], 'seed': [0,1,2], 'hyper_mu': [0,1], 'hyper_r': [0,1],
              ('learn', 'priors'): list(zip(*[learn_list, priors_list])), 'lr': 0.01, 'iter': 20001, 'load': 1, 'lm': 1,
               'linear': 1}

learn_list = ['', 'all']
priors_list = ['', 'all']
param_dict2 = {'N_met': 20, 'N_bug': 20, ('L', 'K'): [(2,2)], 'seed': [0,1,2], 'linear': 0,
              ('learn', 'priors'): list(zip(*[learn_list, priors_list])), 'lr': 0.01, 'iter': 20001, 'load': 1, 'lm': 0,
               'nltype': ['linear', 'sin', 'exp']}

for param_dict in [param_dict1, param_dict2]:
    total_iters = np.prod([len(v) for v in param_dict.values() if hasattr(v, '__len__')])

    i = 0
    list_keys = []
    list_vals = []
    for key, value in param_dict.items():
        list_keys.append(key)
        if hasattr(value, "__len__") and not isinstance(value, str):
            list_vals.append(value)
        else:
            list_vals.append([value])

    zipped_params = list(itertools.product(*list_vals))

    pid_list = []
    for p in zipped_params:
        fin_list = []
        i = 0
        my_str = "python3 ./main_MAP.py"
        for l in list_keys:
            if hasattr(p[i], '__len__') and not isinstance(p[i], str):
                for ii in range(len(p[i])):
                    if isinstance(p[i][ii], tuple):
                        fin = ' '.join(p[i][ii])
                    else:
                        fin = p[i][ii]
                    my_str = my_str + ' -' + l[ii] + ' ' + str(fin)
            else:
                my_str = my_str + ' -' + l + ' ' + str(p[i])
            i += 1
        if args.case:
            my_str = my_str + ' -case ' + args.case
        if args.load:
            my_str = my_str + '-load ' + args.load
        if args.iter:
            my_str = my_str + '-iter ' + args.iter
        cmd = my_str
        # print(cmd)
        # print(my_str_orig + my_str)
        f = open('m2m.lsf', 'w')
        f.write(my_str_orig + my_str)
        f.close()
        os.system('bsub < {}'.format('m2m.lsf'))


# parser = argparse.ArgumentParser()
# parser.add_argument("-load", "--load", help="load or not", type=int)
# parser.add_argument("-case", "--case", help="case", type=str)
# parser.add_argument("-iter", "--iter", help="iter", type=int)
# args = parser.parse_args()
#
# N_met = 20
# N_bug = 20
# learn = 'all'
# pid_list = []
# meas_var = 0.01
# priors = 'all'
# for seed in range(10):
#     # for meas_var in [0.01, 1]:
#     for L, K in [(3,3),(4,4)]:
#         # for repeat_clusters in [0, 1]:
#         for N_met, N_bug in [(25,25)]:
#             f = open('m2m.lsf', 'w')
#             f.write(my_str.format(learn, priors, N_met, N_bug, L, K, meas_var, seed, args.load, args.case, args.iter))
#             f.close()
#             os.system('bsub < {}'.format('m2m.lsf'))