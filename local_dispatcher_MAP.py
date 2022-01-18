import subprocess
from main_MAPv0 import *
max_load = 5

my_str = "python3 ./main_MAP.py -N_met {0} -N_bug {1} -n_local_clusters {2} -L {3} " \
         "" \
         "-K {4} -meas_var {5} -prior_meas_var {6} -seed {7} -learn {8} -priors {9}"

N_met = 10
N_bug = 10
learn = ['r_bug','r_met', 'z']
priors = ['r_bug','r_met', 'z']
pid_list = []
prior_meas_var = 1e6
meas_var = 0.01
local = 1
for seed in np.arange(5):
    for learn, priors in [(['r_bug', 'r_met','z'],['r_bug', 'r_met','z']), (['all'],['all'])]:
        for L, K in [(2,2)]:
            # for local in [1, 5]:
            # for priors in ['all', 'none']:
            cmd = my_str.format(N_met, N_bug, local, L, K, meas_var, prior_meas_var, seed, ' '.join(learn), ' '.join(priors))
            print(cmd)
            args2 = cmd.split(' ')
            pid = subprocess.Popen(args2)
            pid_list.append(pid)
            time.sleep(0.5)
            while sum([x.poll() is None for x in pid_list]) >= max_load:
                time.sleep(30)
