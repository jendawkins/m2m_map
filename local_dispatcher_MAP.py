import subprocess
from main_MAPv0 import *
max_load = 10

my_str = "python3 ./main_MAP.py -learn {0} -priors {1} -N_met {2} -N_bug {3} -n_local_clusters {4} -L {5} -K {6} -meas_var {7} -prior_meas_var {8} -seed {9}"

N_met = 10
N_bug = 10
learn = 'all'
pid_list = []
prior_meas_var = 4
meas_var = 0.001
for seed in [0,1]:
    for L, K in [(2,2),(3,3),(4,4)]:
        for local in [1, 5]:
            for priors in ['all', 'none']:
                cmd = my_str.format(learn, priors, N_met, N_bug, local, L, K, meas_var, prior_meas_var, seed)
                args2 = cmd.split(' ')
                pid = subprocess.Popen(args2)
                pid_list.append(pid)
                while sum([x.poll() is None for x in pid_list]) >= max_load:
                    time.sleep(30)
