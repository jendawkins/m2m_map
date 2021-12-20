import subprocess
from main_MAP import *
max_load = 10

my_str = "python3 ./main_MAP.py -learn {0} -priors {1} -case {2} -N_met {3} -N_bug {4} -N_nuisance {5} -L {6} -K {7} -meas_var {8} -prior_meas_var {9} -seed {10}"

N_met = 10
N_bug = 10
learn = 'all'
pid_list = []
case = '1_diff_tau_scheduler'
prior_meas_var = 4
meas_var = 0.001
for seed in [0,1]:
    for L, K in [(2,2),(3,3),(4,4)]:
        for priors in ['all']:
            if case != 2:
                n_nuisance = 0
                cmd = my_str.format(learn, priors, case, N_met, N_bug, n_nuisance, L, K, meas_var, prior_meas_var, seed)
                args2 = cmd.split(' ')
                pid = subprocess.Popen(args2)
                pid_list.append(pid)
                while sum([x.poll() is None for x in pid_list]) >= max_load:
                    time.sleep(30)
            else:
                for n_nuisance in [2,4,8,16,18]:
                    cmd = my_str.format(learn, priors, case, N_met, N_bug, n_nuisance, L, K, meas_var, prior_meas_var, seed)
                    args2 = cmd.split(' ')
                    pid = subprocess.Popen(args2)
                    pid_list.append(pid)
                    while sum([x.poll() is None for x in pid_list]) >= max_load:
                        time.sleep(30)