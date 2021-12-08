import subprocess
from main_MAP import *
max_load = 20

my_str = "python3 ./main_MAP.py -learn {0} -priors {1} -case {2} -N_met {3} -N_bug {4} -N_nuisance {5}"

N_met = 20
N_bug = 22
learn = 'all'
pid_list = []
for case in [1, 2]:
    for priors in ['all','none']:
        if case == 1:
            n_nuisance = 0
            cmd = my_str.format(learn, priors, case, N_met, N_bug, n_nuisance)
            args2 = cmd.split(' ')
            pid = subprocess.Popen(args2)
            pid_list.append(pid)
            while sum([x.poll() is None for x in pid_list]) >= max_load:
                time.sleep(30)
        else:
            for n_nuisance in [2,4,8,16,18]:
                cmd = my_str.format(learn, priors, case, N_met, N_bug, n_nuisance)
                args2 = cmd.split(' ')
                pid = subprocess.Popen(args2)
                pid_list.append(pid)
                while sum([x.poll() is None for x in pid_list]) >= max_load:
                    time.sleep(30)