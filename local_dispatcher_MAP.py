import subprocess
from main_MAPv0 import *
max_load = 5

parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case", help="case", type=str)
args = parser.parse_args()
# prior_meas_var = 1e6
# meas_var = 0.01
# local = 1
# learn_num_clusters = 1
# learn_list = ['beta','alpha',('r_met', 'mu_met', 'z'),('r_bug','mu_bug'),('pi_met','z')]
learn_list = [('beta','alpha','r_met','mu_met','z','pi_met','e_met'),('beta','alpha','r_met','mu_met','z','pi_met','e_met')]
priors_list = [('beta','alpha','r_met','mu_met','z','pi_met','e_met'),('beta','alpha','r_met','mu_met','z','pi_met')]
param_dict = {'N_met': 10, 'N_bug': 10, ('L', 'K'): [(2,2),(3,3)], 'seed': [0], 'hyper_mu': [0,1], 'hyper_r': [0,1],
              ('learn', 'priors'): list(zip(*[learn_list, priors_list])), 'lr': [0.01], 'iter': 20001, 'load': 1}

total_iters = np.prod([len(v) for v in param_dict.values() if hasattr(v, '__len__')])

i = 0
list_keys = []
list_vals = []
for key, value in param_dict.items():
    list_keys.append(key)
    if hasattr(value, "__len__"):
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
        if hasattr(p[i], '__len__'):
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
    cmd = my_str
    print(cmd)
    args2 = cmd.split(' ')
    pid = subprocess.Popen(args2)
    pid_list.append(pid)
    time.sleep(0.5)
    while sum([x.poll() is None for x in pid_list]) >= max_load:
        time.sleep(1)


# for seed in np.arange(5):
#     for learn, priors in list(zip(*[learn_list, learn_list])):
#         for lr in np.logspace(-1,-5,5):
#             # for local in [1, 5]:
#             # for priors in ['all', 'none']:
#             cmd = my_str.format(N_met, N_bug, local, L, K, meas_var, prior_meas_var, seed,
#                                 ' '.join(learn), ' '.join(priors), learn_num_clusters)
#             print(cmd)
#             args2 = cmd.split(' ')
#             pid = subprocess.Popen(args2)
#             pid_list.append(pid)
#             time.sleep(0.5)
#             while sum([x.poll() is None for x in pid_list]) >= max_load:
#                 time.sleep(30)
