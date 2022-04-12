import subprocess
from main_MAPv0 import *
max_load = 8

parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case", help="case", type=str)
args = parser.parse_args()
# #
# # learn_list = [('mu_bug','r_bug','alpha','beta','sigma'),('mu_bug','r_bug','alpha','beta','sigma','w'),
# #               ('mu_bug','r_bug','alpha','beta','sigma','w')]
# # priors_list = [('mu_bug','r_bug','alpha','beta','sigma'),('mu_bug','r_bug','alpha','beta','sigma','w'),
# #               ('mu_bug','r_bug','alpha','beta','sigma','w')]
# # lw = [0,1,2]
# # hard = [0,0,0]
# # param_dict = {'N_met':20, 'N_bug': 20, ('L', 'K'): [(3,3)], 'seed': [1,2,5],
# #               ('learn', 'priors', 'lw', 'hard'): list(zip(*[learn_list, priors_list, lw, hard])),
# #               'iter': 22001, 'learn_mvar': 1, 'dim': 2,
# #               'lr': [0.001], 'adjust_mvar': 0, 'prior_meas_var': 0.1,'meas_var': 0.1, 'dist_var_perc': [0.5],
# #               'schedule_lr': 1,
# #               'load': 0, 'linear': 1, 'rep_clust': 0, 'case': '3-16-lower_r',
# #               'mz': 1, 'lm': [0], 'lb': [0,1], 'adjust_lr': 1, 'sample_mu':0}


param_dict = {'N_met':20, 'N_bug': 20, ('L', 'K'): [(3,3)], 'seed': [0,1,2,3,4,5,6,7,8,9],
              # ('learn', 'priors'): [(('r_bug','mu_bug','sigma','beta','alpha'),
              #                        ('r_bug','mu_bug','sigma','beta','alpha'))],
              ('learn','priors'): [('all','all')],
              'dim': [2],
              'iter': 30000, 'learn_mvar': 1, 'lw': [0], 'hard': 1,'p': 0.001,
              'lr': [0.01,0.05], 'adjust_mvar': 0, 'meas_var': 0.1, 'dist_var_perc': [0.5,1],
              'schedule_lr': [0], 'w_tau': [(-0.01,-1)],'w_tau2': [(-0.01,-1)],
              'load': 0, 'linear': 1, 'rep_clust': [0],
              'case': '4-11_num_clusters',
              'mz': 1, 'lm': [0,1], 'lb': [1], 'adjust_lr': 1, 'sample_mu':0}


total_iters = np.prod([len(v) for v in param_dict.values() if hasattr(v, '__len__') and not isinstance(v, str)])
print(total_iters)

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
        if isinstance(l, tuple) and isinstance(p[i], tuple):
            for ii in range(len(l)):
                if hasattr(p[i][ii], "__len__") and not isinstance(p[i][ii], str):
                    pout = [str(pp) for pp in p[i][ii]]
                    my_str = my_str + ' -' + l[ii] + ' ' + ' '.join(pout)
                else:
                    fin = p[i][ii]
                    my_str = my_str + ' -' + l[ii] + ' ' + str(fin)
        elif not isinstance(l, tuple) and isinstance(p[i], tuple):
            pout = [str(pp) for pp in p[i]]
            my_str = my_str + ' -' + l + ' ' + ' '.join(pout)
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
