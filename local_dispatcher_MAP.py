import subprocess
from main_MAPv0 import *
max_load = 10

parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case", help="case", type=str)
args = parser.parse_args()
# prior_meas_var = 1e6
# meas_var = 0.01
# local = 1
# learn_num_clusters = 1
# learn_list = [('beta','alpha','r_met','mu_met','z','pi_met','e_met')
# learn_list = [('beta','alpha','r_met','mu_met','z','pi_met','sigma','e_met'),('beta','alpha','r_met','mu_met','z','pi_met','sigma','e_met'),
#               ('beta','alpha','r_met','mu_met','z','pi_met','sigma','e_met'),('beta','alpha','r_met','mu_met','z','pi_met','sigma','e_met')]
# priors_list = [('beta','alpha','r_met','mu_met','z','pi_met','sigma'), ('beta','alpha','r_met','mu_met','z','pi_met','sigma','e_met'),
#                ('beta','alpha','r_met','mu_met','z','pi_met','sigma','e_met'),('beta','alpha','r_met','mu_met','z','pi_met','sigma','e_met')]
# #     # [('beta','alpha','r_met','mu_met','z','pi_met')]
# #
# # hyper_mu = [0,0]
# # hyper_r = [0,0]
# # param_dict = {'N_met': 20, 'N_bug': 20, ('L', 'K'): [(3,3),(4,4),(5,5)], 'seed': [2],
# #               ('learn', 'priors','hyper_mu','hyper_r'): list(zip(*[learn_list, priors_list, hyper_mu, hyper_r])),
# #               'lr': [0.001], 'iter': 22001, 'load': 1, 'learn_mvar': 0,
# #               'mz': 1, 'lm': 1, 'linear': 1, 'meas_var': [0.01], 'prior_meas_var': [0.01],
# #               'case': 'learn_met_clusters-learn_mvar', 'dist_var_perc': [0.5],
# #                'adjust_mvar': 0, 'adjust_lr': 1, 'schedule_lr': 1}
#
# # learn_list = [('beta','alpha','r_met','mu_met','z','pi_met')]
# # priors_list = [('beta','alpha','r_met','mu_met','z','pi_met')]
# param_dict = {'N_met': 20, 'N_bug': 20, ('L', 'K'): [(4,4)], 'seed': [2], 'hyper_mu': 0, 'hyper_r': 0,
#               ('learn', 'priors'): [('alpha','alpha')], 'lr': [0.1,0.01,0.001], 'iter': 22001, 'load': 0, 'lb': 0,
#               'mz': 1, 'lm': 0, 'case': 'learn_non-linear', 'linear': 0, 'nltype': ['poly'],
#               'meas_var': 0.1, 'learn_mvar': 1, 'adjust_mvar':0}
#
# # param_dict = {'N_met': 20, 'N_bug': 20, ('L', 'K'): [(2,2)], 'seed': [0,1,2,3,4,5,6,7,8,9,10], ('learn','priors'):
# #             [('all','all')], 'lr': [0.01], 'l1': [0,1], 'lm': [0,1], 'p': [1,2,4], 'schedule_lr': 1,
# #               'load': 0, ('linear', 'nltype'): [(0,'linear'),(0,'sin'),(0,'exp'), (0,'sigmoid'), (1,'linear')],
# #               'prior_meas_var': 1, 'meas_var': 0.001, 'adjust_mvar': 0, 'adjust_lr': 1}
#
# #
# param_dict = {'N_met':20, 'N_bug': 20, ('L', 'K'): [(2,2),(3,3),(4,4)], 'seed': [9],
#               ('learn','priors'): [('all','all'),('','')],'iter': 22001, 'learn_mvar': 1,
#               'lr': [0.001], 'adjust_mvar': 0, 'prior_meas_var': 0.1,'meas_var': 0.1, 'dist_var_perc': 0.5, 'schedule_lr': 1,
#               'load': 0, ('linear', 'nltype'): [(0,'linear'),(0,'sin'),(0,'exp'), (0,'sigmoid'), (1,'linear')],
#               'mz': 1, 'lm': [0,1], 'lb': [0,1], 'adjust_lr': 1, 'sample_mu':0}
#
# param_dict = {'N_met':20, 'N_bug': 20, ('L', 'K'): [(2,2),(3,3)], 'seed': [2,5],
#               ('learn', 'priors'): [(('mu_bug','r_bug','beta','alpha'),('mu_bug','r_bug','beta','alpha'))],'iter': 22001, 'learn_mvar': 1,
#               'lr': [0.001], 'adjust_mvar': 0, 'prior_meas_var': 0.1,'meas_var': 0.1, 'dist_var_perc': [2.0], 'schedule_lr': 1,
#               'load': 0, 'linear': 1, 'lw': 0, 'rep_clust': 0,
#               'mz': 1, 'lm': [0], 'lb': [1], 'adjust_lr': 1, 'sample_mu':0}
#
# # learn_list = [('mu_bug','r_bug','sigma'),('mu_bug','r_bug','sigma','w'),
# #               ('mu_bug','r_bug','sigma','w'),('mu_bug','r_bug','sigma','w'),
# #               ('mu_bug','r_bug','sigma','w')]
# # priors_list = [('mu_bug','r_bug','sigma'),('mu_bug','r_bug','sigma','w'),
# #               ('mu_bug','r_bug','sigma','w'),('mu_bug','r_bug','sigma','w'),
# #                ('mu_bug','r_bug','sigma','w')]
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
#
# learn_list = [('mu_bug','r_bug','sigma','w'),
#               ('mu_bug','r_bug','alpha','beta','sigma','w')]
# priors_list = [('mu_bug','r_bug','sigma','w'),
#               ('mu_bug','r_bug','alpha','beta','sigma','w')]
# lb = [0,1]
#
# param_dict = {'N_met':20, 'N_bug': 20, ('L', 'K'): [(3,3),(4,4)], 'seed': [0,1,2,3], 'learn': 'all', 'priors': 'all',
#               'iter': 22001, 'learn_mvar': 1, 'lw': 3, 'hard': 0,'p': 0.001,
#               'lr': [0.001], 'adjust_mvar': 0, 'prior_meas_var': 0.1,'meas_var': 0.1, 'dist_var_perc': [0.5],
#               'schedule_lr': 1,'case':'LEARN_ALL',
#               'load': 0, 'linear': 1, 'rep_clust': 0,
#               # 'case': '3-18-lw3-rep_clust',
#               'mz': 1, 'lm': [0,1], 'lb': [0,1], 'adjust_lr': 1, 'sample_mu':0}

param_dict = {'N_met':20, 'N_bug': 20, ('L', 'K'): [(3,3),(4,4)], 'seed': [0,1,2,3,4,5,6,7,8,9],
              'learn': [('mu_bug', 'r_bug', 'sigma', 'alpha', 'beta', 'p')],
              'priors': [('mu_bug', 'r_bug', 'sigma', 'alpha', 'beta', 'p')],
              'iter': 20500, 'learn_mvar': 1, 'lw': [0,], 'hard': 1,'p': 0.001,
              'lr': [0.001], 'adjust_mvar': 0, 'prior_meas_var': 0.1,'meas_var': 0.1, 'dist_var_perc': [0.5],
              'schedule_lr': [1], 'w_tau': [(-0.001,-1)],'w_tau2': [(-0.001,-1)],
              'load': 0, 'linear': 1, 'rep_clust': 1,
              'case': '3-29_OrigMethod_NumClust',
              'mz': 1, 'lm': 0, 'lb': 1, 'adjust_lr': 1, 'sample_mu':0}

param_dict = {'N_met':20, 'N_bug': 20, ('L', 'K'): [(3,3),(4,4),(6,6)], 'seed': [0,1,2,3,4,5,6,7,8,9],
              'learn': [('mu_bug', 'r_bug')],
              'priors': [('mu_bug', 'r_bug')], 'dim': [2],
              'iter': 21000, 'learn_mvar': 1, 'lw': [0], 'hard': 1,'p': 0.001,
              'lr': [0.01,0.005,0.001], 'adjust_mvar': 0, 'meas_var': 0.1, 'dist_var_perc': [1],
              'schedule_lr': [0], 'w_tau': [(-0.001,-1)],'w_tau2': [(-0.001,-1)],
              'load': 0, 'linear': 1, 'rep_clust': [0,1],
              'case': '4-1-test_lr',
              'mz': 1, 'lm': 0, 'lb': 0, 'adjust_lr': 1, 'sample_mu':0}


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
