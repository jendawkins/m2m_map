import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.dirichlet import Dirichlet
import torch.optim as optim
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from helper import *
from sklearn.model_selection import KFold
from scipy.special import logit
from plot_helper import *
from MAP_loss import *
from concrete import *
import copy
import argparse


class Model(nn.Module):
    def __init__(self, met_locs, microbe_locs, temp_grouper = 10, temp_selector = 1, L = 2, K = 2, beta_var = 16.,
                 seed = 0, tau_transformer = 1):
        super(Model, self).__init__()
        self.L = L
        self.K = K
        self.beta_var = beta_var
        self.mu_var_met = (2/met_locs.shape[1])*np.sum(np.var(met_locs.T))
        self.mu_var_bug = (2/microbe_locs.shape[1])*np.sum(np.var(microbe_locs.T))
        if isinstance(temp_grouper, str):
            self.temp_grouper = 1
        else:
            self.temp_grouper = temp_grouper
        if isinstance(temp_selector, str):
            self.temp_selector = 1
        else:
            self.temp_selector = temp_selector
        self.met_locs = met_locs
        self.microbe_locs = microbe_locs
        self.embedding_dim = met_locs.shape[1]
        self.seed = seed
        self.alpha_loc = 1.
        self.tau_transformer = tau_transformer

        range_x = np.max(self.met_locs[:,0]) - np.min(self.met_locs[:,0])
        range_y = np.max(self.met_locs[:,1]) - np.min(self.met_locs[:,1])
        self.r_scale_met = np.sqrt(range_x**2 + range_y**2) / self.K

        range_x = np.max(self.microbe_locs[:,0]) - np.min(self.microbe_locs[:,0])
        range_y = np.max(self.microbe_locs[:,1]) - np.min(self.microbe_locs[:,1])
        self.r_scale_bug = np.sqrt(range_x**2 + range_y**2) / self.L

        self.params = {}
        self.distributions = {}
        self.params['beta'] = {'mean': 0, 'scale': np.sqrt(self.beta_var)}
        self.params['alpha'] = {'loc': self.alpha_loc, 'temp':self.temp_selector}
        self.params['mu_met'] = {'mean': 0, 'var': self.mu_var_met}
        self.params['mu_bug'] = {'mean': 0, 'var': self.mu_var_bug}
        self.params['r_bug'] = {'dof': 2, 'scale': self.r_scale_bug}
        self.params['r_met'] = {'dof': 2, 'scale': self.r_scale_met}
        self.distributions['beta'] = Normal(self.params['beta']['mean'], self.params['beta']['scale'])
        self.distributions['alpha'] = BinaryConcrete(self.params['alpha']['loc'], self.params['alpha']['temp'])
        self.distributions['mu_met'] = MultivariateNormal(torch.zeros(self.embedding_dim), self.params['mu_met']['var']*torch.eye(self.embedding_dim))
        self.distributions['mu_bug'] = MultivariateNormal(torch.zeros(self.embedding_dim), self.params['mu_bug']['var']*torch.eye(self.embedding_dim))
        self.distributions['r_bug'] = Gamma(self.params['r_bug']['dof'], self.params['r_bug']['scale'])
        self.distributions['r_met'] = Gamma(self.params['r_met']['dof'], self.params['r_met']['scale'])
        self.params['pi_met'] = {'epsilon': [1]*self.K}
        self.params['pi_bug'] = {'epsilon': [1]*self.L}
        self.distributions['pi_met'] = Dirichlet(torch.Tensor(self.params['pi_met']['epsilon']))
        self.distributions['pi_bug'] = Dirichlet(torch.Tensor(self.params['pi_bug']['epsilon']))
        self.range_dict = {}
        for param, dist in self.distributions.items():
            sampler = dist.sample([100])
            if len(sampler.shape)>1:
                sampler = sampler[:,0]
            range = sampler.max() - sampler.min()
            self.range_dict[param] = (sampler.min() - range * 0.1, sampler.max() + range * 0.1)
            if 'r_met' in param:
                self.range_dict[param] = (0, np.sum(np.sqrt(self.met_locs[:,0]**2 + self.met_locs[:,1]**2)))
            if 'r_bug' in param:
                self.range_dict[param] = (0, np.sum(np.sqrt(self.microbe_locs[:,0]**2 + self.microbe_locs[:,1]**2)))

        self.range_dict['w'] = (-0.1,1.1)
        self.range_dict['z'] = (-0.1,1.1)
        self.range_dict['alpha'] = (-0.1, 1.1)
        self.initialize(self.seed)


    def initialize(self, seed):
        torch.manual_seed(seed)
        self.initializations = {}
        self.initializations['beta'] = self.distributions['beta']
        self.initializations['alpha'] = Bernoulli(0.5)
        self.initializations['mu_met'] = self.distributions['mu_met']
        self.initializations['mu_bug'] = self.distributions['mu_bug']
        self.initializations['r_met'] = self.distributions['r_met']
        self.initializations['r_bug'] = self.distributions['r_bug']
        self.initializations['pi_bug'] = self.distributions['pi_bug']
        self.initializations['pi_met'] = self.distributions['pi_met']
        self.initializations['z'] = Normal(0,1)
        self.initializations['w'] = Normal(0,1)
        # beta_dist = Normal(0, np.sqrt(self.beta_var))
        self.beta = nn.Parameter(self.initializations['beta'].sample([self.L+1, self.K]), requires_grad=True)

        self.alpha = nn.Parameter(self.initializations['alpha'].sample([self.L, self.K]), requires_grad=True)
        self.alpha_act = torch.sigmoid(self.alpha / self.temp_selector)
        self.mu_met = nn.Parameter(self.initializations['mu_met'].sample(sample_shape = torch.Size([self.K])),
                                        requires_grad = True)
        self.mu_bug = nn.Parameter(self.initializations['mu_bug'].sample(sample_shape = torch.Size([self.L])),
                                        requires_grad = True)
        r_temp = self.initializations['r_met'].sample([self.K])
        self.r_met = nn.Parameter(torch.log(1/r_temp), requires_grad = True)

        r_temp = self.initializations['r_bug'].sample([self.L])
        self.r_bug = nn.Parameter(torch.log(1/r_temp), requires_grad=True)

        self.pi_bug = nn.Parameter(self.initializations['pi_bug'].sample(), requires_grad=True).unsqueeze(0)
        self.pi_met = nn.Parameter(self.initializations['pi_met'].sample(), requires_grad=True).unsqueeze(0)

        # cat_bug = Categorical(self.pi_bug).sample([self.microbe_locs.shape[0]])
        # temp = nn.functional.one_hot(cat_bug.squeeze(), num_classes = self.L).type(torch.FloatTensor)
        self.w = nn.Parameter(self.initializations['w'].sample([self.microbe_locs.shape[0], self.L]), requires_grad=True)
        self.w_act = torch.softmax(self.w,1)

        # cat_met = Categorical(self.pi_met).sample([self.met_locs.shape[0]])
        # temp = nn.functional.one_hot(cat_met.squeeze(), num_classes = self.K).type(torch.FloatTensor)
        # self.range_dict['z'] = (-0.1,1.1)
        self.z = nn.Parameter(self.initializations['z'].sample([self.met_locs.shape[0], self.K]), requires_grad=True)
        self.z_act = torch.softmax(self.z, 1)

    def forward(self, x):
        self.alpha_act = torch.sigmoid(self.alpha/self.tau_transformer)
        self.alpha_act = torch.clamp(self.alpha_act, min=1e-10, max=1-1e-7)
        self.w_act = torch.softmax(self.w / self.tau_transformer, 1)
        self.w_act = torch.clamp(self.w_act, min=1e-10, max=1-1e-7)
        g = torch.matmul(torch.Tensor(x), self.w_act)
        # K
        self.z_act = torch.softmax(self.z / self.tau_transformer, 1)
        self.z_act = torch.clamp(self.z_act, min=1e-10, max=1-1e-7)
        out_clusters = self.beta[0,:] + torch.matmul(g, self.beta[1:,:]*self.alpha_act)
        out = torch.matmul(out_clusters, self.z_act.T)
        return out_clusters, out


# Make X clusters distinct as possible
def generate_synthetic_data(N_met = 10, N_bug = 14, N_samples = 200, N_met_clusters = 2, N_bug_clusters = 2, state = 1,
                            beta_var = 2, cluster_disparity = 50, case = 'Case 1', num_nuisance=2):
    np.random.seed(state)
    choose_from = np.arange(N_met)
    met_gp_ids = []
    for n in range(N_met_clusters-1):
        num_choose = np.random.choice(np.arange(2,len(choose_from)-(N_met_clusters-n)),1)
        chosen = np.random.choice(choose_from, num_choose,replace = False)
        choose_from = list(set(choose_from) - set(chosen))
        met_gp_ids.append(chosen)
    met_gp_ids.append(np.array(choose_from))
    dist_met = np.zeros((N_met, N_met))
    for i, gp in enumerate(met_gp_ids):
        for met in gp:
            dist_met[met, gp] = np.ones((1, len(gp)))
            dist_met[gp, met] = dist_met[met,gp]

    rand = np.random.randint(0,4, size = dist_met.shape)
    rand = (rand + rand.T)/2
    dist_met[dist_met == 0] = 10
    dist_met = dist_met + rand
    np.fill_diagonal(dist_met, 0)

    choose_from = np.arange(N_bug)
    bug_gp_ids = []
    for n in range(N_bug_clusters-1):
        num_choose = np.random.choice(np.arange(2,len(choose_from)-(N_bug_clusters-n)),1)
        if num_choose < 5 and case == 'Case 2':
            num_choose = 5
        chosen = np.random.choice(choose_from, num_choose,replace = False)
        choose_from = list(set(choose_from) - set(chosen))
        bug_gp_ids.append(chosen)
    bug_gp_ids.append(np.array(choose_from))
    dist_bug = np.zeros((N_bug, N_bug))
    for i, gp in enumerate(bug_gp_ids):
        for met in gp:
            dist_bug[met, gp] = np.ones((1, len(gp)))
            dist_bug[gp, met] = dist_bug[met,gp]
    rand = np.random.randint(0,4, size = dist_bug.shape)
    rand = (rand + rand.T)/2
    dist_bug[dist_bug == 0] = 10
    dist_bug = dist_bug + rand
    np.fill_diagonal(dist_bug, 0)

    embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=state)
    met_locs = embedding.fit_transform(dist_met)
    embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=state)
    bug_locs = embedding.fit_transform(dist_bug)

    km_met = KMeans(n_clusters = N_met_clusters, random_state=state)
    kmeans_met = km_met.fit(met_locs)
    u_met_gen = kmeans_met.cluster_centers_
    z_gen = np.array([get_one_hot(kk, l = N_met_clusters) for kk in kmeans_met.labels_])

    km_bug = KMeans(n_clusters = N_bug_clusters, random_state=state)
    kmeans_bug = km_bug.fit(bug_locs)
    u_bug_gen = kmeans_bug.cluster_centers_
    w_gen = np.array([get_one_hot(kk, l = N_bug_clusters) for kk in kmeans_bug.labels_])

    betas = np.random.normal(0, np.sqrt(beta_var), size = (N_bug_clusters+1, N_met_clusters))
    alphas = st.bernoulli(0.5).rvs((N_bug_clusters, N_met_clusters))
    cluster_means = np.random.choice(
        np.arange(1, np.int(N_bug_clusters * cluster_disparity * 2) + 1, cluster_disparity), N_bug_clusters,
        replace=False)
    if N_bug_clusters==2 and N_met_clusters==2:
        betas = np.array([[0,0],[0,-2],[3,0]])
        alphas = np.array([[0, 1], [1, 0]])
        cluster_means = [10,100]
    X = np.zeros((N_samples, N_bug))

    if case == 'Case 2':
        w_gen = np.hstack((w_gen, np.zeros((w_gen.shape[0], 1))))
        N = int(num_nuisance/N_bug_clusters)
    for i in range(N_bug_clusters):
        ixs = np.where(w_gen[:,i]==1)[0]
        if case == 'Case 2':
            s = np.random.choice(ixs, N)
            X[:, s] = st.poisson(np.mean(cluster_means)).rvs(size = (N_samples, N))
            ixs = np.array(list(set(ixs) - set(s)))
            w_gen[s,:] = [0,0,1]
        X[:, ixs] = st.poisson(cluster_means[i]).rvs(size = (N_samples, len(ixs)))
    X = X/np.expand_dims(np.sum(X, 1),1)
    if case == 'Case 2':
        g = X@w_gen[:,:-1]
    else:
        g = X@w_gen
    y = (betas[0,:] + g@(betas[1:,:]*alphas))@z_gen.T + np.random.normal(0,1)
    return X, y, betas, alphas, w_gen, z_gen, bug_locs, met_locs, kmeans_bug, kmeans_met

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-learn", "--learn", help="params to learn", type=str, nargs='+')
    parser.add_argument("-priors", "--priors", help="priors to set", type=str, nargs='+')
    parser.add_argument("-case", "--case", help="case", type=str)
    args = parser.parse_args()
    # ['z','w','alpha','beta','mu_bug','mu_met','r_bug','r_met','pi_bug','pi_met']
    # 'z' 'w' 'alpha' 'beta' 'mu_bug' 'mu_met' 'r_bug' 'r_met' 'pi_bug' 'pi_met'
    if args.learn is not None:
        params2learn = args.learn
    if args.priors is not None:
        priors2set = args.priors
    else:
        priors2set = []
    if args.learn is None and args.priors is None:
        params2learn = ['all']
        priors2set = []

    N_met = 20
    N_bug = 20
    n_nuisance = 0

    if args.case is not None:
        case = args.case
    else:
        case = 'Case 2'

    n_splits = 2
    use_MAP = True
    meas_var = 4
    lr = 0.001
    temp_grouper, temp_selector = 'scheduled', 'scheduled'
    temp_transformer = 0.1
    info = 'lr_' + str(lr).replace('.','d') + '-tauZW_' + str(temp_grouper).replace('.','d') + '-tauA_' + \
        str(temp_selector).replace('.','d') + '-tauT_' + str(temp_transformer).replace('.','d')
    if use_MAP:
        path = 'results_MAP/'
    else:
        path = 'results_ML/'

    path = path + case.replace(' ','_')
    if not os.path.isdir(path):
        os.mkdir(path)

    path = path + '/N_bug' + str(N_bug) + '-N_met' + str(N_met) + '-N_nuisance' + str(n_nuisance) + '/'
    if not os.path.isdir(path):
        os.mkdir(path)
    path = path + '/learn_' + '_'.join(params2learn) + '-priors_' + '_'.join(priors2set) + '-' + info + '/'
    if not os.path.isdir(path):
        os.mkdir(path)

    if 'all' in priors2set:
        priors2set = ['z','w','alpha','beta','mu_bug','mu_met','r_bug','r_met','pi_bug','pi_met']
    x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, kmeans_bug, kmeans_met = generate_synthetic_data(
        case=case, N_met = N_met, N_bug = N_bug, num_nuisance=n_nuisance)

    r_bug = [np.max([np.sqrt(np.sum((kmeans_bug.cluster_centers_[i,:] - l)**2)) for l in gen_bug_locs[gen_w[:,i]==1,:]]) for i in
             range(kmeans_bug.cluster_centers_.shape[0])]
    r_met = [np.max([np.sqrt(np.sum((kmeans_met.cluster_centers_[i,:] - l)**2)) for l in gen_met_locs[gen_z[:,i]==1,:]]) for i in
             range(kmeans_met.cluster_centers_.shape[0])]
    true_vals = {'y':y, 'beta':gen_beta, 'alpha':gen_alpha, 'mu_bug': kmeans_bug.cluster_centers_, 'mu_met': kmeans_met.cluster_centers_,
                 'r_bug': r_bug, 'r_met': r_met, 'z': gen_z, 'w': gen_w, 'pi_met':np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0),
                 'pi_bug':np.expand_dims(np.sum(gen_w,0)/np.sum(np.sum(gen_w)),0)}
    plot_syn_data(path, x, y, gen_w, gen_z, gen_bug_locs, gen_met_locs, kmeans_bug.cluster_centers_,
                  r_bug, kmeans_met.cluster_centers_, r_met)

    dataset = (x,y)
    net_ = Model(gen_met_locs, gen_bug_locs, L = gen_w.shape[1], K = gen_z.shape[1], temp_grouper=temp_grouper,
                temp_selector = temp_selector, tau_transformer=0.1)

    net = Model(gen_met_locs, gen_bug_locs, L=gen_w.shape[1], K=gen_z.shape[1], temp_grouper=temp_grouper,
                 temp_selector=temp_selector, tau_transformer=temp_transformer)
    for param, dist in net.distributions.items():
        parameter_dict = net.params[param]
        plot_distribution(dist, param, true_val = true_vals[param], ptype = 'prior', path = path, **parameter_dict)

    for param, dist in net.initializations.items():
        plot_distribution(dist, param, true_val = getattr(net, param), ptype = 'init', path = path)
    kfold = KFold(n_splits = n_splits, shuffle = True)
    iterations =10001

    # z_vals = [net.z]

    train_x = x
    # fig_dict2, ax_dict2 = {},{}
    # fig_dict3, ax_dict3 = {},{}
    fig_dict4, ax_dict4 = {},{}
    fig_dict5, ax_dict5 = {},{}
    param_dict = {}
    tau_orig = copy.copy(net.temp_grouper)
    tau_alpha_orig = copy.copy(net.temp_selector)
    for fold in np.arange(n_splits):
        param_dict[fold] = {}

        net.temp_grouper = tau_orig
        net.temp_selector = tau_alpha_orig
        net.initialize(fold)
        for name, parameter in net.named_parameters():
            if name not in params2learn and 'all' not in params2learn:
                setattr(net, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))
                parameter = nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False)
            elif name == 'w' or name == 'z' or name == 'alpha':
                parameter = getattr(net, name + '_act')
            elif name == 'r_bug' or name == 'r_met':
                parameter = torch.exp(parameter)
            param_dict[fold][name] = [parameter.clone().detach().numpy()]
        print(param_dict[fold].keys())
        criterion = MAPloss(net, compute_loss_for=priors2set, meas_var = meas_var)
        criterion2 = nn.L1Loss()
        # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum = 0.9)
        optimizer = optim.RMSprop(net.parameters(),lr=lr)
        x_train, targets = x, y
        # x_test, test_targets = x[test_ids,:], y[test_ids]
        cluster_targets = np.stack([targets[:,np.where(gen_z[:,i]==1)[0][0]] for i in np.arange(gen_z.shape[1])]).T
        # cluster_test_targets = np.stack([test_targets[:,np.where(gen_z[:,i]==1)[0][0]] for i in np.arange(gen_z.shape[1])]).T
        loss_vec = []
        test_loss = []
        test_out_vec = []
        train_out_vec = []
        running_loss = 0.0
        timer = []
        end_learning = False

        net_.initialize(fold)
        for name, parameter in net_.named_parameters():
            if 'r' in name:
                setattr(net_, name, nn.Parameter(torch.log(torch.Tensor(true_vals[name])), requires_grad=False))
            elif 'pi' in name:
                val = torch.Tensor(true_vals[name])
                val = torch.log(val) + torch.log(torch.exp(val).sum())
                setattr(net_, name, nn.Parameter(val, requires_grad=False))
            else:
                setattr(net_, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))

        net_.z_act, net_.w_act, net_.alpha_act = torch.softmax(net_.z/0.1,1), torch.softmax(net_.w/0.1, 1), torch.sigmoid(net_.alpha/0.1)
        criterion_ = MAPloss(net_, compute_loss_for=priors2set, meas_var=meas_var)
        # lowest_loss = criterion_.compute_loss(torch.Tensor(targets), torch.Tensor(targets))
        lowest_loss = criterion_.compute_loss(torch.Tensor(cluster_targets), torch.Tensor(targets))
        print('Lowest Loss:' + str(lowest_loss.item()))
        # for param in criterion_.loss_dict.keys():
        #     print(param + ' Lowest Loss:' + str(criterion_.loss_dict[param].item()))

        tau_vec = []
        alpha_tau_vec = []
        lowest_loss_vec = []
        w_grad = []
        z_grad = []
        for epoch in range(iterations):
            if isinstance(temp_grouper, str) and ('z' in params2learn or 'w' in params2learn or 'all' in params2learn):
                if epoch%200==0 and epoch>400 and net.temp_grouper > 1e-20:
                    net.temp_grouper = net.temp_grouper/((epoch-400)/200)
                    net_.temp_grouper = net_.temp_grouper/((epoch-500)/200)
                tau_vec.append(net.temp_grouper)
            if isinstance(temp_selector, str) and ('alpha' in params2learn or 'all' in params2learn):
                if epoch%200==0 and epoch>400 and net.temp_selector > 1e-20:
                    net.temp_selector = net.temp_selector/((epoch-400)/200)
                    net_.temp_selector = net_.temp_selector/((epoch-500)/200)
                alpha_tau_vec.append(net.temp_selector)
            optimizer.zero_grad()
            cluster_outputs, outputs = net(x_train)
            train_out_vec.append(outputs)
            start = time.time()
            # if torch.isnan(net.w).any():
            #     net.w[torch.isnan(net.w)] = net.initializations['w'].sample([1, net.L])
            #     print('debug')
            if use_MAP:
                try:
                    loss = criterion.compute_loss(cluster_outputs,torch.Tensor(targets))
                except:
                    print('debug')
                    loss = criterion.compute_loss(cluster_outputs,torch.Tensor(targets))
            else:
                loss = criterion2(outputs, torch.Tensor(targets))

            lowest_loss_vec.append(lowest_loss.item())
            loss_vec.append(loss.item())
            timer.append(time.time()-start)
            loss.backward()
            w_grad.append(net.w.grad.detach().numpy())
            z_grad.append(net.z.grad.detach().numpy())
            optimizer.step()
            for name, parameter in net.named_parameters():
                if name == 'w' or name == 'z' or name == 'alpha':
                    parameter = getattr(net, name + '_act')
                elif name == 'r_bug' or name == 'r_met':
                    parameter = torch.exp(parameter)
                param_dict[fold][name].append(parameter.clone().detach().numpy())


            if (epoch%1000 == 0 and epoch != 0) or end_learning:
                print('Epoch ' + str(epoch) + ' Loss: ' + str(loss_vec[-1]))
                lowest_loss = criterion_.compute_loss(torch.Tensor(cluster_targets), torch.Tensor(targets))
                print('Lowest Loss: ' + str(lowest_loss.item()))
                print('Tau: ' + str(net.temp_grouper))
                print('')
                # train_out_vec.append(outputs)
                # fig_dict2[fold], ax_dict2[fold] = plt.subplots(y.shape[1], 1, figsize=(8, 4 * y.shape[1]))
                # fig_dict3[fold], ax_dict3[fold] = plt.subplots(gen_z.shape[1], 1, figsize=(8, 4 * gen_z.shape[1]))
                fig_dict4[fold], ax_dict4[fold] = plt.subplots(y.shape[1], 1, figsize=(8, 4 * y.shape[1]))
                fig_dict5[fold], ax_dict5[fold] = plt.subplots(gen_z.shape[1], 1, figsize=(8, 4 * gen_z.shape[1]))
                plot_param_traces(path, param_dict[fold], params2learn, true_vals, net, fold)
                # plot_output(path, test_loss, test_out_vec, test_targets, gen_z, gen_w, param_dict[fold],
                #                 fig_dict2, ax_dict2, fig_dict3,ax_dict3, fold, type = 'test')
                plot_output(path, loss_vec, train_out_vec, targets, gen_z, gen_w, param_dict[fold],
                                     fig_dict4, ax_dict4, fig_dict5, ax_dict5, fold, type = 'train')
                plot_output_locations(path, net, loss_vec, param_dict[fold])

                # loss_vec.append(running_loss/10)
                # running_loss = 0
            if epoch%100==0 or end_learning:
                # with torch.no_grad():
                #     test_cluster_out, test_out = net(x_test)
                #     test_out_vec.append(test_out)
                #     if use_MAP:
                #         test_loss.append(criterion.compute_loss(test_cluster_out,torch.Tensor(cluster_test_targets)))
                #         # test_loss.append(criterion.compute_loss(test_out, torch.Tensor(test_targets)))
                #     else:
                #         test_loss.append(criterion2(test_out, torch.Tensor(test_targets)))
                # # if len(test_loss)>5 and test_loss[-1]>=test_loss[-2] and test_loss[-2]>=test_loss[-3] and test_loss[-3]>=test_loss[-4]:
                # #     end_learning = True
                if epoch != 0:
                    fig3, ax3 = plt.subplots(figsize=(8, 4 * n_splits))
                    fig3, ax3 = plot_loss(fig3, ax3, fold, epoch+1, loss_vec, lowest_loss=lowest_loss_vec)
                    fig3.tight_layout()
                    fig3.savefig(path + 'loss_fold_' + str(fold) + '.pdf')
                    plt.close(fig3)

            if end_learning:
                print('end learning, fold ' + str(fold))
                break

        if isinstance(temp_grouper, str) and len(tau_vec)>0:
            fig, ax = plt.subplots()
            ax.semilogy(range(iterations), tau_vec)
            fig.savefig(path + 'fold' + str(fold) + '_tau_scheduler.pdf')
            plt.close(fig)
        if isinstance(temp_selector, str) and len(alpha_tau_vec)>0:
            fig, ax = plt.subplots()
            ax.semilogy(range(iterations), alpha_tau_vec)
            fig.savefig(path + 'fold' + str(fold) + '_alpha_tau_scheduler.pdf')
            plt.close(fig)
        # if fold == 0:
        #     plot_param_traces(path, param_dict, params2learn, true_vals, net, fig_dict, ax_dict)

        # fig3, ax3 = plot_loss(fig3, ax3, fold, iterations, loss_vec, test_loss)

