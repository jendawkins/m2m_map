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


class Model(nn.Module):
    def __init__(self, met_locs, microbe_locs, temp_grouper = 10, temp_selector = 10, L = 2, K = 2, beta_var = 10.,
                 r_scale_met = 20., r_scale_bug = 20., seed = 0, tau_transformer = 1):
        super(Model, self).__init__()
        self.L = L
        self.K = K
        self.beta_var = beta_var
        self.mu_var_met = (1/met_locs.shape[1])*np.sum(np.var(met_locs.T))
        self.mu_var_bug = (1/microbe_locs.shape[1])*np.sum(np.var(microbe_locs.T))
        self.temp_grouper = temp_grouper
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
        self.r_scale_bug = np.sqrt(range_x**2 + range_y**2)

        self.params = {}
        self.distributions = {}
        self.params['beta'] = {'mean': 0, 'scale': self.beta_var}
        self.params['alpha'] = {'loc': self.alpha_loc, 'temp':self.temp_selector}
        self.params['mu_met'] = {'mean': 0, 'var': self.mu_var_met}
        self.params['mu_bug'] = {'mean': 0, 'var': self.mu_var_bug}
        self.params['r_bug'] = {'dof': self.L, 'scale': self.r_scale_bug}
        self.params['r_met'] = {'dof': self.K, 'scale': self.r_scale_met}
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
            self.range_dict[param] = (sampler.min()-range*0.1, sampler.max() + range*0.1)
        self.range_dict['w'] = (-0.1,1.1)
        self.range_dict['z'] = (-0.1,1.1)
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
        # beta_dist = Normal(0, np.sqrt(self.beta_var))
        self.beta = nn.Parameter(self.initializations['beta'].sample([self.L+1, self.K]), requires_grad=True)

        self.alpha = nn.Parameter(self.initializations['alpha'].sample([self.K, self.L]), requires_grad=True)
        self.alpha_act = torch.sigmoid(self.alpha / self.temp_selector)
        self.mu_met = nn.Parameter(self.initializations['mu_met'].sample(sample_shape = torch.Size([self.K])),
                                        requires_grad = True)
        self.mu_bug = nn.Parameter(self.initializations['mu_bug'].sample(sample_shape = torch.Size([self.L])),
                                        requires_grad = True)
        r_temp = self.initializations['r_met'].sample([self.K])
        self.r_met = nn.Parameter(1/r_temp, requires_grad = True)

        r_temp = self.initializations['r_bug'].sample([self.L])
        self.r_bug = nn.Parameter(1/r_temp, requires_grad=True)

        self.pi_bug = nn.Parameter(self.initializations['pi_bug'].sample(), requires_grad=True)
        self.pi_met = nn.Parameter(self.initializations['pi_met'].sample(), requires_grad=True)

        # cat_bug = Categorical(self.pi_bug).sample([self.microbe_locs.shape[0]])
        # temp = nn.functional.one_hot(cat_bug.squeeze(), num_classes = self.L).type(torch.FloatTensor)
        self.w = nn.Parameter(Normal(0,1).sample([self.microbe_locs.shape[0], self.L]), requires_grad=True)
        self.w_act = torch.softmax(self.w,1)

        # cat_met = Categorical(self.pi_met).sample([self.met_locs.shape[0]])
        # temp = nn.functional.one_hot(cat_met.squeeze(), num_classes = self.K).type(torch.FloatTensor)
        # self.range_dict['z'] = (-0.1,1.1)
        self.z = nn.Parameter(Normal(0,1).sample([self.met_locs.shape[0], self.K]), requires_grad=True)
        self.z_act = torch.softmax(self.z, 1)

    def forward(self, x):
        self.alpha_act = torch.sigmoid(self.alpha/self.tau_transformer)
        self.alpha_act = torch.clamp(self.alpha_act, min=1e-20, max=1-1e-7)
        self.w_act = torch.softmax(self.w / self.tau_transformer, 1)
        self.w_act = torch.clamp(self.w_act, min=1e-20, max=1-1e-7)
        g = torch.matmul(torch.Tensor(x), self.w_act)
        # K
        self.z_act = torch.softmax(self.z / self.tau_transformer, 1)
        self.z_act = torch.clamp(self.z_act, min=1e-20, max=1-1e-7)
        out_clusters = self.beta[0,:] + torch.matmul(g, self.beta[1:,:]*self.alpha_act)
        out = torch.matmul(out_clusters, self.z_act.T)
        return out_clusters, out


# Make X clusters distinct as possible
def generate_synthetic_data(N_met = 6, N_bug = 5, N_samples = 200, N_met_clusters = 2, N_bug_clusters = 2, state = 1,
                            beta_var = 2, cluster_disparity = 50):
    np.random.seed(state)
    choose_from = np.arange(N_met)
    met_gp_ids = []
    for n in range(N_met_clusters-1):
        num_choose = np.random.choice(np.arange(2,len(choose_from)-n),1)
        chosen = np.random.choice(choose_from, num_choose,replace = False)
        choose_from = list(set(choose_from) - set(chosen))
        met_gp_ids.append(chosen)
    met_gp_ids.append(np.array(choose_from))
    dist_met = np.zeros((N_met, N_met))
    for i, gp in enumerate(met_gp_ids):
        for met in gp:
            dist_met[met, gp] = np.random.randint(1,2, size = (1, len(gp)))
            dist_met[gp, met] = dist_met[met,gp]
    dist_met[dist_met == 0] = np.random.randint(12, 20, size=np.sum(dist_met == 0))
    dist_met = (dist_met.T@dist_met)/15
    np.fill_diagonal(dist_met, 0)

    choose_from = np.arange(N_bug)
    bug_gp_ids = []
    for n in range(N_bug_clusters-1):
        num_choose = np.random.choice(np.arange(2,len(choose_from)-n),1)
        chosen = np.random.choice(choose_from, num_choose,replace = False)
        choose_from = list(set(choose_from) - set(chosen))
        bug_gp_ids.append(chosen)
    bug_gp_ids.append(np.array(choose_from))
    dist_bug = np.zeros((N_bug, N_bug))
    for i, gp in enumerate(bug_gp_ids):
        for met in gp:
            dist_bug[met, gp] = np.random.randint(1,2, size = (1, len(gp)))
            dist_bug[gp, met] = dist_met[met,gp]

    dist_bug[dist_bug == 0] = np.random.randint(12, 20, size=np.sum(dist_bug == 0))
    dist_bug = (dist_bug.T@dist_bug)/15
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
        alphas = np.array([[1, 1], [1, 1]])
        cluster_means = [10,100]
    X = np.zeros((N_samples, N_bug))
    for i in range(N_bug_clusters):
        ixs = np.where(w_gen[:,i]==1)[0]
        X[:, ixs] = st.poisson(cluster_means[i]).rvs(size = (N_samples, len(ixs)))
    X = X/np.expand_dims(np.sum(X, 1),1)
    g = X@w_gen
    y = (betas[0,:] + g@(betas[1:,:]*alphas))@z_gen.T + np.random.normal(0,1)
    return X, y, betas, alphas, w_gen, z_gen, bug_locs, met_locs, kmeans_bug, kmeans_met

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    # ['z','w','alpha','beta','mu_bug','mu_met','r_bug','r_met','pi_bug','pi_met']
    params2learn = ['z']
    priors2set = []
    n_splits = 2
    use_MAP = True
    meas_var = 4
    lr = 0.01
    temp_grouper = 0.1
    info = 'lr_' + str(lr).replace('.','d') + '-tau_' + str(temp_grouper).replace('.','d')
    if use_MAP:
        path = 'results_MAP/learn_' + '_'.join(params2learn) + '-priors_' + '_'.join(priors2set) + '-' + info + '/'
    else:
        path = 'results_ML/learn_' + '_'.join(params2learn) + '/'
    if not os.path.isdir(path):
        os.mkdir(path)
    x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, kmeans_bug, kmeans_met = generate_synthetic_data()
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
    net = Model(gen_met_locs, gen_bug_locs, L = gen_w.shape[1], K = gen_z.shape[1], temp_grouper=temp_grouper)


    for param, dist in net.distributions.items():
        parameter_dict = net.params[param]
        plot_distribution(dist, param, ptype = 'prior', path = path, **parameter_dict)

    for param, dist in net.initializations.items():
        parameter_dict = net.params[param]
        plot_distribution(dist, param, ptype = 'init', path = path, **parameter_dict)
    kfold = KFold(n_splits = n_splits, shuffle = True)
    iterations =5001

    # z_vals = [net.z]

    train_x = x
    test_out_vec = []
    train_out_vec = []
    loss_vec = []
    test_loss = []
    fig_dict2, ax_dict2 = {},{}
    fig_dict3, ax_dict3 = {},{}
    fig_dict4, ax_dict4 = {},{}
    fig_dict5, ax_dict5 = {},{}
    param_dict = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(x)):
        param_dict[fold] = {}
        net.initialize(net.seed)
        # net.true_z = torch.softmax(torch.Tensor(gen_z) / 0.001, 1)
        # net.z = torch.nn.Parameter(torch.clone(net.true_z), requires_grad=True)
        # net.z_act = torch.softmax(net.z/0.1,1)
        # print(net.z_act)
        for name, parameter in net.named_parameters():
            if name not in params2learn and 'all' not in params2learn:
                setattr(net, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))
                parameter = nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False)
            elif name == 'w' or name == 'z' or name == 'alpha':
                parameter = getattr(net, name + '_act')
            elif name == 'r_bug' or name == 'r_met':
                parameter = torch.exp(parameter)
            param_dict[fold][name] = [parameter.clone().detach().numpy()]
        criterion = MAPloss(net, compute_loss_for=priors2set, meas_var = meas_var)
        criterion2 = nn.L1Loss()
        # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum = 0.9)
        optimizer = optim.RMSprop(net.parameters(),lr=lr)
        x_train, targets = x[train_ids,:], y[train_ids]
        x_test, test_targets = x[test_ids,:], y[test_ids]
        cluster_targets = np.stack([targets[:,np.where(gen_z[:,i]==1)[0][0]] for i in np.arange(gen_z.shape[1])]).T
        cluster_test_targets = np.stack([test_targets[:,np.where(gen_z[:,i]==1)[0][0]] for i in np.arange(gen_z.shape[1])]).T
        loss_vec = []
        test_loss = []
        running_loss = 0.0
        timer = []
        end_learning = False

        temp = Normal(torch.Tensor(cluster_targets), np.sqrt(meas_var)
                      ).log_prob(torch.Tensor(cluster_targets))
        lowest_loss = (-torch.log(torch.matmul(torch.exp(temp), net.true_z.T))).sum()
        print('Lowest Loss:' + str(lowest_loss.item()))

        for epoch in range(iterations):
            optimizer.zero_grad()
            cluster_outputs, outputs = net(x_train)
            train_out_vec.append(outputs)
            start = time.time()
            if epoch%1200==0 and epoch!=0:
                print('debug')
            if use_MAP:
                loss = criterion.compute_loss(cluster_outputs,torch.Tensor(cluster_targets))
                # loss = criterion.compute_loss(outputs, torch.Tensor(targets))
            else:
                loss = criterion2(outputs, torch.Tensor(targets))

            loss_vec.append(loss.item())
            timer.append(time.time()-start)
            loss.backward()
            optimizer.step()
            for name, parameter in net.named_parameters():
                if name == 'w' or name == 'z' or name == 'alpha':
                    parameter = getattr(net, name + '_act')
                elif name == 'r_bug' or name == 'r_met':
                    parameter = torch.exp(parameter)
                param_dict[fold][name].append(parameter.clone().detach().numpy())


            if (epoch%1000 == 0 and epoch != 0) or end_learning:
                print('Epoch ' + str(epoch) + ' Loss: ' + str(loss_vec[-1]))
                print('Total time: ' + str(np.cumsum(timer)[-1]/60) + ' min')
                print('Epoch time: ' + str(np.mean(timer[-100:])) + ' sec')
                # train_out_vec.append(outputs)
                fig_dict2[fold], ax_dict2[fold] = plt.subplots(y.shape[1], 1, figsize=(8, 4 * y.shape[1]))
                fig_dict3[fold], ax_dict3[fold] = plt.subplots(gen_z.shape[1], 1, figsize=(8, 4 * gen_z.shape[1]))
                fig_dict4[fold], ax_dict4[fold] = plt.subplots(y.shape[1], 1, figsize=(8, 4 * y.shape[1]))
                fig_dict5[fold], ax_dict5[fold] = plt.subplots(gen_z.shape[1], 1, figsize=(8, 4 * gen_z.shape[1]))
                plot_param_traces(path, param_dict[fold], params2learn, true_vals, net, fold)
                plot_output(path, test_loss, test_out_vec, test_targets, gen_z, gen_w, param_dict[fold],
                                fig_dict2, ax_dict2, fig_dict3,ax_dict3, fold)
                plot_training_output(path, loss_vec, train_out_vec, targets, gen_z, gen_w, param_dict[fold],
                                     fig_dict4, ax_dict4, fig_dict5, ax_dict5, fold)

                # loss_vec.append(running_loss/10)
                # running_loss = 0
            if epoch%100==0 or end_learning:
                with torch.no_grad():
                    test_cluster_out, test_out = net(x_test)
                    test_out_vec.append(test_out)
                    if use_MAP:
                        test_loss.append(criterion.compute_loss(test_cluster_out,torch.Tensor(cluster_test_targets)))
                        # test_loss.append(criterion.compute_loss(test_out, torch.Tensor(test_targets)))
                    else:
                        test_loss.append(criterion2(test_out, torch.Tensor(test_targets)))
                # if len(test_loss)>5 and test_loss[-1]>=test_loss[-2] and test_loss[-2]>=test_loss[-3] and test_loss[-3]>=test_loss[-4]:
                #     end_learning = True
                if epoch != 0:
                    fig3, ax3 = plt.subplots(figsize=(8, 4 * n_splits))
                    fig3, ax3 = plot_loss(fig3, ax3, fold, epoch+1, loss_vec, test_loss)
                    fig3.tight_layout()
                    fig3.savefig(path + 'loss_fold_' + str(fold) + '.pdf')
                    plt.close(fig3)

            if end_learning:
                print('end learning, fold ' + str(fold))
                break
        # if fold == 0:
        #     plot_param_traces(path, param_dict, params2learn, true_vals, net, fig_dict, ax_dict)

        # fig3, ax3 = plot_loss(fig3, ax3, fold, iterations, loss_vec, test_loss)

