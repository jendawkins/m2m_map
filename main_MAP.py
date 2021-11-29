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
                 r_scale_met = 1., r_scale_bug = 1., seed = 0, tau_transformer = 0.1):
        super(Model, self).__init__()
        self.L = L
        self.K = K
        self.beta_var = beta_var
        self.mu_var_met = (1/met_locs.shape[1])*np.sum(np.var(met_locs.T))
        self.mu_var_bug = (1/microbe_locs.shape[1])*np.sum(np.var(microbe_locs.T))
        self.r_scale_met = r_scale_met
        self.r_scale_bug = r_scale_bug
        self.temp_grouper = temp_grouper
        self.temp_selector = temp_selector
        self.met_locs = met_locs
        self.microbe_locs = microbe_locs
        self.embedding_dim = met_locs.shape[1]
        self.seed = seed
        self.alpha_loc = 1.
        self.tau_transformer = tau_transformer
        self.initialize(self.seed)

    def priors(self):
        self.distributions = {}
        self.distributions['beta'] = Normal(0, np.sqrt(self.beta_var))
        self.distributions['alpha'] = BinaryConcrete(self.alpha_loc, self.temp_selector)

        self.e_bug = []
        len_st = self.microbe_locs.shape[0]
        for i in range(self.L-1):
            samp = np.random.choice(np.arange(1, len_st-(self.L-1)),1)
            self.e_bug.append(samp[0])
            len_st = len_st - samp[0]
        self.e_bug.append(len_st)
        self.e_met = []
        len_st = self.met_locs.shape[0]
        for i in range(self.K-1):
            samp = np.random.choice(np.arange(1, len_st-(self.K-1)),1)
            self.e_met.append(samp[0])
            len_st = len_st - samp[0]
        self.e_met.append(len_st)

        self.distributions['pi_met'] = Dirichlet(torch.Tensor(self.e_met))
        self.distributions['pi_bug'] = Dirichlet(torch.Tensor(self.e_bug))
        self.distributions['mu_met'] = MultivariateNormal(torch.zeros(self.embedding_dim), self.mu_var_met*torch.eye(self.embedding_dim))
        self.distributions['mu_bug'] = MultivariateNormal(torch.zeros(self.embedding_dim), self.mu_var_bug*torch.eye(self.embedding_dim))
        self.distributions['r_bug'] = Gamma(self.L, self.r_scale_met)

        # self.distributions['w'] = BinaryConcrete(self.)


    def initialize(self, seed):
        torch.manual_seed(seed)
        self.range_dict = {}
        beta_dist = Normal(0, np.sqrt(self.beta_var))
        self.beta = nn.Parameter(beta_dist.sample([self.L+1, self.K]), requires_grad=True)
        sampler = beta_dist.sample([100])
        self.range_dict['beta'] = (sampler.min(), sampler.max())

        self.alpha = nn.Parameter(Bernoulli(0.5).sample([self.K, self.L]), requires_grad=True)
        self.alpha_act = torch.sigmoid(self.alpha / self.temp_selector)
        self.range_dict['alpha'] = (-0.1, 1.1)

        mv_normal = MultivariateNormal(torch.zeros(self.embedding_dim), self.mu_var_met*torch.eye(self.embedding_dim))
        self.range_dict['mu_met'] = (mv_normal.sample([100]).min(), mv_normal.sample([100]).max())
        self.range_dict['mu_bug'] = (mv_normal.sample([100]).min(), mv_normal.sample([100]).max())
        self.mu_met = nn.Parameter(mv_normal.sample(sample_shape = torch.Size([self.K])),
                                        requires_grad = True)
        self.mu_bug = nn.Parameter(mv_normal.sample(sample_shape = torch.Size([self.L])),
                                        requires_grad = True)
        range_x = np.max(self.met_locs[:,0]) - np.min(self.met_locs[:,0])
        range_y = np.max(self.met_locs[:,1]) - np.min(self.met_locs[:,1])
        euc = np.sqrt(range_x**2 + range_y**2)
        self.r0_met = euc / self.K
        self.r_scale_met = euc
        # r0_met = self.met_locs[np.random.choice(np.arange(self.met_locs.shape[0]), size = self.K),:]
        gamma = Gamma(self.r_scale_met/2, torch.tensor((self.r_scale_met*(self.r0_met)))/2)
        plot_distribution(gamma, 'r_met', ptype='init', scale=self.r_scale_met, loc=self.r0_met)
        r_temp = gamma.sample([self.K])
        self.range_dict['r_met'] = (0, euc)
        self.r_met = nn.Parameter(1/r_temp, requires_grad = True)
        self.r_scale_met, self.r0_met = 2.,10.

        range_x = np.max(self.microbe_locs[:,0]) - np.min(self.microbe_locs[:,0])
        range_y = np.max(self.microbe_locs[:,1]) - np.min(self.microbe_locs[:,1])
        euc = np.sqrt(range_x**2 + range_y**2)
        self.r0_bug = euc / self.K
        gamma = Gamma(self.r_scale_bug/2, torch.tensor((self.r_scale_bug*(self.r0_bug))/2))
        self.range_dict['r_bug'] = (0, euc)
        r_temp = gamma.sample([self.L])
        self.r_bug = nn.Parameter(1/r_temp, requires_grad=True)
        self.r_scale_bug = 2

        self.e_bug = []
        len_st = self.microbe_locs.shape[0]
        for i in range(self.L-1):
            samp = np.random.choice(np.arange(1, len_st-(self.L-1)),1)
            self.e_bug.append(samp[0])
            len_st = len_st - samp[0]
        self.e_bug.append(len_st)
        self.e_met = []
        len_st = self.met_locs.shape[0]
        for i in range(self.K-1):
            samp = np.random.choice(np.arange(1, len_st-(self.K-1)),1)
            self.e_met.append(samp[0])
            len_st = len_st - samp[0]
        self.e_met.append(len_st)
        self.pi_bug = nn.Parameter(torch.Tensor(st.dirichlet(self.e_bug).rvs()), requires_grad=True)
        self.pi_met = nn.Parameter(torch.Tensor(st.dirichlet(self.e_met).rvs()), requires_grad=True)
        self.range_dict['pi_bug'] = (0,1)
        self.range_dict['pi_met'] = (0,1)

        cat_bug = Categorical(self.pi_bug).sample([self.microbe_locs.shape[0]])
        temp = nn.functional.one_hot(cat_bug.squeeze(), num_classes = self.L).type(torch.FloatTensor)
        self.range_dict['w'] = (-0.1,1.1)
        self.w = nn.Parameter(temp, requires_grad=True)
        self.w_act = torch.softmax(self.w/self.temp_grouper,1)

        cat_met = Categorical(self.pi_met).sample([self.met_locs.shape[0]])
        temp = nn.functional.one_hot(cat_met.squeeze(), num_classes = self.K).type(torch.FloatTensor)
        self.range_dict['z'] = (-0.1,1.1)
        self.z = nn.Parameter(temp, requires_grad=True)
        self.z_act = torch.softmax(self.z / self.tau_transformer, 1)

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
def generate_synthetic_data(N_met = 6, N_bug = 4, N_samples = 200, N_met_clusters = 2, N_bug_clusters = 2, state = 1,
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
    params2learn = ['beta']
    priors2set = []
    n_splits = 2
    use_MAP = True
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

    kfold = KFold(n_splits = n_splits, shuffle = True)
    iterations = 10000

    # z_vals = [net.z]

    train_x = x
    test_out_vec = []
    train_out_vec = []
    loss_vec = []
    test_loss = []
    fig_dict2, ax_dict2 = {},{}
    fig_dict3, ax_dict3 = {},{}
    param_dict = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(x)):
        param_dict[fold] = {}
        net.initialize(net.seed)
        for name, parameter in net.named_parameters():
            if name not in params2learn and 'all' not in params2learn:
                setattr(net, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))
                parameter = nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False)
            elif name == 'w' or name == 'z' or name == 'alpha':
                parameter = getattr(net, name + '_act')
            elif name == 'r_bug' or name == 'r_met':
                parameter = torch.exp(parameter)
            param_dict[fold][name] = [parameter.clone().detach().numpy()]
        criterion = MAPloss(net, compute_loss_for=priors2set)
        criterion2 = nn.L1Loss()
        # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum = 0.9)
        optimizer = optim.RMSprop(net.parameters(),lr=lr)
        x_train, targets = x[train_ids,:], y[train_ids]
        x_test, test_targets = x[test_ids,:], y[test_ids]
        loss_vec = []
        test_loss = []
        running_loss = 0.0
        timer = []
        end_learning = False
        for epoch in range(iterations):
            optimizer.zero_grad()
            cluster_outputs, outputs = net(x_train)
            start = time.time()
            if use_MAP:
                loss = criterion.compute_loss(cluster_outputs,
                                              torch.matmul(torch.Tensor(targets), torch.Tensor(gen_z)))
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
                print(epoch)
                print('Total time: ' + str(np.cumsum(timer)[-1]/60) + ' min')
                print('Epoch time: ' + str(np.mean(timer[-100:])) + ' sec')
                # train_out_vec.append(outputs)
                fig_dict2[fold], ax_dict2[fold] = plt.subplots(y.shape[1], 1, figsize=(8, 4 * y.shape[1]))
                fig_dict3[fold], ax_dict3[fold] = plt.subplots(gen_z.shape[1], 1, figsize=(8, 4 * gen_z.shape[1]))
                plot_param_traces(path, param_dict[fold], params2learn, true_vals, net, fold)
                plot_output(path, test_loss, test_out_vec, test_targets, gen_z, gen_w, param_dict[fold],
                                fig_dict2, ax_dict2, fig_dict3,ax_dict3, fold)

                # loss_vec.append(running_loss/10)
                # running_loss = 0
            if epoch%100==0 or end_learning:
                train_out_vec.append(outputs)
                with torch.no_grad():
                    test_cluster_out, test_out = net(x_test)
                    test_out_vec.append(test_out)
                    if use_MAP:
                        test_loss.append(criterion.compute_loss(test_cluster_out,
                                                            torch.matmul(torch.Tensor(test_targets), torch.Tensor(gen_z))))
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

