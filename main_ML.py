import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import torch.optim as optim
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from helper import *
from sklearn.model_selection import KFold
from scipy.special import logit
from plot_helper import *

class Model(nn.Module):
    def __init__(self, met_locs, microbe_locs, temp_grouper = 1, temp_selector = 1, L = 2, K = 2, beta_var = 10, c_var = 1,
                 mu_var_met = 10, mu_var_bug = 10, r_scale_met = 10, r_scale_bug = 10, seed = 0):
        super(Model, self).__init__()
        self.L = L
        self.K = K
        self.beta_var = beta_var
        self.c_var = c_var
        self.mu_var_met = mu_var_met
        self.mu_var_bug = mu_var_bug
        self.r_scale_met = r_scale_met
        self.r_scale_bug = r_scale_bug
        self.temp_grouper = temp_grouper
        self.temp_selector = temp_selector
        self.met_locs = met_locs
        self.microbe_locs = microbe_locs
        self.embedding_dim = met_locs.shape[1]
        self.seed = seed
        self.initialize(self.seed)

    def initialize(self, seed):
        torch.manual_seed(seed)
        self.range_dict = {}
        self.beta = nn.Parameter(torch.normal(0, self.beta_var, size = (self.L+1, self.K), requires_grad=True))
        sampler = torch.normal(0, self.beta_var, size = [100])
        self.range_dict['beta'] = (sampler.min(), sampler.max())
        self.alpha = nn.Parameter(Bernoulli(0.5).sample([self.K, self.L]), requires_grad=True)
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
        # r0_met = self.met_locs[np.random.choice(np.arange(self.met_locs.shape[0]), size = self.K),:]
        gamma = Gamma(self.r_scale_met/2, torch.tensor((self.r_scale_met*(self.r0_met)))/2)
        r_temp = gamma.sample([self.K])
        self.range_dict['r_met'] = (0, (1/gamma.sample([100])).max())
        self.r_met = nn.Parameter(1/r_temp, requires_grad = True)

        range_x = np.max(self.microbe_locs[:,0]) - np.min(self.microbe_locs[:,0])
        range_y = np.max(self.microbe_locs[:,1]) - np.min(self.microbe_locs[:,1])
        euc = np.sqrt(range_x**2 + range_y**2)
        self.r0_bug = euc / self.K
        gamma = Gamma(self.r_scale_bug/2, torch.tensor((self.r_scale_bug*(self.r0_bug))/2))
        self.range_dict['r_bug'] = (0, (1/gamma.sample([100])).max())
        r_temp = gamma.sample([self.L])
        self.r_bug = nn.Parameter(1/r_temp, requires_grad=True)

        self.e_bug = []
        len_st = self.microbe_locs.shape[0]
        for i in range(self.L-1):
            samp = np.random.choice(np.arange(len_st - self.L),1)
            self.e_bug.append(samp[0])
            len_st = len_st - samp[0]
        self.e_bug.append(len_st)
        self.e_met = []
        len_st = self.met_locs.shape[0]
        for i in range(self.K-1):
            samp = np.random.choice(np.arange(len_st-self.K),1)
            self.e_met.append(samp[0])
            len_st = len_st - samp[0]
        self.e_met.append(len_st)
        self.pi_bug = nn.Parameter(torch.Tensor(st.dirichlet(self.e_bug).rvs()), requires_grad=True)
        self.pi_met = nn.Parameter(torch.Tensor(st.dirichlet(self.e_met).rvs()), requires_grad=True)
        self.range_dict['pi_bug'] = (0,1)
        self.range_dict['pi_met'] = (0,1)

        cat_bug = Categorical(self.pi_bug).sample([self.microbe_locs.shape[0]])
        self.w = nn.Parameter(nn.functional.one_hot(cat_bug.squeeze(), num_classes = self.L).type(torch.FloatTensor), requires_grad=True)
        self.range_dict['w'] = (-0.1,1.1)

        cat_met = Categorical(self.pi_met).sample([self.met_locs.shape[0]])
        self.z = nn.Parameter(nn.functional.one_hot(cat_met.squeeze(), num_classes = self.K).type(torch.FloatTensor), requires_grad=True)
        self.range_dict['z'] = (-0.1,1.1)

    def forward(self, x):
        # Transform z and w with softmax to be on simplex
        # Will have to do this with the prior anyways
        # Add alpha to simplex
        self.alpha_act = torch.sigmoid(self.alpha/self.temp_selector)
        self.w_act = torch.softmax(self.w / self.temp_grouper, 1)
        self.z_act = torch.softmax(self.z / self.temp_grouper, 1)

        g = torch.matmul(torch.Tensor(x), self.w_act.float())
        # K
        out_clusters = self.beta[0,:] + torch.matmul(g, self.beta[1:,:]*self.alpha_act)
        out = torch.matmul(out_clusters, self.z_act.float().T)
        return out


# Clusters need to be not correlated to eachother - if too similar to eachother, correlation in the data
# Linear relationship is with sums - i.e. aggregation of clusters and metabolites
# All y_j's in the same cluster will have basically the same estimation
# 2 clusters metabolites, 2 clusters microbes
# Make sure betas are separate enough
# Toy example where the sums of x create y
# Main thing - do we get the clusters right?
def generate_synthetic_data(N_met = 20, N_bug = 15, N_samples = 200, N_met_clusters = 2, N_bug_clusters = 2, state = 1,
                            beta_var = 10):
    np.random.seed(state)
    dist_met = np.random.randint(1,10,size = (N_met, N_met))
    dist_met = (dist_met.T@dist_met)/15
    np.fill_diagonal(dist_met, 0)

    dist_bug = np.random.randint(1,10,size = (N_bug, N_bug))
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
    # betas = np.array([[0,0],[1,0],[0,1]])
    alphas = st.bernoulli(0.5).rvs((N_bug_clusters, N_met_clusters))
    # alphas = np.array([[1,1],[1,1]])
    cluster_means = np.random.choice(np.arange(np.int(-N_bug_clusters*2), np.int(N_bug_clusters*2)+1, 4), N_bug_clusters, replace = False)
    X = np.zeros((N_samples, N_bug))
    for i in range(N_bug_clusters):
        ixs = np.where(w_gen[:,i]==1)[0]
        X[:, ixs] = np.random.normal(cluster_means[i], 1, size = (N_samples, len(ixs)))
    g = X@w_gen
    y = (betas[0,:] + g@(betas[1:,:]*alphas))@z_gen.T
    return X, y, betas, alphas, w_gen, z_gen, bug_locs, met_locs, kmeans_bug, kmeans_met

if __name__ == "__main__":
    params2learn = ['w']
    n_splits = 2
    path = 'results_ML/learn_' + '_'.join(params2learn) + '/'
    if not os.path.isdir(path):
        os.mkdir(path)
    x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, kmeans_bug, kmeans_met = generate_synthetic_data()
    r_bug = [np.max([np.sqrt(np.sum((kmeans_bug.cluster_centers_[i,:] - l)**2)) for l in gen_bug_locs[gen_w[:,i]==1,:]]) for i in
             range(kmeans_bug.cluster_centers_.shape[0])]
    r_met = [np.max([np.sqrt(np.sum((kmeans_met.cluster_centers_[i,:] - l)**2)) for l in gen_met_locs[gen_z[:,i]==1,:]]) for i in
             range(kmeans_met.cluster_centers_.shape[0])]
    true_vals = {'y':y, 'beta':gen_beta, 'alpha':gen_alpha, 'mu_bug': kmeans_bug.cluster_centers_, 'mu_met': kmeans_met.cluster_centers_,
                 'r_bug': r_bug, 'r_met': r_met, 'z': gen_z, 'w': gen_w, 'pi_met':np.sum(gen_z,0)/np.sum(np.sum(gen_z)),
                 'pi_bug':np.sum(gen_w,0)/np.sum(np.sum(gen_w))}
    plot_syn_data(path, x, y, gen_w, gen_z, gen_bug_locs, gen_met_locs)

    dataset = (x,y)
    net = Model(gen_met_locs, gen_bug_locs, L = gen_w.shape[1], K = gen_z.shape[1])

    kfold = KFold(n_splits = n_splits, shuffle = True)
    iterations = 10000

    train_x = x
    test_out_vec = []
    train_out_vec = []
    loss_vec = []
    test_loss = []
    fig_dict = {}
    ax_dict = {}
    fig_dict2, ax_dict2 = {},{}
    fig_dict3, ax_dict3 = {},{}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(x)):
        param_dict = {}
        net.initialize(net.seed)
        for name, parameter in net.named_parameters():
            if name not in params2learn and 'all' not in params2learn:
                print(name)
                setattr(net, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))
                parameter = nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False)
            param_dict[name] = [parameter.clone().detach().numpy()]
        criterion = nn.L1Loss()
        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        x_train, targets = x[train_ids,:], y[train_ids]
        x_test, test_targets = x[test_ids,:], y[test_ids]
        loss_vec = []
        test_loss = []
        running_loss = 0.0
        for epoch in range(iterations):
            optimizer.zero_grad()
            outputs = net(x_train)
            loss = criterion(outputs, torch.Tensor(targets))
            loss.backward()
            optimizer.step()

            loss_vec.append(loss.item())
            for name, parameter in net.named_parameters():
                if name == 'w' or name == 'z' or name == 'alpha':
                    parameter = getattr(net, name + '_act')
                elif name == 'r_bug' or name == 'r_met':
                    parameter = torch.exp(parameter)

                param_dict[name].append(parameter.clone().detach().numpy())

            if epoch%1000 == 0 and epoch != 0:
                print(epoch)
                # train_out_vec.append(outputs)
                fig_dict2[fold], ax_dict2[fold] = plt.subplots(y.shape[1], 1, figsize=(8, 4 * y.shape[1]))
                fig_dict3[fold], ax_dict3[fold] = plt.subplots(gen_z.shape[1], 1, figsize=(8, 4 * gen_z.shape[1]))
                plot_param_traces(path, param_dict, params2learn, true_vals, net, fold)
                plot_output(path, test_loss, test_out_vec, test_targets, gen_z, fig_dict2, ax_dict2, fig_dict3,
                            ax_dict3, fold)

            if epoch%100==0:
                train_out_vec.append(outputs)
                with torch.no_grad():
                    test_out = net(x_test)
                    test_out_vec.append(test_out)
                    test_loss.append(criterion(test_out, torch.Tensor(test_targets)))
                if epoch != 0:
                    fig3, ax3 = plt.subplots(figsize=(8, 4 * n_splits))
                    fig3, ax3 = plot_loss(fig3, ax3, fold, epoch+1, loss_vec, test_loss)
                    fig3.tight_layout()
                    fig3.savefig(path + 'loss_fold_' + str(fold) + '.pdf')
                    plt.close(fig3)

    #     if fold == 0:
    #         plot_param_traces(path, param_dict, params2learn, true_vals, net, fig_dict, ax_dict)
    #
    #     plot_output(path, test_loss, test_out_vec, test_targets, gen_z, fig_dict2, ax_dict2, fig_dict3, ax_dict3, fold)
    #
    #     fig3, ax3 = plot_loss(fig3, ax3, fold, iterations, loss_vec, test_loss)
    # fig3.tight_layout()
    # fig3.savefig(path + 'train_test_loss.pdf')

