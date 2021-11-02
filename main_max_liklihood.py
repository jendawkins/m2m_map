import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
import torch.optim as optim
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from helper import *
from sklearn.model_selection import KFold
from scipy.special import logit

class Model(nn.Module):
    def __init__(self, met_locs, microbe_locs, temp_grouper = 10, temp_selector = 10, L = 2, K = 2, beta_var = 10, c_var = 1,
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
        self.c = nn.Parameter(torch.normal(0,self.c_var, size = (self.L, self.K), requires_grad=True))
        sampler = torch.normal(0, self.c_var, size=[100])
        self.range_dict['alpha'] = (torch.sigmoid(-self.temp_selector*sampler).min() - 0.1,
                                   torch.sigmoid(-self.temp_selector*sampler).max() + 0.1)
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
        r0_met = euc / self.K
        # r0_met = self.met_locs[np.random.choice(np.arange(self.met_locs.shape[0]), size = self.K),:]
        gamma = Gamma(self.r_scale_met/2, torch.tensor((self.r_scale_met*(r0_met)))/2)
        r_temp = gamma.sample([self.K])
        self.range_dict['r_met'] = (0, (1/gamma.sample([100])).max())
        self.r_met = nn.Parameter(1/r_temp, requires_grad = True)

        range_x = np.max(self.microbe_locs[:,0]) - np.min(self.microbe_locs[:,0])
        range_y = np.max(self.microbe_locs[:,1]) - np.min(self.microbe_locs[:,1])
        euc = np.sqrt(range_x**2 + range_y**2)
        r0_bug = euc / self.K
        gamma = Gamma(self.r_scale_bug/2, torch.tensor((self.r_scale_bug*(r0_bug))/2))
        self.range_dict['r_bug'] = (0, (1/gamma.sample([100])).max())
        r_temp = gamma.sample([self.L])
        self.r_bug = nn.Parameter(1/r_temp, requires_grad=True)

    def forward(self, x):
        # N x L
        # embedding_dim x num_bug_clusters
        # num_bug x embedding_dim
        temp_mu = self.mu_bug.unsqueeze(1).repeat(1,self.microbe_locs.shape[0],1)
        euc = ((torch.Tensor(self.microbe_locs) - temp_mu)).pow(2).sum(2).sqrt()
        w = torch.softmax(-self.temp_grouper*(euc/self.r_bug.unsqueeze(1).repeat(1,euc.shape[1])), dim = 0)
        g = torch.matmul(torch.Tensor(x), w.float().T)
        # N x K
        temp_mu = self.mu_met.unsqueeze(1).repeat(1, self.met_locs.shape[0], 1)
        euc = ((torch.Tensor(self.met_locs) - temp_mu)).pow(2).sum(2).sqrt()
        z = torch.softmax(-self.temp_grouper*(euc/self.r_met.unsqueeze(1).repeat(1,euc.shape[1])),dim = 0)
        alpha = torch.sigmoid(-self.temp_selector*self.c)
        # K
        out_clusters = self.beta[0,:] + torch.matmul(g, self.beta[1:,:]*alpha)
        out = torch.matmul(out_clusters, z.float())
        return out


# Clusters need to be not correlated to eachother - if too similar to eachother, correlation in the data
# Linear relationship is with sums - i.e. aggregation of clusters and metabolites
# All y_j's in the same cluster will have basically the same estimation
# 2 clusters metabolites, 2 clusters microbes
# Make sure betas are separate enough
# Toy example where the sums of x create y
# Main thing - do we get the clusters right?
def generate_synthetic_data(N_met = 20, N_bug = 15, N_samples = 100, N_met_clusters = 2, N_bug_clusters = 2, state = 1,
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
    params2learn = ['r_met', 'r_bug', 'mu_met', 'mu_bug', 'c', 'beta']
    path = 'results/learn_' + '_'.join(params2learn) + '/'
    if not os.path.isdir(path):
        os.mkdir(path)
    x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, kmeans_bug, kmeans_met = generate_synthetic_data()
    r_bug = [np.max([np.sqrt(np.sum((kmeans_bug.cluster_centers_[i,:] - l)**2)) for l in gen_bug_locs[gen_w[:,i]==1,:]]) for i in
             range(kmeans_bug.cluster_centers_.shape[0])]
    r_met = [np.max([np.sqrt(np.sum((kmeans_met.cluster_centers_[i,:] - l)**2)) for l in gen_met_locs[gen_z[:,i]==1,:]]) for i in
             range(kmeans_met.cluster_centers_.shape[0])]
    tv = gen_alpha.astype(float).copy()
    tv[np.where(gen_alpha<0.5)] = 1e-5
    tv[np.where(gen_alpha>0.5)] = 1 - (1e-5)
    true_vals = {'y':y, 'beta':gen_beta, 'alpha':gen_alpha, 'c': logit(1-tv), 'mu_bug': kmeans_bug.cluster_centers_, 'mu_met': kmeans_met.cluster_centers_,
                 'r_bug': r_bug, 'r_met': r_met}
    fig, ax = plt.subplots(1,2)
    fig2, ax2 = plt.subplots(2,1)
    for i in range(gen_w.shape[1]):
        ix = np.where(gen_w[:,i]==1)[0]
        ax[0].scatter(gen_bug_locs[ix,0], gen_bug_locs[ix,1], label = 'Cluster ' + str(i))
        ax[0].set_title('Microbes')
        # for ii in ix:
        ax2[0].hist(x[:,ix].flatten(), range = (x.min(), x.max()), label = 'Cluster ' + str(i), alpha = 0.5)
        ax2[0].set_xlabel('Standardized microbe values')
        ax2[0].set_ylabel('# Samples')
        ax2[0].set_title('Microbes')
    ax[0].legend()
    ax2[0].legend()
    for i in range(gen_z.shape[1]):
        ix = np.where(gen_z[:,i]==1)[0]
        ax[1].scatter(gen_met_locs[ix,0], gen_met_locs[ix,1], label = 'Cluster ' + str(i))
        ax[1].set_title('Metabolites')
        # for ii in ix:
        ax2[1].hist(y[:,ix].flatten(), range = (y.min(), y.max()), label = 'Cluster ' + str(i), alpha = 0.5)
        ax2[1].set_xlabel('Standardized metabolite values')
        ax2[1].set_ylabel('# Samples')
        ax2[1].set_title('Metabolites')
    ax[1].legend()
    ax2[1].legend()
    fig.tight_layout()
    fig.savefig(path + 'embedded_locations.pdf')
    fig2.tight_layout()
    fig2.savefig(path + 'cluster_histogram.pdf')

    dataset = (x,y)
    net = Model(gen_met_locs, gen_bug_locs, L = gen_w.shape[1], K = gen_z.shape[1])

    kfold = KFold(n_splits = 5, shuffle = True)
    iterations = 10000

    train_x = x
    test_out_vec = []
    train_out_vec = []
    loss_vec = []
    test_loss = []
    fig3, ax3 = plt.subplots(5,1, figsize = (8, 4*5))
    fig_dict = {}
    ax_dict = {}
    for name, parameter in net.named_parameters():
        if name not in params2learn:
            continue
        if name == 'c':
            name = 'alpha'
        if len(parameter.shape)==1:
            fig_dict[name], ax_dict[name] = plt.subplots(parameter.shape[0],
                                                         figsize = (5, 4*parameter.shape[0]))
        elif len(parameter.shape)==2:
            fig_dict[name], ax_dict[name] = plt.subplots(parameter.shape[0], parameter.shape[1],
                                                         figsize = (5*parameter.shape[1], 4*parameter.shape[0]))

    fig_dict2, ax_dict2 = {},{}
    fig_dict3, ax_dict3 = {},{}
    for fold in range(5):
        fig_dict2[fold], ax_dict2[fold] = plt.subplots(y.shape[1],1, figsize = (8, 4*y.shape[1]))
        fig_dict3[fold], ax_dict3[fold] = plt.subplots(gen_z.shape[1], 1, figsize=(8, 4 * gen_z.shape[1]))

    for fold, (train_ids, test_ids) in enumerate(kfold.split(x)):
        param_dict = {}
        net.initialize(net.seed)
        for name, parameter in net.named_parameters():
            if name not in params2learn:
                setattr(net, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))
                parameter = nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False)
            if name == 'c':
                param_dict['alpha'] = [torch.sigmoid(-net.temp_selector * parameter.clone().detach()).numpy()]
                param_dict[name] = [parameter.clone().detach().numpy()]
            else:
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
            if fold == 0:
                for name, parameter in net.named_parameters():
                    if name == 'c':
                        param_dict['alpha'].append(torch.sigmoid(-net.temp_selector*parameter.clone().detach()).numpy())
                        param_dict[name].append(parameter.clone().detach().numpy())
                    else:
                        param_dict[name].append(parameter.clone().detach().numpy())

            if epoch%100 == 0:
                train_out_vec.append(outputs)
                # loss_vec.append(running_loss/10)
                # running_loss = 0
                with torch.no_grad():
                    test_out = net(x_test)
                    test_out_vec.append(test_out)
                    test_loss.append(criterion(test_out, torch.Tensor(test_targets)))

        if fold == 0:
            for name, plist in param_dict.items():
                if name in params2learn or (name=='alpha' and 'c' in params2learn):
                    if name == 'c':
                        name = 'alpha'
                    for k in range(plist[0].shape[0]):
                        if len(plist[0].shape)==1:
                            trace = [p[k] for p in plist]
                            ax_dict[name][k].plot(trace, label = 'Trace')
                            ax_dict[name][k].plot([true_vals[name][k]]*len(trace), c='r', label = 'True')
                            ax_dict[name][k].set_title(name + ', ' + str(k))
                            ax_dict[name][k].set_ylim(net.range_dict[name])
                            ax_dict[name][k].legend()
                            ax_dict[name][k].set_xlabel('Iterations')
                            ax_dict[name][k].set_ylabel('Parameter Values')
                        else:
                            for j in range(plist[0].shape[1]):
                                trace = [p[k,j] for p in plist]
                                ax_dict[name][k,j].plot(trace, label = 'Trace')
                                ax_dict[name][k, j].plot([true_vals[name][k, j]] * len(trace), c='r', label = 'True')
                                ax_dict[name][k,j].set_title(name + ', ' + str(k) + ', ' + str(j))
                                ax_dict[name][k,j].set_ylim(net.range_dict[name])
                                ax_dict[name][k, j].legend()
                                ax_dict[name][k,j].set_xlabel('Iterations')
                                ax_dict[name][k,j].set_ylabel('Parameter Values')
                    fig_dict[name].tight_layout()
                    fig_dict[name].savefig(path + name + '_parameter_trace.pdf')

        best_mod = np.argmin(test_loss)
        preds = test_out_vec[best_mod]
        total = np.concatenate([test_targets, preds])
        for met in range(test_targets.shape[1]):
            cluster_id = np.where(gen_z[met,:]==1)[0]
            ax_dict2[fold][met].hist(preds[:,met].detach().numpy(), range = (total.min(), total.max()), label = 'guess', alpha = 0.5)
            ax_dict2[fold][met].hist(test_targets[:, met], range=(total.min(), total.max()), label='true', alpha = 0.5)
            ax_dict2[fold][met].set_title('Metabolite ' + str(met) + ', Cluster ' + str(cluster_id))
            ax_dict2[fold][met].legend()
            ax_dict2[fold][met].set_xlabel('Metabolite values')
            ax_dict2[fold][met].set_ylabel('# Samples')
        fig_dict2[fold].tight_layout()
        fig_dict2[fold].savefig(path + 'fold' + str(fold) + '_output_histograms.pdf')

        for cluster in range(gen_z.shape[1]):
            met_ids = np.where(gen_z[:,cluster]==1)[0]
            ax_dict3[fold][cluster].hist(preds[:,met_ids].flatten().detach().numpy(), range = (total.min(), total.max()), label = 'guess', alpha = 0.5)
            ax_dict3[fold][cluster].hist(test_targets[:, met_ids].flatten(), range=(total.min(), total.max()), label='true', alpha = 0.5)
            ax_dict3[fold][cluster].set_title('Cluster ' + str(cluster))
            ax_dict3[fold][cluster].legend()
            ax_dict3[fold][cluster].set_xlabel('Metabolite values')
            ax_dict3[fold][cluster].set_ylabel('# Samples')
        fig_dict3[fold].tight_layout()
        fig_dict3[fold].savefig(path + 'fold' + str(fold) + '_output_cluster_histograms.pdf')

        ax3[fold].set_title('Fold ' + str(fold))
        ax3[fold].plot(np.arange(iterations), loss_vec, label = 'training loss')
        ax3[fold].plot(np.arange(0,iterations,100), test_loss, label = 'test loss')
        ax3[fold].legend()
    fig3.tight_layout()
    fig3.savefig(path + 'train_test_loss.pdf')

