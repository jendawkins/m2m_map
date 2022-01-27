from helper import *
from sklearn.manifold import MDS
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.uniform import Uniform
import torch.optim as optim
from concrete import *
import pandas as pd

def plot_output_locations(net, best_z, best_mu, best_r, path='cluster_results/'):
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(best_z.shape[1]):
        ix = np.where(best_z[:, i] > 0.5)[0]
        p2 = ax.scatter(net.met_locs[ix, 0], net.met_locs[ix, 1])
        ax.scatter(best_mu[i, 0], best_mu[i, 1], marker='*', color=p2.get_facecolor().squeeze())
        ax.text(best_mu[i, 0], best_mu[i, 1], 'predicted\ncluster ' + str(i) + ' mean')
        ax.set_title('Metabolites')
        circle2 = plt.Circle((best_mu[i, 0], best_mu[i, 1]), best_r[i],
                             alpha=0.2, color=p2.get_facecolor().squeeze(), label='Cluster ' + str(i))
        ax.add_patch(circle2)
        ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(path + 'predicted_metab_clusters.pdf')
    plt.close(fig)

def plot_loss(loss_vec, iterations, path):
    fig3, ax3 = plt.subplots()
    ax3.set_title('Loss')
    ax3.plot(np.arange(iterations), loss_vec, label='training loss')
    ax3.set_xlabel('iterations')
    ax3.set_ylabel(loss)
    fig3.tight_layout()
    fig3.savefig(path + 'loss.pdf')
    plt.close(fig3)

def generate_synthetic_data(N_met = 10, N_met_clusters = 2, state = 1):
    np.random.seed(state)
    choose_from = np.arange(N_met)
    met_gp_ids = []
    for n in range(N_met_clusters-1):
        # num_choose = np.random.choice(np.arange(2,len(choose_from)-(N_met_clusters-n)),1)
        chosen = np.random.choice(choose_from, int(N_met/N_met_clusters),replace = False)
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

    embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=state)
    met_locs = embedding.fit_transform(dist_met)

    mu_met = np.array([[met_locs[bg, 0].sum()/len(bg),
                        met_locs[bg, 1].sum()/len(bg)] for bg in met_gp_ids])
    z_gen = np.array([get_one_hot(kk, l=N_met) for kk in met_gp_ids]).T

    r_met = np.array([np.max([np.sqrt(np.sum((mu_met[i,:] - l)**2)) for l in met_locs[z_gen[:,i]==1,:]]) for i in
             range(mu_met.shape[0])])
    r_met = np.append(r_met, np.zeros(N_met-1-len(r_met)))
    gen_z = np.hstack((z_gen, np.zeros((N_met, N_met - 1 - N_met_clusters))))
    mu_met = np.vstack((mu_met, np.zeros((N_met - N_met_clusters - 1, mu_met.shape[1]))))
    pi_met = np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0)
    return met_locs, mu_met, r_met, gen_z, pi_met

class Clusterer(nn.Module):
    def __init__(self, met_locs, K, K_true, tau, learn_n_clusters = True, learn = ['r','mu','z','pi','b','lambda','C']):
        super(Clusterer, self).__init__()
        self.K = K
        self.K_true = K_true
        self.N = met_locs.shape[0]
        self.D = met_locs.shape[1]
        self.met_locs = met_locs
        self.tau = tau
        self.learn_n_clusters = learn_n_clusters
        self.learn = learn

    def initialize(self):
        ix = np.random.choice(np.arange(len(self.met_locs)), self.K, replace = False)
        self.mu = nn.Parameter(torch.Tensor(self.met_locs[ix,:]), requires_grad=True)

        R = np.array(
            [np.max(self.met_locs[:, d]) - np.min(self.met_locs[:, d]) for
             d in np.arange(self.met_locs.shape[1])])
        self.Ro = torch.diag(torch.Tensor(R))
        r_scale_met = np.sqrt(np.sum(R ** 2)) / (self.K_true * 2)

        self.pi = nn.Parameter(Dirichlet((1/self.K)*torch.ones(self.K)).sample())

        if self.learn_n_clusters:
            self.lambda_mu = nn.Parameter(torch.log(Gamma(0.5, 0.5).sample([self.D])), requires_grad=True)
            Lambda = torch.diag(torch.sqrt(torch.exp(self.lambda_mu)))
            Bo = Lambda@self.Ro@Lambda
            self.b = nn.Parameter(
                MultivariateNormal(torch.zeros(self.D), torch.eye(self.D)).sample([self.K]), requires_grad=True)
            self.C =  nn.Parameter(torch.Tensor(np.log(r_scale_met)*np.ones(self.K)), requires_grad=True)
            self.c = 1.25 + (self.D - 1) / 4
            self.g = 0.25 + (self.D - 1)/4
            self.G = self.c/(50*self.g) * np.sqrt(np.sum(R**2))


        self.mu = nn.Parameter(MultivariateNormal(torch.zeros(self.D), self.Ro).sample([self.K]), requires_grad=True)
        self.r = nn.Parameter(torch.Tensor(np.log(r_scale_met)*np.ones(self.K)), requires_grad=True)
        self.z = nn.Parameter(Normal(0, 1).sample([self.N, self.K]), requires_grad=True)


    def z_loss(self):
        z_act = torch.softmax(self.z / 1, 1)
        mvn = [MultivariateNormal(self.mu[k,:], (torch.eye(self.mu.shape[1]) *
                                                           torch.exp(self.r[k])).float()) for k in np.arange(self.mu.shape[0])]
        con = Concrete(torch.softmax(self.pi,0), self.tau)
        multi = MultDist(con, mvn)
        log_probs = torch.stack([-torch.log(multi.pdf(z_act[m,:], torch.Tensor(self.met_locs[m,:])))
                                 for m in np.arange(self.met_locs.shape[0])]).sum()
        return log_probs

    def pi_loss(self):
        log_probs = -(torch.Tensor(np.array([1/self.K]*self.K)-1) * torch.log(torch.softmax(self.pi,0))).sum()
        return log_probs

    def mu_loss(self):
        Lambda = torch.diag(self.lambda_mu)
        Bo = Lambda@self.Ro@Lambda
        mvn = MultivariateNormal(self.b, Bo)
        log_probs = -mvn.log_prob(self.mu).sum()
        return log_probs

    def lambda_loss(self):
        gamma = Gamma(0.5, 0.5)
        log_probs = -gamma.log_prob(torch.exp(self.lambda_mu)).sum()
        return log_probs

    def b_loss(self):
        mvn = MultivariateNormal(torch.zeros(self.D), torch.eye(self.D, self.D))
        return -mvn.log_prob(self.b).sum()

    def r_loss(self):
        gamma = Gamma(self.c, self.C)
        log_probs = -gamma.log_prob(1/torch.exp(self.r)).sum()
        return log_probs

    def C_loss(self):
        gamma = Gamma(self.g, self.G)
        log_probs = -gamma.log_prob(1/torch.exp(self.C)).sum()
        return log_probs

    def forward(self):
        total_loss = 0
        for param in self.learn:
            fun = getattr(self, param + '_loss')
            log_probs = fun()
            total_loss += log_probs
        return total_loss


if __name__=="__main__":
    torch.autograd.set_detect_anomaly(True)

    N_met = 20
    N_met_clusters = 3
    seed = 0
    iterations = 10000
    lr = 0.01
    path = 'cluster_results/'
    if not os.path.isdir(path):
        os.mkdir(path)

    locs, mu, r, z, pi = generate_synthetic_data(N_met, N_met_clusters, state = seed)
    tau_vec = np.logspace(-0.5, -4, iterations)

    cluster_net = Clusterer(locs, N_met, N_met_clusters, tau_vec[0])
    cluster_net.initialize()
    if not os.path.isdir(path+ 'synthetic_data/'):
        os.mkdir(path+ 'synthetic_data/')
    plot_output_locations(cluster_net, z, mu, r, path = path + 'synthetic_data/')
    optimizer = optim.RMSprop(cluster_net.parameters(), lr=lr)

    param_dict = {}
    for name, parameter in cluster_net.named_parameters():
        if name == 'z':
            parameter = torch.softmax(parameter / 0.1, 1)
        elif name == 'r':
            parameter = torch.exp(parameter)
        elif name == 'pi':
            parameter = torch.softmax(parameter,0)
        param_dict[name] = [parameter.clone().detach().numpy()]

    loss_vec = []
    for epoch in range(iterations):
        if epoch > 100:
            ix = int(epoch - 100)
            cluster_net.tau = tau_vec[ix]

        optimizer.zero_grad()
        loss = cluster_net()
        loss_vec.append(loss.item())
        loss.backward()
        optimizer.step()

        for name, parameter in cluster_net.named_parameters():
            if name == 'z':
                parameter = torch.softmax(parameter / 0.1, 1)
            elif name == 'r':
                parameter = torch.exp(parameter)
            elif name == 'pi':
                parameter = torch.softmax(parameter,0)
            param_dict[name].append(parameter.clone().detach().numpy())

        if epoch%100==0 and epoch > 0:
            print(epoch)
            path = path + str(epoch) + '/'
            if not os.path.isdir(path):
                os.mkdir(path)
            mapping = unmix_clusters(mu, param_dict['mu'][-1], param_dict['r'][-1],locs)
            mapping = pd.Series(mapping).sort_index()
            best_z = param_dict['z'][-1][:, mapping]
            best_mu = param_dict['mu'][-1][mapping, :]
            best_r = param_dict['r'][-1][mapping]
            plot_output_locations(cluster_net, best_z, best_mu, best_r, path)
            plot_loss(loss_vec, epoch+1, path)

