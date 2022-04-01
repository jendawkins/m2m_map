from helper import *
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from plot_helper import *
import datetime


def generate_synthetic_data(N_met = 10, N_bug = 14, N_samples = 200, N_met_clusters = 2, N_bug_clusters = 2,
                            N_local_clusters=1, state = 3,
                            beta_var = 2, cluster_disparity = 100, meas_var = 0.001,
                            cluster_per_met_cluster = 0, repeat_clusters = 1,embedding_dim = 2,
                            deterministic = True, linear = False, nl_type = "linear",dist_var_perc = 0.9,
                            overlap_frac = 0.5):
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

    # rand = np.random.randint(0,2, size = dist_met.shape)
    # rand = (rand + rand.T)/2
    dist_met[dist_met == 0] = 10
    dist_met = dist_met + 10*dist_var_perc
    np.fill_diagonal(dist_met, 0)

    if N_local_clusters <= 1:
        N_clusters_by_dist = N_bug_clusters
    else:
        N_clusters_by_dist = N_local_clusters
    choose_from = np.arange(N_bug)
    bug_gp_ids = []
    for n in range(N_clusters_by_dist-1):
        chosen = np.random.choice(choose_from, int((N_bug/N_clusters_by_dist)),replace = False)
        if repeat_clusters==1:
            choose_from = np.array(list(set(choose_from) - set(chosen)) + list(np.random.choice(chosen, np.int(len(chosen)*overlap_frac))))
        else:
            choose_from = np.array(list(set(choose_from) - set(chosen)))
        bug_gp_ids.append(chosen)

    if repeat_clusters==2:
        bug_gp_ids.append(np.concatenate((choose_from, bug_gp_ids[-1])))
    else:
        bug_gp_ids.append(choose_from)
    dist_bug = np.zeros((N_bug, N_bug))
    for i, gp in enumerate(bug_gp_ids):
        others = bug_gp_ids.copy()
        others.pop(i)
        others = np.concatenate(others)
        for met in gp:
            dist_bug[met, gp] = 2*np.ones((1, len(gp)))
            dist_bug[gp, met] = dist_bug[met,gp]
    # rand = np.random.randint(0,2, size = dist_bug.shape)
    # rand = (rand + rand.T)/2
    # dist_bug[dist_bug == 0] = 7
    dist_bug[dist_bug == 0] = 10
    dist_bug = dist_bug + 10*dist_var_perc
    # dist_bug = dist_bug + rand
    np.fill_diagonal(dist_bug, 0)

    embedding = MDS(n_components=embedding_dim, dissimilarity='precomputed', random_state=state)
    met_locs = embedding.fit_transform(dist_met)
    embedding = MDS(n_components=embedding_dim, dissimilarity='precomputed', random_state=state)
    bug_locs = embedding.fit_transform(dist_bug)

    mu_met = np.array([[met_locs[bg, i].sum()/len(bg) for i in np.arange(met_locs.shape[1])] for bg in met_gp_ids])
    z_gen = np.array([get_one_hot(kk, l=N_met) for kk in met_gp_ids]).T

    cluster_centers = np.array([[bug_locs[bg, i].sum()/len(bg) for i in np.arange(bug_locs.shape[1])] for bg in bug_gp_ids])
    if N_local_clusters<=1:
        # w_gen = Num bugs x num bug clusters
        w_gen = np.array([get_one_hot(kk, l=N_bug) for kk in bug_gp_ids]).T
        temp = w_gen
        mu_bug = cluster_centers
        r_bug = np.array([np.max(
            [np.sqrt(np.sum((cluster_centers[i, :] - l) ** 2)) for l in bug_locs[temp[:, i] == 1, :]]) for i
                          in
                          range(cluster_centers.shape[0])])
        if cluster_per_met_cluster:
            # w_gen = Num bugs x Num_bug_clusters x Num_met_clusters
            w_gen = np.stack([w_gen for i in range(N_met_clusters)], axis = -1)
            temp = w_gen[:,:,0]
            mu_bug = np.repeat(mu_bug[:,:,np.newaxis], N_met_clusters, axis = -1)
            r_bug = np.repeat(r_bug[:, np.newaxis], N_met_clusters, axis = -1)
        u = None
    else:
        # w_gen = Num_clusters x num_local_clusters
        w_gen = np.array([get_one_hot(np.random.choice(range(N_bug_clusters),1), N_bug_clusters) for i in range(N_local_clusters)]).T
        u = np.array([get_one_hot(kk, l = N_bug) for kk in bug_gp_ids]).T
        # u_ = num_bugs x num_local_clusters
        temp = u
        mu_bug = np.repeat(cluster_centers[np.newaxis,:,:], N_bug_clusters, axis = 0)
        r_bug = np.array([np.max(
            [np.sqrt(np.sum((cluster_centers[i, :] - l) ** 2)) for l in bug_locs[temp[:, i] == 1, :]]) for i
                          in
                          range(cluster_centers.shape[0])])
        r_bug = np.repeat(r_bug[np.newaxis, :], N_bug_clusters, axis=0)

    r_met = np.array([np.max([np.sqrt(np.sum((mu_met[i,:] - l)**2)) for l in met_locs[z_gen[:,i]==1,:]]) for i in
             range(mu_met.shape[0])])

    if not deterministic:
        betas = np.random.normal(0, np.sqrt(beta_var), size = (N_bug_clusters+1, N_met_clusters))
        alphas = st.bernoulli(0.5).rvs((N_bug_clusters, N_met_clusters))
        cluster_starts = np.arange(1, np.int(N_bug_clusters * cluster_disparity) + 1, cluster_disparity)
        cluster_ends = cluster_starts[1:] - cluster_disparity/10
        cluster_ends = np.append(cluster_ends, cluster_starts[-1] + cluster_disparity - cluster_disparity/10)
    else:
        betas = np.zeros((11,10))
        betas[0,:] = [-1,1,-1.3,1.3,-0.4,0.4,-1.7,1.7,-0.6,0.6]
        vals = [-5.4,4.1,-4.8,6.7,-4.5,3.9,-3.4,0.8,-5.9,4.9]
        betas[1:,:] = np.diag(vals)
        betas = betas[:N_bug_clusters+1, :N_met_clusters]
        # betas[-1,0] = 0
        alphas = np.ones((N_bug_clusters, N_met_clusters))
        alphas[-1,0] = -1
        # cluster_starts = np.arange(200,200+(100*10) + 20, 200)
        # cluster_ends = np.arange(300,300+(100*10) + 20, 200)
        cluster_starts = [100,350,510,650,870,1000,1200,1400,1600,1800]
        cluster_ends = [250,410,550,770,900,1100,1300,1500,1700,1900]

    X = np.zeros((N_samples, N_bug))
    if len(w_gen.shape)>2:
        temp2 = w_gen[:,:,0]
    else:
        temp2 = w_gen
    g = np.zeros((N_samples, N_bug_clusters))
    for i in range(N_bug_clusters):
        g[:,i] = st.uniform(cluster_starts[i], cluster_ends[i]-cluster_starts[i]).rvs(size = N_samples)
        outer_ixs = np.where(temp2[:,i]==1)[0]
        conc = np.repeat(g[:, i:i+1], len(outer_ixs), axis = 1) / len(outer_ixs)
        p = [st.dirichlet(conc[n,:]).rvs() for n in range(conc.shape[0])]
        if N_local_clusters <= 1:
            X[:, outer_ixs] = np.stack([st.multinomial(int(np.round(g[n,i])), p[n].squeeze()).rvs() for n in range(len(p))]).squeeze()
        else:
            for dix in outer_ixs:
                ixs = np.where(temp[:, dix]==1)[0]
                conc = np.repeat(g[:, i:i + 1], len(ixs), axis=1) / len(ixs)
                p = [st.dirichlet(conc[n, :]).rvs() for n in range(conc.shape[0])]
                X[:, ixs] = np.stack([st.multinomial(int(np.round(g[n,i])), p[n].squeeze()).rvs() for n in range(len(p))]).squeeze()

    if N_local_clusters > 1:
        temp_w = np.repeat(w_gen[:,:,np.newaxis], N_samples, axis = 2)
        g = np.sum(temp_w*x.T,1).T

    y = np.zeros((N_samples, N_met))
    for j in range(N_met):
        k = np.argmax(z_gen[j,:])
        if not linear:
            g = (g - np.mean(g,0))/np.std(g-np.mean(g,0))
            if nl_type == 'exp':
                y[:, j] = np.random.normal(betas[0, k] + np.exp(g) @ (betas[1:, k] * alphas[:, k]), meas_var)
            if nl_type == 'sigmoid':
                y[:, j] = np.random.normal(betas[0, k] + sigmoid(g) @ (betas[1:, k] * alphas[:, k]), meas_var)
            if nl_type == 'sin':
                y[:, j] = np.random.normal(betas[0, k] + np.sin(g) @ (betas[1:, k] * alphas[:, k]), meas_var)
            if nl_type == 'poly':
                y[:,j] = np.random.normal(betas[0, k] + (g)**5 @ (betas[1:, k] * alphas[:, k]) - (g)**4 @ (betas[1:, k] * alphas[:, k]), meas_var)
            if nl_type == 'linear':
                y[:, j] = np.random.normal(betas[0, k] + g @ (betas[1:, k] * alphas[:, k]), meas_var)
        else:
            y[:, j] = np.random.normal(betas[0, k] + g @ (betas[1:, k] * alphas[:, k]), meas_var)

    y = (y - np.mean(y, 0)) / np.std((y - np.mean(y)), 0)
    y_per_clust = np.vstack([y[:,z_gen[:,i]==1].mean(1) for i in np.arange(N_met_clusters)]).T
    g_new = np.hstack((np.ones((g.shape[0], 1)), g))
    betas = np.linalg.inv(g_new.T@g_new)@(g_new.T@(y_per_clust))
    return X, y, betas, alphas, w_gen, z_gen, bug_locs, met_locs, mu_bug, mu_met, r_bug, r_met, temp


if __name__ == "__main__":
    N_bug = 20
    N_met = 20
    K=3
    L=3
    dist_var_perc = 0.001
    n_local_clusters = 1
    cluster_per_met_cluster = 0
    meas_var = 0.1
    # path = datetime.date.today().strftime('%m %d %Y').replace(' ','-') + '/'
    orig_path = 'data_gen/'
    if not os.path.isdir(orig_path):
        os.mkdir(orig_path)
    repeat_clusters = 0

    x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
    mu_met, r_bug, r_met, gen_u = generate_synthetic_data(
        N_met=N_met, N_bug=N_bug, N_met_clusters=K,
        N_bug_clusters=L, meas_var=meas_var,
        repeat_clusters=2, N_samples=100, linear=1,
        nl_type='linear', dist_var_perc=0.25)

    # y = ((y.T - np.mean(y,1))/np.std(y.T - np.mean(y,1), 0)).T

    # y = (y - np.mean(y, 0)) / np.std((y - np.mean(y)), 0)
    plot_syn_data(orig_path, x, y, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
                  r_bug, mu_met, r_met, gen_u)
    #
    # for meas_var in [0.001,1,10,100]:
    #     x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
    #     mu_met, r_bug, r_met, gen_u = generate_synthetic_data(
    #         N_met = N_met, N_bug = N_bug, N_met_clusters = K,
    #         N_bug_clusters = L,meas_var = meas_var,
    #         repeat_clusters= 0, N_samples=100, linear = 1,
    #         nl_type = 'linear', dist_var_perc=10)
    #
    #     # y = ((y.T - np.mean(y,1))/np.std(y.T - np.mean(y,1), 0)).T
    #
    #     y = (y-np.mean(y,0))/np.std((y-np.mean(y)),0)
    #     path = orig_path + str(int(meas_var)) + '/'
    #     if not os.path.isdir(path):
    #         os.mkdir(path)
    #     plot_syn_data(path, x, y, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
    #                   r_bug, mu_met, r_met, gen_u)


    # for type in ['poly','sin','exp','sigmoid']:
    #     x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
    #     mu_met, r_bug, r_met, gen_u = generate_synthetic_data(
    #         N_met=N_met, N_bug=N_bug, N_met_clusters=K, N_local_clusters=n_local_clusters, N_bug_clusters=L,
    #         meas_var=meas_var, repeat_clusters=False,
    #         N_samples=1000,deterministic = True, linear = False, nl_type = type)
    #
    #     fig, ax = plt.subplots(K, L, figsize=(8 * L, 8 * K))
    #     # ranges = [[np.max(microbe_sum[:,i]/out[:,j]) - np.min(microbe_sum[:,i]/out[:,j]) for i in range(out.shape[1])] for j in range(out.shape[1])]
    #     # ixs = [np.argmin(r) for r in ranges]
    #     g = x@gen_w
    #     for i in range(K):
    #         ixs = np.where(gen_z[:,i]==1)[0]
    #         for j in range(L):
    #             # ax[i].scatter(microbe_sum[:,ixs[i]], out[:,i])
    #             for ii in ixs:
    #                 ax[i, j].scatter(g[:, j], y[:, ii])
    #             ax[i, j].set_xlabel('Microbe sum')
    #             ax[i, j].set_ylabel(r'$y_{i}$ when $i=$' + str(i))
    #             ax[i, j].set_title('Metabolite Cluster ' + str(i) + ' vs Microbe Cluster ' + str(j))
    #     fig.tight_layout()
    #     fig.savefig(orig_path + type +  '-sum_x_v_y.pdf')
    #     plt.close(fig)