from helper import *
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from plot_helper import *

def generate_synthetic_data(N_met = 10, N_bug = 14, N_samples = 200, N_met_clusters = 2, N_bug_clusters = 2, N_local_clusters=4, state = 1,
                            beta_var = 2, cluster_disparity = 100, meas_var = 0.001,
                            cluster_per_met_cluster = 1, repeat_clusters = True):
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

    if N_local_clusters <= 1:
        N_clusters_by_dist = N_bug_clusters
    else:
        N_clusters_by_dist = N_local_clusters
    choose_from = np.arange(N_bug)
    bug_gp_ids = []
    for n in range(N_clusters_by_dist-1):
        chosen = np.random.choice(choose_from, int((N_bug/N_clusters_by_dist)),replace = False)
        if repeat_clusters:
            choose_from = np.array(list(set(choose_from) - set(chosen)) + list(np.random.choice(chosen, 1)))
        else:
            choose_from = np.array(list(set(choose_from) - set(chosen)))
        bug_gp_ids.append(chosen)
    bug_gp_ids.append(choose_from)
    dist_bug = np.zeros((N_bug, N_bug))
    for i, gp in enumerate(bug_gp_ids):
        others = bug_gp_ids.copy()
        others.pop(i)
        others = np.concatenate(others)
        for met in gp:
            if met in others:
                dist_bug[met, gp] = 2.5*np.ones((1, len(gp)))
            else:
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

    mu_met = np.array([[met_locs[bg, 0].sum()/len(bg),
                        met_locs[bg, 1].sum()/len(bg)] for bg in met_gp_ids])
    z_gen = np.array([get_one_hot(kk, l=N_met) for kk in met_gp_ids]).T

    cluster_centers = np.array([[bug_locs[bg, 0].sum()/len(bg),
                        bug_locs[bg, 1].sum()/len(bg)] for bg in bug_gp_ids])
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

    betas = np.random.normal(0, np.sqrt(beta_var), size = (N_bug_clusters+1, N_met_clusters))
    alphas = st.bernoulli(0.5).rvs((N_bug_clusters, N_met_clusters))
    cluster_means = np.random.choice(
        np.arange(1, np.int(N_bug_clusters * cluster_disparity * 2) + 1, cluster_disparity), N_bug_clusters,
        replace=False)
    if N_bug_clusters==2 and N_met_clusters==2:
        betas = np.array([[-1,7],[-2.2,0.5],[0,3]])
        alphas = np.array([[1, 1], [0, 1]])
        cluster_means = [10,110]
    X = np.zeros((N_samples, N_bug))

    cluster_means = cluster_means / (np.max(cluster_means) + 10)
    if len(w_gen.shape)>2:
        temp2 = w_gen[:,:,0]
    else:
        temp2 = w_gen
    for i in range(N_bug_clusters):
        outer_ixs = np.where(temp2[:,i]==1)[0]
        mu = cluster_means[i]
        v = (mu * (1 - mu)) / meas_var
        if N_local_clusters <= 1:
            X[:, outer_ixs] = st.beta(mu * v, (1 - mu) * v).rvs(size=(N_samples, len(outer_ixs)))
        else:
            for dix in outer_ixs:
                ixs = np.where(temp[:, dix]==1)[0]
                mu = cluster_means[i]
                v = (mu * (1 - mu)) / meas_var
                X[:, ixs] = st.beta(mu * v, (1 - mu) * v).rvs(size=(N_samples, len(ixs)))
        # X[:, ixs] = st.nbinom(cluster_means[i], 0.5).rvs(size = (N_samples, len(ixs)))
    X = X/np.expand_dims(np.sum(X, 1),1)

    b = X@temp
    if N_local_clusters > 1:
        temp_w = np.repeat(w_gen[:,:,np.newaxis], N_samples, axis = 2)
        b = np.sum(temp_w*b.T,1).T

    y = np.zeros((N_samples, N_met))
    for k in range(z_gen.shape[1]):
        vals = np.zeros((N_samples, N_met))
        ix = np.where(z_gen[:,k]==1)[0]
        cluster_vals = betas[0,k] + b@(betas[1:,k]*alphas[:,k])
        vals[:, ix] = np.random.normal(cluster_vals/len(ix), 0.1, size = (len(ix), N_samples)).T
        which_sum = np.random.choice(ix,1)[0]
        vals[:, which_sum] = 0
        total_sum = np.sum(vals,1)
        vals[:, which_sum] = cluster_vals - total_sum
        y[:, ix] = vals[:, ix]

    return X, y, betas, alphas, w_gen, z_gen, bug_locs, met_locs, mu_bug, mu_met, r_bug, r_met, temp


if __name__ == "__main__":
    N_bug = 10
    N_met = 10
    K=2
    L=2
    n_local_clusters = 1
    cluster_per_met_cluster = 0
    meas_var = 0.001
    path = 'test/'
    repeat_clusters = 0

    x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
    mu_met, r_bug, r_met, gen_u = generate_synthetic_data(
        N_met = N_met, N_bug = N_bug, N_met_clusters = K, N_local_clusters = n_local_clusters, N_bug_clusters = L,
        meas_var = meas_var, cluster_per_met_cluster= cluster_per_met_cluster, repeat_clusters=repeat_clusters)

    # r_scale_met =
    bug_clusters = np.argmax(gen_w,1)
    met_clusters = np.argmax(gen_z, 1)

    div = y[:, met_clusters==1]/y[:, met_clusters==0]
    print((div.min(), div.max()))

    div = x[:, bug_clusters==1]/x[:, bug_clusters==0]
    print((div.min(), div.max()))

    plot_syn_data(path, x, y, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
                  r_bug, mu_met, r_met, gen_u)
