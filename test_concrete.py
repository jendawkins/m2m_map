from main_MAP import *

alphas = np.logspace(0,-5, 100)
taus = np.logspace(-2, -10, 10)
for tau in taus:
    alphas = np.logspace(0, np.log10(tau), 100)
    BinCon = BinaryConcrete(1,tau)
    samp_alphas = BinCon.sample(size = [100])
    fig, ax = plt.subplots()
    ax.hist(samp_alphas)
    ax.set_xscale('log')
    ax.set_title('tau = ' + str(tau))
    fig.savefig('alpha_hist/tau' + str(tau).replace('.','d') + '.pdf')
    plt.close(fig)
    probs = []
    # fig, ax = plt.subplots()
    # for alpha in alphas:
    #     probs.append(BinCon.pdf(alpha).item())
    # ax.semilogx(alphas, probs, '*-', ms = 10)
    # ax.set_title('tau = ' + str(tau))
    # ax.set_xlabel('alpha')
    # ax.set_ylabel('probs')
    # if tau == 1:
    #     ax.set_ylim(-1e-10,1.01)
    # fig.savefig('alpha_probs/tau' + str(tau).replace('.','d') + '.pdf')
    # plt.close(fig)

x, y, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, kmeans_bug, kmeans_met = generate_synthetic_data()
r_bug = [
    np.max([np.sqrt(np.sum((kmeans_bug.cluster_centers_[i, :] - l) ** 2)) for l in gen_bug_locs[gen_w[:, i] == 1, :]])
    for i in
    range(kmeans_bug.cluster_centers_.shape[0])]
r_met = [
    np.max([np.sqrt(np.sum((kmeans_met.cluster_centers_[i, :] - l) ** 2)) for l in gen_met_locs[gen_z[:, i] == 1, :]])
    for i in
    range(kmeans_met.cluster_centers_.shape[0])]
true_vals = {'y': y, 'beta': gen_beta, 'alpha': gen_alpha, 'mu_bug': kmeans_bug.cluster_centers_,
             'mu_met': kmeans_met.cluster_centers_,
             'r_bug': r_bug, 'r_met': r_met, 'z': gen_z, 'w': gen_w,
             'pi_met': np.expand_dims(np.sum(gen_z, 0) / np.sum(np.sum(gen_z)), 0),
             'pi_bug': np.expand_dims(np.sum(gen_w, 0) / np.sum(np.sum(gen_w)), 0)}
net = Model(gen_met_locs, gen_bug_locs)
net2 = Model(gen_met_locs, gen_bug_locs)
net.initialize(0)
net2.initialize(0)
optimizer = optim.RMSprop(net.parameters(),lr=0.001)
mu_met_vals = []
r_met_vals = []
z_vals = []
z_vals2 = []
loss_vec = []
loss2_vec = []
best_z = []
for epoch in range(1000):
    net2.z[:, 0] = net.z[:, 1].clone()
    net2.z[:, 1] = net.z[:, 0].clone()
    if epoch%10==0:
        mu_met_vals.append(net.mu_met.detach().numpy())
        r_met_vals.append(net.r_met.detach().numpy())
        z_vals.append(net.z_act.detach().numpy())
        z_vals2.append(net2.z_act.detach().numpy())
    optimizer.zero_grad()
    cluster_outputs, outputs = net(x)
    criterion = MAPloss(net, meas_var = 4)
    criterion2 = MAPloss(net2, meas_var=4)
    loss = criterion.compute_loss(cluster_outputs, torch.Tensor(y))
    loss2 = criterion2.compute_loss(cluster_outputs, torch.Tensor(y))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        loss_vec.append(loss)
        loss2_vec.append(loss2)
    if loss > loss2:
        best_z.append(net2.z_act)
    else:
        best_z.append(net.z_act)

color_dict = {0: 'tab:blue', 1: 'tab:orange'}
for i in np.arange(len(mu_met_vals)):
    fig, ax = plt.subplots()
    z_guess = torch.argmax(best_z[i],1)
    for cluster in range(mu_met_vals[i].shape[1]):
        ax.scatter(mu_met_vals[i][cluster, 0], mu_met_vals[i][cluster, 1], marker='*', c = color_dict[cluster])
        circle2 = plt.Circle((mu_met_vals[i][cluster, 0], mu_met_vals[i][cluster, 1]),
                             r_met_vals[i][cluster],
                             alpha=0.2, color=color_dict[cluster], label='Cluster ' + str(cluster))

        ax.add_patch(circle2)
        ixs = np.where(z_guess==cluster)[0]
        ax.scatter(gen_met_locs[ixs, 0], gen_met_locs[ixs, 1], c = color_dict[cluster])
    fig.savefig('test/epoch' + str(i) + '.pdf')
    plt.close(fig)


z_dist = Concrete(net.pi_met, 1)
loc_dist = [MultivariateNormal(torch.Tensor(true_vals['mu_met'][l,:]),
                               torch.Tensor([true_vals['r_met'][l]])*torch.eye(net.mu_bug.shape[1]))
            for l in range(net.mu_met.shape[0])]
md = MultDist(z_dist, loc_dist)
var = 4
ax = var*(np.random.randn(50))
ay = var*(np.random.randn(50))
locs = np.stack((ax, ay)).T
path = 'random_locs'
if not os.path.isdir(path):
    os.mkdir(path)

md.make_sampling_table(torch.Tensor(locs))

loc, z = md.prob_table.index.values, md.prob_table.columns.values
fig, ax = plt.subplots()

for xix in range(len(md.prob_table.index.values)):
    probs2 = md.prob_table.iloc[xix, :]
    probs2[probs2>1] = 0
    highest_prob = np.max(md.prob_table.iloc[xix,:])
    zix = np.argmax(md.prob_table.iloc[xix,:])
    z_guess = torch.argmax(torch.stack(md.prob_table.columns.values[zix]))
    print(z_guess)
    if z_guess == 0:
        c = 'tab:blue'
    else:
        c = 'tab:orange'
    prob = md.prob_table.iloc[xix, zix]
    if prob > 1:
        prob = 0
    if prob<0:
        prob = 0
    if np.isnan(prob):
        prob = 0
    ax.scatter(loc[xix][0], loc[xix][1], s = 10, c = c)
    ax.text(loc[xix][0], loc[xix][1], 'prob=' + str(np.round(prob,3)))
    ax.set_xlabel('d1')
    ax.set_ylabel('d2')
    fig2, ax2 = plt.subplots()
    probs = md.prob_table.iloc[xix, :]
    probs[probs>1] = 1
    zvals0 = [zz[0].item() for zz in md.prob_table.columns.values]
    ax2.scatter(zvals0, probs, label = 'P(z=0)')
    zvals1 = [zz[1].item() for zz in md.prob_table.columns.values]
    ax2.scatter(zvals1, probs, label = 'P(z=1)')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.set_xlabel('Value of z[0]')
    ax2.set_ylabel('Probability')
    ax2.set_ylim([-0.1,1.1])
    fig2.savefig(path + '/pdf' + str(np.round(loc[xix][0].item(),3)).replace('.','d') + '_'
                 + str(np.round(loc[xix][1].item(),3)).replace('.','d') + '.pdf')

    print(xix)
    print('')
    plt.close(fig2)

for cluster in range(net.mu_met.shape[0]):
    ax.scatter(true_vals['mu_met'][cluster,0], true_vals['mu_met'][cluster,1],marker='*')
    ax.text(true_vals['mu_met'][cluster,0], true_vals['mu_met'][cluster,1],'Cluster ' + str(cluster))
    circle2 = plt.Circle((true_vals['mu_met'][cluster,0], true_vals['mu_met'][cluster,1]),
                         true_vals['r_met'][cluster],
                         alpha=0.2, color='k', label='Cluster ' + str(cluster))

    ax.add_patch(circle2)

fig.savefig(path + '/locs.pdf')
plt.close(fig)