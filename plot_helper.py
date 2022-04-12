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
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from draw_layers import *
from matplotlib.pyplot import cm

def plot_syn_data(path, x, y, gen_z, gen_bug_locs, gen_met_locs,
                  mu_bug, r_bug, mu_met, r_met, gen_u):
    bug_clusters = [np.where(gen_u[:,i])[0] for i in np.arange(gen_u.shape[1])]
    for ii,clust in enumerate(bug_clusters):
        if not os.path.isfile(path + 'microbe_clusters.txt'):
            with open(path + 'microbe_clusters.txt', 'w') as f:
                f.writelines('Cluster ' + str(ii) + ': ' + str(clust) + '\n')
        else:
            with open(path + 'microbe_clusters.txt', 'a') as f:
                f.writelines('Cluster ' + str(ii) + ': ' + str(clust) + '\n')

    fig2, ax2 = plt.subplots(3, 1)
    if mu_bug.shape[1]==2:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        p1 = ax[0].scatter(gen_bug_locs[:, 0], gen_bug_locs[:, 1], color = 'k', alpha = 0.5)
    if len(mu_bug.shape)>2:
        mu_bug = mu_bug[0,:,:]
        r_bug = r_bug[0,:]
    for i in range(mu_bug.shape[0]):
        if mu_bug.shape[1] == 2:
            p1 = ax[0].scatter(mu_bug[i,0], mu_bug[i,1], marker='*')
            circle1 = plt.Circle((mu_bug[i,0], mu_bug[i,1]), r_bug[i],
                                 alpha=0.2, label='Cluster ' + str(i),color=p1.get_facecolor().squeeze())
            # for ii in ix:
            #     ax[0].text(gen_bug_locs[ii,0], gen_bug_locs[ii,1], 'Bug ' + str(ii))
            ax[0].add_patch(circle1)
            ax[0].set_title('Microbes')
            ax[0].text(mu_bug[i,0], mu_bug[i,1], 'Cluster ' + str(i))
        # for ii in ix:
        bins = int((x.max() - x.min()) / 5)
        if bins<=10:
            bins = 10
        # dixs = np.where(gen_w[i, :] == 1)[0]
        # for dix in dixs:
        ix = np.where(gen_u[:, i]==1)[0]
        ax2[0].hist(x[:, ix].flatten(), range=(x.min(), x.max()), label='Cluster ' + str(i), alpha=0.5, bins = bins)
    ax2[0].set_xlabel('Microbial relative abundances')
    ax2[0].set_ylabel('# Microbes in Cluster x\n# Samples Per Microbe', fontsize = 10)
    ax2[0].set_title('Microbes')
    if mu_bug.shape[1] == 2:
        ax[0].set_aspect('equal')

    b = x@gen_u
    bins = int((b.max() - b.min()) / 5)
    if bins <= 10:
        bins = 10
    for k in range(b.shape[1]):
        ax2[1].hist(b[:,k].flatten(), range = (b.min(), b.max()), label = 'Cluster ' + str(k), alpha = 0.5, bins = bins)
    ax2[1].set_title('Histogram of microbe cluster sums')
    ax2[1].legend(loc = 'upper right')
    # ax3[0].set_aspect('equal')

    ax2[0].legend(loc = 'upper right')
    ax2[1].legend(loc = 'upper right')
    for i in range(gen_z.shape[1]):
        ix = np.where(gen_z[:, i] == 1)[0]
        if mu_bug.shape[1] == 2:
            p2 = ax[1].scatter(gen_met_locs[ix, 0], gen_met_locs[ix, 1])
            ax[1].scatter(mu_met[i, 0], mu_met[i, 1], marker='*', color=p2.get_facecolor().squeeze())
            ax[1].set_title('Metabolites')
            ax[1].text(mu_met[i, 0], mu_met[i, 1], 'Cluster ' + str(i))
            # p2 = None
            # for ii in ix:
                # p2 = ax[1].scatter(gen_met_locs[ii, 0], gen_met_locs[ii, 1], alpha = )
                # ax[1].text(gen_met_locs[ii,0], gen_met_locs[ii,1], 'Metabolite ' + str(ii))
            circle2 = plt.Circle((mu_met[i,0], mu_met[i,1]), r_met[i],
                                 alpha=0.2, color=p2.get_facecolor().squeeze(), label = 'Cluster ' + str(i))
            ax[1].add_patch(circle2)
        bins = int((y.max() - y.min())/5)
        if bins<=10:
            bins = 10
        ax2[2].hist(y[:, ix].flatten(), range=(y.min(), y.max()),
                    label='Cluster ' + str(i), alpha=0.5, bins = bins)
    ax2[2].set_xlabel('Standardized metabolite levels')
    ax2[2].set_ylabel('# Metabolites in Cluster x\n# Samples Per Metabolite', fontsize = 10)
    ax2[2].set_title('Metabolites')

    if mu_bug.shape[1] == 2:
        ax[1].set_aspect('equal')
        fig.tight_layout()
        fig.savefig(path + 'embedded_locations.png')
        plt.close(fig)
    fig2.tight_layout()
    fig2.savefig(path + 'cluster_histogram.png')
    plt.close(fig2)

    K = gen_z.shape[1]
    L = gen_u.shape[1]
    fig, ax = plt.subplots(K, L, figsize=(8 * L, 8 * K))
    # ranges = [[np.max(microbe_sum[:,i]/out[:,j]) - np.min(microbe_sum[:,i]/out[:,j]) for i in range(out.shape[1])] for j in range(out.shape[1])]
    # ixs = [np.argmin(r) for r in ranges]
    ax_ylim = (np.min(y.flatten()), np.max(y.flatten()))
    g = x @ gen_u
    ax_xlim = (np.min(g.flatten())-10, np.max(g.flatten())+10)
    for i in range(K):
        ixs = np.where(gen_z[:,i]==1)[0]
        if len(ixs) == 0:
            continue
        for j in range(L):
            # ax[i].scatter(microbe_sum[:,ixs[i]], out[:,i])
            for ii in ixs:
                ax[i, j].scatter(g[:, j], y[:, ii])
            ax[i, j].set_xlabel('Microbe sum')
            ax[i, j].set_ylabel(r'$y_{i}$ when $i=$' + str(i))
            ax[i, j].set_xlim(ax_xlim[0], ax_xlim[1])
            ax[i, j].set_ylim(ax_ylim[0], ax_ylim[1])
            ax[i, j].set_title('Metabolite Cluster ' + str(i) + ' vs Microbe Cluster ' + str(j))
            slope = np.round((np.max(y[:,ixs[0]]) - np.min(y[:, ixs[0]]))/((np.max(g[:,j]) - np.min(g[:,j]))),3)
            ax[i, j].text(0.6, 0.8, 'slope = ' + str(slope), horizontalalignment='center',
                            verticalalignment='center', transform=ax[i, j].transAxes)
    fig.tight_layout()
    fig.savefig(path + '-sum_x_v_y.png')
    plt.close(fig)

    fig, ax = plt.subplots(8,1, figsize = (8,4*8))
    for i in range(gen_z.shape[1]):
        ixs = np.where(gen_z[:, i] == 1)[0]
        for s in range(8):
            ax[s].hist(y[s, ixs].flatten(), range=(y[s,:].min(), y[s,:].max()),
                        label='Cluster ' + str(i), alpha=0.5, bins=bins)
    fig.tight_layout()
    fig.savefig(path + '-per_part_metabolites.png')
    plt.close(fig)


def plot_distribution(dist, param, true_val = None, ptype = 'init', path = '', **kwargs):
    if ptype == 'init':
        label = 'Initialized values'
    elif ptype == 'prior':
        label = 'True values'
    vals = dist.sample([500])
    if 'r' in param:
        vals = 1/vals
    elif 'z' in param or 'w' in param and ptype != 'init':
        vals = dist.sample([500, true_val.shape[1]])
        vals = torch.softmax(vals, 1)
    elif 'z' in param or 'w' in param and ptype == 'init':
        vals = dist.sample([500, true_val.shape[0], true_val.shape[1]])
        vals = torch.softmax(vals, 1)
        if true_val is not None:
            true_val = torch.softmax(true_val, 1)
    mean, std = np.round(vals.mean().item(),2), np.round(vals.std().item(),2)
    if len(vals.shape)>1:
        vals = vals.flatten()
    fig, ax = plt.subplots()
    bins = 10
    ax.hist(vals, bins = bins)
    ax.set_title(param + ', mean=' + str(mean) + ', std=' + str(std))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if true_val is not None:
        if isinstance(true_val, float):
            tv = [true_val]
        elif isinstance(true_val, list) or len(true_val.shape)<=1:
            tv = true_val
        else:
            tv = true_val.flatten()
        for k in np.arange(len(tv)):
            if k == 0:
                ax.axvline(tv[k], c = 'r', label = label)
            else:
                ax.axvline(tv[k], c='r')
            ax.legend(loc = 'upper right')
    r = [k + '_' + str(np.round(item,2)).replace('.','d') for k, item in kwargs.items()]
    if not os.path.isdir(path + '/' + ptype + 's/'):
        os.mkdir(path + '/' + ptype + 's/')
    plt.tight_layout()
    plt.savefig(path + '/' + ptype + 's/' + param + '-' + '-'.join(r))
    plt.close(fig)

def plot_param_traces(path, param_dict, params2learn, true_vals, net, fold):
    fig_dict, ax_dict = {},{}
    for name, plist in param_dict.items():
        if name in params2learn or 'all' in params2learn or name == 'z' or 'w_' in name:
            if len(plist[0].shape) == 0:
                n = 1
            else:
                n = plist[0].squeeze().shape[0]
                if n > 50:
                    n = 50
            if len(plist[0].squeeze().shape)<=1:
                fig_dict[name], ax_dict[name] = plt.subplots(n,
                                                             figsize=(5, 4 * n))
            else:
                nn = plist[0].squeeze().shape[1]
                if nn > 10:
                    nn = 10
                fig_dict[name], ax_dict[name] = plt.subplots(n, nn,
                                                             figsize = (5*nn, 4*n))
            for k in range(n):
                if len(plist[0].squeeze().shape) <= 1:
                    if name == 'sigma' or name == 'p':
                        trace = plist
                        ax_dict[name].plot(trace, label='Trace')
                        if name in true_vals.keys():
                            if not hasattr(true_vals[name], '__len__'):
                                tv= [true_vals[name]]
                            else:
                                tv = true_vals[name]
                            ax_dict[name].plot([tv[k]] * len(trace), c='r', label='True')
                        ax_dict[name].set_title(name + ', ' + str(k))
                        ax_dict[name].set_ylim(net.range_dict[name])
                        ax_dict[name].legend(loc='upper right')
                        ax_dict[name].set_xlabel('Iterations')
                        ax_dict[name].set_ylabel('Parameter Values')
                    else:
                        trace = [p.squeeze()[k] for p in plist]
                        # if name == 'r_bug' or name == 'r_met' or name == 'pi_met' or name == 'e_met':
                        #     nm = name.split('_')[-1]
                        #     trace = [p.squeeze()[mapping[nm]][k] for p in plist]
                        # else:
                        #     trace = [p.squeeze()[mapping['bug']][k] for p in plist]
                        ax_dict[name][k].plot(trace, label='Trace')
                        if name in true_vals.keys():
                            if not isinstance(true_vals[name], list):
                                true_vals[name] = true_vals[name].squeeze()
                            if np.array(true_vals[name]).shape[0] > k:
                                ax_dict[name][k].plot([true_vals[name][k]] * len(trace), c='r', label='True')
                        ax_dict[name][k].set_title(name + ', ' + str(k))
                        if name in net.range_dict.keys():
                            ax_dict[name][k].set_ylim(net.range_dict[name])
                        ax_dict[name][k].legend(loc = 'upper right')
                        ax_dict[name][k].set_xlabel('Iterations')
                        ax_dict[name][k].set_ylabel('Parameter Values')
                else:
                    for j in range(nn):
                        new_k, new_j = k, j
                        # if 'beta' in name or 'alpha' in name:
                        #     if k > 0 and 'beta' in name:
                        #         new_k = mapping['bug'][k-1] + 1
                        #     if 'alpha' in name:
                        #         new_k = mapping['bug'][k]
                        #     new_j = mapping['met'][j]
                        # elif 'mu' in name:
                        #     new_k = mapping[name.split('_')[-1]][k]
                        # elif 'w' in name:
                        #     try:
                        #         new_j = mapping['bug'][j]
                        #     except:
                        #         import pdb; pdb.set_trace()
                        # elif 'z' in name:
                        #     new_j = mapping['met'][j]

                        trace = [p.squeeze()[new_k, new_j] for p in plist]
                        if name == 'z' or name == 'w' or name == 'alpha':
                            trace_ma = [np.sum(trace[i:i + 5:]) / 5 for i in np.arange(len(trace) - 5)]
                            trace_ma[:0] = [trace_ma[0]]*(len(trace)-len(trace_ma))
                            trace = trace_ma
                        ax_dict[name][k, j].plot(trace, label='Trace')
                        if name in true_vals.keys():
                            if np.array(true_vals[name]).shape[1]>j and np.array(true_vals[name]).shape[0] > k:
                                ax_dict[name][k, j].plot([true_vals[name][k, j]] * len(trace), c='r', label='True')
                        ax_dict[name][k, j].set_title(name + ', ' + str(k) + ', ' + str(j))
                        if name in net.range_dict.keys():
                            ax_dict[name][k, j].set_ylim(net.range_dict[name])
                        ax_dict[name][k, j].legend(loc = 'upper right')
                        ax_dict[name][k, j].set_xlabel('Iterations')
                        ax_dict[name][k, j].set_ylabel('Parameter Values')
            fig_dict[name].tight_layout()
            fig_dict[name].savefig(path + 'seed' + str(fold) + '_' + name + '_parameter_trace.png')
            plt.close(fig_dict[name])

def plot_output_locations(path, net, best_mod, param_dict, fold, gen_w, type = 'best', plot_zeros = False):
    best_w = param_dict['w'][best_mod]
    # best_w = best_w[:, mapping['bug']]
    best_mu = param_dict['mu_bug'][best_mod]
    best_r = param_dict['r_bug'][best_mod]
    best_alpha = param_dict['alpha'][best_mod]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(net.microbe_locs[:,0], net.microbe_locs[:,1], facecolors='none',
               edgecolors='k')
    for i in range(best_w.shape[1]):
        ix = np.where(best_w[:,i] > 0.5)[0]
        if (len(ix) == 0 or np.sum(best_alpha[i,:])<0.5) and not plot_zeros:
            ax.scatter([], [])
            continue
        p2 = ax.scatter(net.microbe_locs[ix, 0], net.microbe_locs[ix, 1])
        ax.scatter(best_mu[i, 0], best_mu[i, 1], marker='*', color=p2.get_facecolor().squeeze())
        ax.text(best_mu[i, 0], best_mu[i, 1], 'predicted\ncluster ' + str(i) + ' mean')
        ax.set_title('Microbes')
        circle2 = plt.Circle((best_mu[i, 0], best_mu[i, 1]), best_r[i],
                             alpha=0.2, color=p2.get_facecolor().squeeze(), label='Cluster ' + str(i))
        ax.add_patch(circle2)
        ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(path + 'seed' + str(fold) + '-' + type + '-plot_zeros'*plot_zeros + '-bug_clusters.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    best_z = param_dict['z'][best_mod]
    # [:,mapping['met']]
    best_mu = param_dict['mu_met'][best_mod]
    best_r = param_dict['r_met'][best_mod]
    ax.scatter(net.met_locs[:, 0], net.met_locs[:, 1], facecolors='none',
               edgecolors='k')
    for i in range(param_dict['z'][0].shape[1]):
        ix = np.where(best_z[:,i] > 0.5)[0]
        if len(ix) == 0 and not plot_zeros:
            ax.scatter([], [])
            continue

        p2 = ax.scatter(net.met_locs[ix, 0], net.met_locs[ix, 1])
        ax.scatter(best_mu[i, 0], best_mu[i, 1], marker='*', color=p2.get_facecolor().squeeze())
        ax.text(best_mu[i, 0], best_mu[i, 1], 'predicted\ncluster ' + str(i) + ' mean')
        ax.set_title('Metabolites')
        circle2 = plt.Circle((best_mu[i, 0], best_mu[i, 1]), best_r[i],
                             alpha=0.2, color=p2.get_facecolor().squeeze(), label='Cluster ' + str(i))
        ax.add_patch(circle2)
        ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(path + 'seed' + str(fold) + '-' + type +'-plot_zeros'*plot_zeros + '-predicted_metab_clusters.png')
    plt.close(fig)

def plot_xvy(path, net, x, out_vec, best_mod, targets, gen_z, gen_w, param_dict, seed):
    out = out_vec[best_mod].detach().numpy()
    best_w = param_dict[seed]['w'][best_mod]
    best_z = param_dict[seed]['z'][best_mod]
    best_alpha = param_dict[seed]['alpha'][best_mod]

    # out = out[:, mapping['met']]
    microbe_sum = x.detach().numpy() @ best_w
    true_sum = x.detach().numpy() @ gen_w
    # microbe_sum = microbe_sum[:, mapping['bug']]
    # target_sum = targets @ gen_z
    num_active_microbe = len(np.where(np.sum(best_w,0)>(0.1))[0])
    num_active_met = len(np.where(np.sum(best_z, 0) > 0)[0])
    if num_active_met==0:
        num_active_met = 1
    if num_active_microbe==0:
        num_active_microbe = 1
    fig, ax = plt.subplots(num_active_microbe, num_active_met, figsize = (8*num_active_met,8*num_active_microbe))
    if num_active_microbe==1:
        ax = np.expand_dims(ax,0)
    if num_active_met==1:
        ax = np.expand_dims(ax,1)
    ax_xlim = (np.min(true_sum.flatten())-10, np.max(true_sum.flatten())+10)
    ax_ylim = (np.min(targets.flatten()), np.max(targets.flatten()))

    if np.min(microbe_sum.flatten())>ax_xlim[1] or np.max(microbe_sum.flatten()) < ax_xlim[0]:
        ax_xlim = (np.min(microbe_sum.flatten()), np.max(microbe_sum.flatten()))
    if np.min(out.flatten())>ax_ylim[1] or np.max(out.flatten())<ax_ylim[0]:
        ax_ylim = (np.min(out.flatten()), np.max(out.flatten()))
    if np.max(out.flatten())>ax_ylim[1]:
        ax_ylim = (ax_ylim[0], np.max(out.flatten()))
    if np.min(out.flatten())< ax_ylim[0]:
        ax_ylim = (np.min(out.flatten()), ax_ylim[1])
    if np.max(microbe_sum.flatten())>ax_xlim[1]:
        ax_xlim = (ax_xlim[0], np.max(microbe_sum.flatten()))
    if np.min(microbe_sum.flatten())< ax_xlim[0]:
        ax_xlim = (np.min(microbe_sum.flatten()), ax_xlim[1])
    # ranges = [[np.max(microbe_sum[:,i]/out[:,j]) - np.min(microbe_sum[:,i]/out[:,j]) for i in range(out.shape[1])] for j in range(out.shape[1])]
    # ixs = [np.argmin(r) for r in ranges]
    ii=0
    for i in range(out.shape[1]):
        ixs = np.where(best_z[:,i]==1)[0]
        if len(ixs) == 0:
            continue
        jj = 0
        for j in range(microbe_sum.shape[1]):
            ixs = np.where(best_w[:,j]>0.1)[0]
            if len(ixs) == 0 or best_alpha[j,i]<0.1:
                continue
            ax[jj,ii].scatter(microbe_sum[:, j], out[:, i], c = 'b', label = 'Guess')
            ax[jj,ii].set_xlabel('Microbe sum')
            ax[jj,ii].set_ylabel(r'$y_{i}$ when $i=$' + str(i))
            ax[jj,ii].set_title('Metabolite Cluster ' + str(i) + ' vs Microbe Cluster ' + str(j))
            ax[jj,ii].legend(loc = 'upper right')
            slope = np.round((np.max(out[:,i]) - np.min(out[:, i]))/((np.max(microbe_sum[:,j]) - np.min(microbe_sum[:,j]))),3)
            try:
                ax[jj,ii].text(0.6, 0.8,'slope = ' + str(slope), horizontalalignment='center',
                     verticalalignment='center', transform=ax[ii,jj].transAxes)
            except:
                return
            ax[jj,ii].set_xlim(ax_xlim[0], ax_xlim[1])
            ax[jj,ii].set_ylim(ax_ylim[0], ax_ylim[1])
            jj+=1
        ii+=1
    fig.savefig(path + 'seed' + str(seed) + '-sum_x_v_y.png')
    plt.close(fig)

def plot_interactions(path, best_mod, param_dict,seed):
    plt.figure()
    if isinstance(param_dict['alpha'], list):
        best_alpha = param_dict['alpha'][best_mod]
        best_beta = param_dict['beta'][best_mod]
        best_w = param_dict['w'][best_mod]
        best_z = param_dict['z'][best_mod]
    else:
        best_alpha = param_dict['alpha']
        best_beta = param_dict['beta']
        best_w = param_dict['w']
        best_z = param_dict['z']
    try:
        weights = best_alpha*best_beta[1:,:]
    except:
        weights = best_alpha

    best_w[best_w < 0.5] = 0
    # best_beta = best_beta*100

    widest = max([best_w.shape[0], best_z.shape[0]])
    network = NeuralNetwork(number_of_neurons_in_widest_layer=widest, neuron_radius=4,
                            vertical_distance_between_layers = 50)
    network.add_layer(best_w.shape[0], np.round(best_w.T, 4), label='microbes')
    network.add_layer(best_alpha.shape[0], np.round(weights.T, 4), label='microbe clusters', node_colors='cmap')
    network.add_layer(best_alpha.shape[1], best_z, label = 'metabolite clusters', node_colors='cmap')
    network.add_layer(best_z.shape[0], label = 'metabolites')
    # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)
    network.draw(path + 'seed' + str(seed) + '-layers.png')
    plt.close()


def plot_rules_detectors_tree(path, net, best_mod, param_dict, microbe_locs, seed):
    plt.figure()
    N_bug = microbe_locs.shape[0]
    num_rules = param_dict['w'][best_mod].shape[0]
    num_detectors = param_dict['w'][best_mod].shape[1]
    kappa = np.stack([((param_dict['mu_bug'][best_mod] - microbe_locs[m,:])**2).sum(-1) for m in range(microbe_locs.shape[0])])
    u = sigmoid((param_dict['r_bug'][best_mod] - kappa)/net.temp_scheduled)
    u = np.hstack([u[:,:,i] for i in range(u.shape[-1])])

    widest = max([u.shape[1], num_rules*num_detectors])
    network = NeuralNetwork(number_of_neurons_in_widest_layer = widest)
    network.add_layer(N_bug, np.round(u.T, 4), label = 'microbes')
    w_temp = np.zeros((param_dict['w'][best_mod].shape[0], param_dict['w'][best_mod].shape[1]*num_rules))
    st = 0
    for i in range(w_temp.shape[0]):
        w_temp[i,st:st + param_dict['w'][best_mod].shape[1]] = param_dict['w'][best_mod][i, :]
        st = param_dict['w'][best_mod].shape[1]*(i+1)
    # w_temp = np.repeat(param_dict['w'][best_mod], 2, axis = 0)
    network.add_layer(param_dict['w'][best_mod].shape[1]*num_rules, weights = np.round(w_temp, 4),
                      label = 'detectors', weight_label = 'u', node_labels = list(range(0, num_detectors))*num_rules)
    network.add_layer(param_dict['w'][best_mod].shape[0], label = 'rules', weight_label = 'w')
    # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)
    network.draw(path + 'seed' + str(seed) + '-rules_detectors.png')
    plt.figure()


def plot_output(path, best_mod, out_vec, targets, gen_z, gen_w, param_dict, fold, type = 'unknown', meas_var = 0.1):
    best_w = param_dict['w'][best_mod]
    bug_clusters = [np.where(best_w[:,i]>0.5)[0] for i in np.arange(best_w.shape[1])]
    for ii,clust in enumerate(bug_clusters):
        if not os.path.isfile(path + 'seed' + str(fold) + 'microbe_clusters.txt'):
            with open(path + 'seed' + str(fold) + 'microbe_clusters.txt', 'w') as f:
                f.writelines('Cluster ' + str(ii) + ': ' + str(clust) + '\n')
        else:
            with open(path + 'seed' + str(fold) + 'microbe_clusters.txt', 'a') as f:
                f.writelines('Cluster ' + str(ii) + ': ' + str(clust) + '\n')

    tp, fp, tn, fn, ri = pairwise_eval(np.round(best_w), gen_w)
    pairwise_cf = {'Same cluster': {'Predicted Same cluster': tp, 'Predicted Different cluster': fn},
                   'Different cluster': {'Predicted Same cluster': fp, 'Predicted Different cluster': tn}}
    pd.DataFrame(pairwise_cf).T.to_csv(
        path + 'seed' + str(fold) + '_PairwiseConfusionBugs_' + type + '_' + str(np.round(ri, 3)).replace('.',
                                                                                                            'd') + '.csv')
    ri = str(np.round(ri, 3))
    if not os.path.isfile(path + type +'-ri_bug.txt'):
        with open(path + type +'-ri_bug.txt', 'w') as f:
            f.write('Seed ' + str(fold) + ':RI ' + ri + '\n')
    else:
        with open(path + type +'-ri_bug.txt', 'a') as f:
            f.write('Seed ' + str(fold) + ':RI ' + ri + '\n')

    # fig_dict2, ax_dict2 = plt.subplots(targets.shape[1], 1,
    #                                                figsize=(8, 4 * targets.shape[1]))
    # fig_dict3, ax_dict3 = plt.subplots(gen_z.shape[1], 1, figsize=(8, 4 * gen_z.shape[1]))
    pred_clusters = out_vec[best_mod]
    pred_z = param_dict['z'][best_mod]
    preds = torch.matmul(pred_clusters + meas_var*torch.randn(pred_clusters.shape), torch.Tensor(pred_z).T)
    total = np.concatenate([targets, preds.detach().numpy()])

    # pred_z = pred_z[:, mapping['met']]
    true = np.argmax(gen_z,1)
    z_guess = np.argmax(pred_z, 1)
    nmi = np.round(normalized_mutual_info_score(true, z_guess), 3)
    try:
        tp, fp, tn, fn, ri = pairwise_eval(z_guess, true)
        pairwise_cf = {'Same cluster': {'Predicted Same cluster': tp, 'Predicted Different cluster': fn},
                       'Different cluster':{'Predicted Same cluster': fp, 'Predicted Different cluster': tn}}
        pd.DataFrame(pairwise_cf).T.to_csv(
            path + 'seed' + str(fold) + '_PairwiseConfusionMetabs_' + type + '_' + str(np.round(ri,3)).replace('.', 'd') + '.csv')
        ri = str(np.round(ri, 3))
    except:
        ri = 'NA'
    if not os.path.isfile(path + type +'-nmi_ri.txt'):
        with open(path + type +'-nmi_ri.txt', 'w') as f:
            f.write('Seed ' + str(fold) + ': NMI ' + str(nmi) + ', RI ' + ri + '\n')
    else:
        with open(path + type +'-nmi_ri.txt', 'a') as f:
            f.write('Seed ' + str(fold) + ': NMI ' + str(nmi) + ', RI ' + ri + '\n')


    # gen_z = gen_z[:, mapping['met']]

    figx, axx = plt.subplots(figsize = (8, 8))
    color = cm.rainbow(np.linspace(0, 1, gen_z.shape[1]))

    RMSE = np.zeros(gen_z.shape[0])
    residuals = np.zeros(targets.shape)
    preds_t = np.zeros(targets.shape)

    fig, ax = plt.subplots(8,1,figsize = (8,4*8))
    for s in range(8):
        ax[s].bar(np.arange(preds.shape[1])-0.2, preds[s,:].flatten().detach().numpy(), width = 0.4, label = 'Predicted')
        ax[s].bar(np.arange(targets.shape[1]) + 0.2, targets[s, :].flatten(), width=0.4, label = 'True')
        ax[s].set_title('Subject ' + str(s))
        ax[s].legend(loc= 'upper right')
    fig.savefig(path + 'seed' + str(fold) + '-predictions.png')
    plt.close(fig)

    for cluster in range(gen_z.shape[1]):
        met_ids = np.where(gen_z[:, cluster] == 1)[0]
        if len(met_ids)==0:
            continue
    #     bins = int((total.max() - total.min()) / 5)
        residuals[:, met_ids] = targets[:, met_ids] - preds[:, met_ids].detach().numpy()
        preds_t[:, met_ids] = preds[:, met_ids].detach().numpy()
        for i in range(len(met_ids)):
            RMSE[met_ids[i]] = np.sqrt(np.sum(((preds[:, met_ids[i]].detach().numpy() - targets[:, met_ids[i]])**2))/preds.shape[0])
    #     if bins<10:
    #         bins = 10
    #     ax_dict3[cluster].hist(preds[:, met_ids].flatten().detach().numpy(), range=(total.min(), total.max()),
    #                                  label='guess', alpha=0.5, bins = bins)
    #     ax_dict3[cluster].hist(targets[:, met_ids].flatten(), range=(total.min(), total.max()), label='true',
    #                                  alpha=0.5, bins = bins)
    #     ax_dict3[cluster].set_title('Cluster ' + str(cluster))
    #     ax_dict3[cluster].legend(loc = 'upper right')
    #     ax_dict3[cluster].set_xlabel('Metabolite Levels')
    #     ax_dict3[cluster].set_ylabel('# Metabolites in Cluster x\n# Samples Per Metabolite')

        axx.scatter(preds_t[:, met_ids], residuals[:, met_ids], c=[color[cluster]],
                    label='Cluster ' + str(cluster))
    # fig_dict3.tight_layout()
    # fig_dict3.savefig(path + 'seed' + str(fold) + '_cluster_histograms_' + type + '.pdf')
    # plt.close(fig_dict3)
    RMSE_avg = np.round(np.mean(RMSE),4)
    RMSE_std = np.round(np.std(RMSE),4)
    if not os.path.isfile(path + 'RMSE.txt'):
        with open(path + 'RMSE.txt', 'w') as f:
            f.write('SEED ' + str(fold) + ' RMSE: ' + str(RMSE_avg) + ' +-' + str(RMSE_std) + '\n')
    else:
        with open(path + 'RMSE.txt', 'a') as f:
            f.write('SEED ' + str(fold) + ' RMSE: ' + str(RMSE_avg) + ' +-' + str(RMSE_std) + '\n')

    RMSE_df = pd.Series(RMSE, index = ['Metabolite ' + str(i) for i in range(len(RMSE))])
    RMSE_df.to_csv(path + 'seed' + str(fold) + 'RMSE.csv')

    axx.set_title('Residuals plot for metabolite level predictions')
    axx.set_xlabel('Predicted Levels')
    axx.set_ylabel('Residuals')
    axx.legend()
    figx.savefig(path + 'seed' + str(fold) + '_residuals_' + type + '.png')
    plt.close(figx)


def plot_loss(fig3, ax3, fold, iterations, loss_vec, test_loss=None, lowest_loss = None):
    ax3.set_title('Seed ' + str(fold))
    ax3.plot(iterations, loss_vec, label='training loss')
    if test_loss is not None:
        ax3.plot(iterations, test_loss, label='test loss')
        ax3.legend(loc = 'upper right')
    if lowest_loss is not None:
        ax3.plot(iterations, lowest_loss, label='lowest loss')
        ax3.legend(loc = 'upper right')
    return fig3, ax3

def plot_loss_dict(path, fold, loss_dict):
    params = loss_dict.keys()
    fig, ax = plt.subplots(len(params),1, figsize = (8, 5*len(params)))
    for i,param in enumerate(params):
        loss = loss_dict[param]
        ax[i].plot(np.arange(len(loss)), loss)
        ax[i].set_xlabel('Iterations')
        ax[i].set_ylabel('Loss')
        ax[i].set_title(param)
    fig.savefig(path + 'seed' + str(fold) + '_loss_dict.png')