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
    fig, ax = plt.subplots(1, 2, figsize = (10,5))
    fig2, ax2 = plt.subplots(3, 1)
    p1 = ax[0].scatter(gen_bug_locs[:, 0], gen_bug_locs[:, 1], color = 'k')
    for i in range(gen_bug_locs.shape[0]):
        ax[0].text(gen_bug_locs[i, 0], gen_bug_locs[i, 1], str(i))
    if len(mu_bug.shape)>2:
        mu_bug = mu_bug[0,:,:]
        r_bug = r_bug[0,:]
    for i in range(mu_bug.shape[0]):
        p1 = ax[0].scatter(mu_bug[i,0], mu_bug[i,1], marker='*')
        circle1 = plt.Circle((mu_bug[i,0], mu_bug[i,1]), r_bug[i],
                             alpha=0.2, label='Cluster ' + str(i),color=p1.get_facecolor().squeeze())
        # for ii in ix:
        #     ax[0].text(gen_bug_locs[ii,0], gen_bug_locs[ii,1], 'Bug ' + str(ii))
        ax[0].add_patch(circle1)
        ax[0].set_title('Microbes')
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

    ax[0].legend(loc = 'upper right')
    ax2[0].legend(loc = 'upper right')
    ax2[1].legend(loc = 'upper right')
    for i in range(gen_z.shape[1]):
        ix = np.where(gen_z[:, i] == 1)[0]
        p2 = ax[1].scatter(gen_met_locs[ix, 0], gen_met_locs[ix, 1])
        ax[1].scatter(mu_met[i, 0], mu_met[i, 1], marker='*', color=p2.get_facecolor().squeeze())
        ax[1].set_title('Metabolites')
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


    ax[1].set_aspect('equal')
    ax[1].legend(loc = 'upper right')
    ax2[1].legend(loc = 'upper right')
    fig.tight_layout()
    fig.savefig(path + 'embedded_locations.pdf')
    fig2.tight_layout()
    fig2.savefig(path + 'cluster_histogram.pdf')
    plt.close(fig)
    plt.close(fig2)

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
        fig, ax = plt.subplots(vals.shape[1], figsize = (5, 4*vals.shape[1]))
        for i in np.arange(true_val.shape[-1]):
            bins = 10
            if len(vals.shape)>2:
                ax[i].hist(vals[:,:,i].flatten(), bins = bins)
            else:
                ax[i].hist(vals[:,i], bins=bins)
            ax[i].set_title(param + ', mean=' + str(mean) + ', std=' + str(std))
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('F(x)')
            if true_val is not None:
                if len(true_val.shape) > 2:
                    tv = true_val[:, :, i].flatten()
                else:
                    tv = true_val[:, i]
                for j in range(tv.shape[0]):
                    if j == 0:
                        ax[i].axvline(tv[j], c='r', label=label)
                    else:
                        ax[i].axvline(tv[j], c='r')
                ax[i].legend(loc = 'upper right')
    else:
        fig, ax = plt.subplots()
        bins = 10
        ax.hist(vals, bins = bins)
        ax.set_title(param + ', mean=' + str(mean) + ', std=' + str(std))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if true_val is not None:
            if isinstance(true_val, list) or len(true_val.shape)<=1:
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
        n = plist[0].squeeze().shape[0]
        if n > 50:
            n = 50
        if len(plist[0].squeeze().shape)==1:
            fig_dict[name], ax_dict[name] = plt.subplots(n,
                                                         figsize=(5, 4 * n))
        else:
            nn = plist[0].squeeze().shape[1]
            if nn > 10:
                nn = 10
            fig_dict[name], ax_dict[name] = plt.subplots(n, nn,
                                                         figsize = (5*nn, 4*n))
        if name in params2learn or 'all' in params2learn:
            for k in range(n):
                if len(plist[0].squeeze().shape) == 1:
                    trace = [p.squeeze()[k] for p in plist]
                    ax_dict[name][k].plot(trace, label='Trace')
                    if not isinstance(true_vals[name], list):
                        true_vals[name] = true_vals[name].squeeze()
                    if np.array(true_vals[name]).shape[0] > k:
                        ax_dict[name][k].plot([true_vals[name][k]] * len(trace), c='r', label='True')
                    ax_dict[name][k].set_title(name + ', ' + str(k))
                    ax_dict[name][k].set_ylim(net.range_dict[name])
                    ax_dict[name][k].legend(loc = 'upper right')
                    ax_dict[name][k].set_xlabel('Iterations')
                    ax_dict[name][k].set_ylabel('Parameter Values')
                else:
                    for j in range(nn):
                        trace = [p.squeeze()[k, j] for p in plist]
                        ax_dict[name][k, j].plot(trace, label='Trace')
                        if np.array(true_vals[name]).shape[1]>j and np.array(true_vals[name]).shape[0] > k:
                            ax_dict[name][k, j].plot([true_vals[name][k, j]] * len(trace), c='r', label='True')
                        ax_dict[name][k, j].set_title(name + ', ' + str(k) + ', ' + str(j))
                        ax_dict[name][k, j].set_ylim(net.range_dict[name])
                        ax_dict[name][k, j].legend(loc = 'upper right')
                        ax_dict[name][k, j].set_xlabel('Iterations')
                        ax_dict[name][k, j].set_ylabel('Parameter Values')
            fig_dict[name].tight_layout()
            fig_dict[name].savefig(path + 'fold' + str(fold) + '_' + name + '_parameter_trace.pdf')
            plt.close(fig_dict[name])

def plot_output_locations(path, net, best_mod, param_dict, fold, gen_w, type = 'best'):
    if 'w' not in param_dict.keys():
        best_w = param_dict['u'][best_mod]
    else:
        best_w = param_dict['w'][best_mod]
    best_mu = param_dict['mu_bug'][best_mod]
    best_r = param_dict['r_bug'][best_mod]
    if best_w.shape[0]==net.microbe_locs.shape[0] and best_w.shape[1]==net.L:
        if best_w.shape[-1] == net.K and len(best_w.shape)>2:
            fig, ax = plt.subplots(net.K, 1, figsize = (5*net.K, 5))
            for k in range(net.K):
                for i in range(best_w.shape[1]):
                    ix = np.where(best_w[:,i, k] > 0.5)[0]
                    p2 = ax[k].scatter(net.microbe_locs[ix, 0], net.microbe_locs[ix, 1])
                    ix_true = np.where(gen_w[:,i,k]>0.5)[0]
                    ax[k].scatter(net.microbe_locs[ix_true, 0], net.microbe_locs[ix_true, 1], facecolors = 'none',
                                  edgecolors = 'k')
                    ax[k].scatter(best_mu[i, 0, k], best_mu[i, 1, k], marker='*', color=p2.get_facecolor().squeeze())
                    ax[k].text(best_mu[i, 0, k], best_mu[i, 1, k], 'predicted\ncluster ' + str(i) + ' mean')
                    ax[k].set_title('Microbe clusters for metabolite cluster ' + str(k))
                    circle2 = plt.Circle((best_mu[i, 0, k], best_mu[i, 1, k]), best_r[i, k],
                                         alpha=0.2, color=p2.get_facecolor().squeeze(), label='Cluster ' + str(i))
                    ax[k].add_patch(circle2)
                    ax[k].set_aspect('equal')
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
            for i in range(best_w.shape[1]):
                ix = np.where(best_w[:,i] > 0.5)[0]
                if len(ix) == 0:
                    continue
                p2 = ax.scatter(net.microbe_locs[ix, 0], net.microbe_locs[ix, 1])
                ax.scatter(best_mu[i, 0], best_mu[i, 1], marker='*', color=p2.get_facecolor().squeeze())
                ix_true = np.where(gen_w[:, i] > 0.5)[0]
                ax.scatter(net.microbe_locs[ix_true, 0], net.microbe_locs[ix_true, 1], facecolors='none',
                              edgecolors='k')
                ax.text(best_mu[i, 0], best_mu[i, 1], 'predicted\ncluster ' + str(i) + ' mean')
                ax.set_title('Microbes')
                circle2 = plt.Circle((best_mu[i, 0], best_mu[i, 1]), best_r[i],
                                     alpha=0.2, color=p2.get_facecolor().squeeze(), label='Cluster ' + str(i))
                ax.add_patch(circle2)
                ax.set_aspect('equal')

    else:
        if best_w.shape[-1] == net.K and len(best_w.shape) > 2:
            fig, ax = plt.subplots(net.K, best_w.shape[1], figsize=(5 * best_w.shape[1], 5* net.K))
            for k in range(net.K):
                for i in range(best_w.shape[0]):
                    ax[k,i].scatter(net.microbe_locs[:, 0], net.microbe_locs[:, 1], facecolors='r', edgecolors='r')
                    for ii in range(net.microbe_locs.shape[0]):
                        ax[k,i].text(net.microbe_locs[ii, 0], net.microbe_locs[ii, 1],
                                   str(ii))
                    ix = range(best_w.shape[1])
                    ix_on = np.where(best_w[i, :, k] > 0.5)[0]
                    ax[k,i].set_title('Microbes, Outer Cluster ' + str(i) + ', Metab Cluster ' + str(k))
                    color = cm.rainbow(np.linspace(0, 1, best_w.shape[1]))
                    ct = 0
                    for ixx in ix:
                        if ixx in ix_on:
                            ax[k,i].text(best_mu[i, ixx, k, 0], best_mu[i, ixx, k, 1], 'detector ' + str(ixx) + ' ON')
                            ax[k,i].scatter(best_mu[i, ixx, 0], best_mu[i, ixx, k, 1], marker='*', color=color[ixx])
                        else:
                            ax[k,i].text(best_mu[i, ixx,k,  0], best_mu[i, ixx, k, 1], 'detector ' + str(ixx) + ' OFF')
                            ax[k,i].scatter(best_mu[i, ixx, k, 0], best_mu[i, ixx, k, 1], marker='o', color=color[ixx])
                        circle2 = plt.Circle((best_mu[i, ixx, k, 0], best_mu[i, ixx, k, 1]), best_r[i, ixx],
                                             alpha=0.2, label='Detector ' + str(ixx), color=color[ixx])

                        ax[k,i].add_patch(circle2)
                    ax[k,i].set_aspect('equal')
                    ax[k,i].legend(loc = 'upper right')
        else:
            fig, ax = plt.subplots(1, best_w.shape[0], figsize=(5 * best_w.shape[0], 5))
            # iterate over rules
            for i in range(best_w.shape[0]):
                ax[i].scatter(net.microbe_locs[:, 0], net.microbe_locs[:, 1], facecolors='none', edgecolors='r')
                for ii in range(net.microbe_locs.shape[0]):
                    ax[i].text(net.microbe_locs[ii, 0], net.microbe_locs[ii, 1],
                               str(ii))
                ix = range(best_w.shape[1])
                ix_on = np.where(best_w[i, :] > 0.5)[0]
                ax[i].set_title('Microbes, Rule ' + str(i))
                color = cm.rainbow(np.linspace(0, 1, best_w.shape[1]))
                ct = 0
                for ixx in ix:
                    if ixx in ix_on:
                        ax[i].text(best_mu[i, ixx, 0], best_mu[i, ixx, 1], 'detector ' + str(ixx) + ' ON')
                        ax[i].scatter(best_mu[i, ixx, 0], best_mu[i, ixx, 1], marker='*', color=color[ixx])
                    else:
                        ax[i].text(best_mu[i, ixx, 0], best_mu[i, ixx, 1], 'detector ' + str(ixx) + ' OFF')
                        ax[i].scatter(best_mu[i, ixx, 0], best_mu[i, ixx, 1], marker='o', color=color[ixx])
                    circle2 = plt.Circle((best_mu[i, ixx, 0], best_mu[i, ixx, 1]), best_r[i, ixx],
                                         alpha=0.2, label='Detector ' + str(ixx), color=color[ixx])

                    ax[i].add_patch(circle2)
                ax[i].set_aspect('equal')
                ax[i].legend(loc = 'upper right')
    fig.tight_layout()
    fig.savefig(path + 'fold' + str(fold) + '-' + type + '-bug_clusters.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    best_z = np.argmax(param_dict['z'][best_mod],1)
    best_mu = param_dict['mu_met'][best_mod]
    best_r = param_dict['r_met'][best_mod]
    for i in range(param_dict['z'][0].shape[1]):
        ix = np.where(best_z == i)[0]
        if len(ix) == 0:
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
    fig.savefig(path + 'fold' + str(fold) + '-' + type + '-predicted_metab_clusters.pdf')
    plt.close(fig)

def plot_xvy(path, net, x, out_vec, best_mod, param_dict, microbe_locs, seed):
    out = out_vec[best_mod]
    kappa = np.stack(
        [((param_dict[seed]['mu_bug'][best_mod] - microbe_locs[m, :]) ** 2).sum(-1) for m in range(microbe_locs.shape[0])])
    u = sigmoid((param_dict[seed]['r_bug'][best_mod] - kappa) / net.temp_scheduled)
    microbe_sum = x @ u
    fig, ax = plt.subplots(out.shape[1], 1, figsize = (8,8*out.shape[1]))
    for i in range(out.shape[1]):
        ax[i].scatter(microbe_sum[:,i], out[:,i].detach().numpy())
        ax[i].set_xlabel('Microbe sum')
        ax[i].set_ylabel('y')
        ax[i].set_title('Cluster ' + str(i))
    fig.tight_layout()
    fig.savefig(path + 'fold' + str(seed) + '-sum_x_v_y.pdf')
    plt.close(fig)

def plot_rules_detectors_tree(path, net, best_mod, param_dict, microbe_locs, seed):
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
    network.draw(path + 'seed' + str(seed) + '-rules_detectors.pdf')

def plot_output(path, path_orig, best_mod, out_vec, targets, gen_z,  param_dict, fold, type = 'unknown', meas_var = 0.1):

    fig_dict2, ax_dict2 = plt.subplots(targets.shape[1], 1,
                                                   figsize=(8, 4 * targets.shape[1]))
    fig_dict3, ax_dict3 = plt.subplots(gen_z.shape[1], 1, figsize=(8, 4 * gen_z.shape[1]))
    pred_clusters = out_vec[best_mod]
    pred_z = param_dict['z'][best_mod]
    preds = torch.matmul(pred_clusters + meas_var*torch.randn(pred_clusters.shape), torch.Tensor(pred_z).T)
    total = np.concatenate([targets, preds.detach().numpy()])

    true = np.argmax(gen_z,1)
    z_guess = np.argmax(pred_z, 1)
    nmi = np.round(normalized_mutual_info_score(true, z_guess), 3)
    try:
        tp, fp, tn, fn, ri = pairwise_eval(z_guess, true)
        pairwise_cf = {'Same cluster': {'Predicted Same cluster': tp, 'Predicted Different cluster': fn},
                       'Different cluster':{'Predicted Same cluster': fp, 'Predicted Different cluster': tn}}
        pd.DataFrame(pairwise_cf).T.to_csv(
            path + 'fold' + str(fold) + '_PairwiseConfusionMetabs_' + type + '_' + str(np.round(ri,3)).replace('.', 'd') + '.csv')
        ri = str(np.round(ri, 3))
    except:
        ri = 'NA'
    if not os.path.isfile(path_orig + 'fold' + str(fold) + '-' + type +'-nmi_ri.txt'):
        with open(path_orig + 'fold' + str(fold) + '-' + type +'-nmi_ri.txt', 'w') as f:
            f.write('Epoch ' + str(len(out_vec)) + ': NMI ' + str(nmi) + ', RI ' + ri + '\n')
    else:
        with open(path_orig + 'fold' + str(fold) + '-' + type +'-nmi_ri.txt', 'a') as f:
            f.write('Epoch ' + str(len(out_vec)) + ': NMI ' + str(nmi) + ', RI ' + ri + '\n')



    for met in range(targets.shape[1]):
        cluster_id = np.where(gen_z[met, :] == 1)[0]
        # range_ratio = (preds[:, met].max() - preds[:, met].min()) / (total.max() - total.min())
        bins = int((total.max() - total.min())/5)
        if bins<10:
            bins = 10
        ax_dict2[met].hist(preds[:, met].detach().numpy(), range=(total.min(), total.max()), label='guess',
                                 alpha=0.5, bins = bins)
        ax_dict2[met].hist(targets[:, met], range=(total.min(), total.max()),
                                 label='true', alpha=0.5, bins = bins)
        ax_dict2[met].set_title('Metabolite ' + str(met) + ', Cluster ' + str(cluster_id))
        ax_dict2[met].legend(loc = 'upper right')
        ax_dict2[met].set_xlabel('Metabolite Levels')
        ax_dict2[met].set_ylabel('# Samples Per Metabolite')
    fig_dict2.tight_layout()
    fig_dict2.savefig(path + 'fold' + str(fold) + '_histograms_ ' + type + '.pdf')
    plt.close(fig_dict2)

    for cluster in range(gen_z.shape[1]):
        met_ids = np.where(gen_z[:, cluster] == 1)[0]
        bins = int((total.max() - total.min()) / 5)
        if bins<10:
            bins = 10
        ax_dict3[cluster].hist(preds[:, met_ids].flatten().detach().numpy(), range=(total.min(), total.max()),
                                     label='guess', alpha=0.5, bins = bins)
        ax_dict3[cluster].hist(targets[:, met_ids].flatten(), range=(total.min(), total.max()), label='true',
                                     alpha=0.5, bins = bins)
        ax_dict3[cluster].set_title('Cluster ' + str(cluster))
        ax_dict3[cluster].legend(loc = 'upper right')
        ax_dict3[cluster].set_xlabel('Metabolite Levels')
        ax_dict3[cluster].set_ylabel('# Metabolites in Cluster x\n# Samples Per Metabolite')
    fig_dict3.tight_layout()
    fig_dict3.savefig(path + 'fold' + str(fold) + '_cluster_histograms_' + type + '.pdf')
    plt.close(fig_dict3)

def plot_loss(fig3, ax3, fold, iterations, loss_vec, test_loss=None, lowest_loss = None):
    ax3.set_title('Fold ' + str(fold))
    ax3.plot(np.arange(iterations), loss_vec, label='training loss')
    if iterations%100 != 1:
        xvals = np.append(np.arange(0, iterations, 100), iterations)
    else:
        xvals = np.arange(0, iterations, 100)
    if test_loss is not None:
        ax3.plot(xvals, test_loss, label='test loss')
        ax3.legend(loc = 'upper right')
    if lowest_loss is not None:
        ax3.plot(np.arange(iterations), lowest_loss, label='lowest loss')
        ax3.legend(loc = 'upper right')
    return fig3, ax3