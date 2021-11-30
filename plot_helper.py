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

def plot_syn_data(path, x, y, gen_w, gen_z, gen_bug_locs, gen_met_locs,
                  mu_bug, r_bug, mu_met, r_met):
    fig, ax = plt.subplots(1, 2)
    fig2, ax2 = plt.subplots(2, 1)
    for i in range(gen_w.shape[1]):
        ix = np.where(gen_w[:, i] == 1)[0]
        p1 = ax[0].scatter(gen_bug_locs[ix, 0], gen_bug_locs[ix, 1])
        circle1 = plt.Circle((mu_bug[i,0], mu_bug[i,1]), r_bug[i],
                             alpha=0.2, color=p1.get_facecolor().squeeze(),label='Cluster ' + str(i))
        for ii in ix:
            ax[0].text(gen_bug_locs[ii,0], gen_bug_locs[ii,1], 'Bug ' + str(ii))
        ax[0].add_patch(circle1)
        ax[0].set_title('Microbes')
        # for ii in ix:
        bins = int((x.max() - x.min()) / 5)
        if bins<=10:
            bins = 10
        ax2[0].hist(x[:, ix].flatten(), range=(x.min(), x.max()), label='Cluster ' + str(i), alpha=0.5, bins = bins)
        ax2[0].set_xlabel('Microbial relative abundances')
        ax2[0].set_ylabel('# Microbes in Cluster x\n# Samples Per Microbe', fontsize = 10)
        ax2[0].set_title('Microbes')
        ax[0].set_aspect('equal')
    ax[0].legend()
    ax2[0].legend()
    for i in range(gen_z.shape[1]):
        ix = np.where(gen_z[:, i] == 1)[0]
        p2 = ax[1].scatter(gen_met_locs[ix, 0], gen_met_locs[ix, 1])
        ax[1].set_title('Metabolites')
        # p2 = None
        for ii in ix:
            # p2 = ax[1].scatter(gen_met_locs[ii, 0], gen_met_locs[ii, 1], alpha = )
            ax[1].text(gen_met_locs[ii,0], gen_met_locs[ii,1], 'Metabolite ' + str(ii))
        circle2 = plt.Circle((mu_met[i,0], mu_met[i,1]), r_met[i],
                             alpha=0.2, color=p2.get_facecolor().squeeze(), label = 'Cluster ' + str(i))
        ax[1].add_patch(circle2)
        bins = int((y.max() - y.min())/5)
        if bins<=10:
            bins = 10
        ax2[1].hist(y[:, ix].flatten(), range=(y.min(), y.max()),
                    label='Cluster ' + str(i), alpha=0.5, bins = bins)
        ax2[1].set_xlabel('Standardized metabolite levels')
        ax2[1].set_ylabel('# Metabolites in Cluster x\n# Samples Per Metabolite', fontsize = 10)
        ax2[1].set_title('Metabolites')
        ax[1].set_aspect('equal')
    ax[1].legend()
    ax2[1].legend()
    fig.tight_layout()
    fig.savefig(path + 'embedded_locations.pdf')
    fig2.tight_layout()
    fig2.savefig(path + 'cluster_histogram.pdf')
    plt.close(fig)
    plt.close(fig2)

def plot_distribution(dist, param, ptype = 'init', path = '', **kwargs):
    if 'r' in param:
        vals = 1/dist.sample([500])
        mean = 1/np.round(dist.mean.item(),2)
        std = 1/np.round(dist.stddev.item(),2)
    else:
        vals = dist.sample([500])
        try:
            mean = np.round(dist.mean.item(),2)
            std = np.round(dist.stddev.item(),2)
        except:
            mean, std = 'NA','NA'
    if len(vals.shape)>1:
        fig, ax = plt.subplots(vals.shape[1], figsize = (5, 4*vals.shape[1]))
        for i in np.arange(vals.shape[1]):
            bins = int((vals[:,i].max() - vals[:,i].min()) / 5)
            if bins <= 1:
                bins = 10
            ax[i].hist(vals[:,i], bins=bins)
            ax[i].set_title(param + ', mean=' + str(mean) + ', std=' + str(std))
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('F(x)')
    else:
        fig, ax = plt.subplots()
        bins = int((vals.max() - vals.min()) / 5)
        if bins<=1:
            bins = 10
        ax.hist(vals, bins = bins)
        ax.set_title(param + ', mean=' + str(mean) + ', std=' + str(std))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    r = [k + '_' + str(np.round(item,2)).replace('.','d') for k, item in kwargs.items()]
    if not os.path.isdir(path + '/' + ptype + 's/'):
        os.mkdir(path + '/' + ptype + 's/')
    plt.tight_layout()
    plt.savefig(path + '/' + ptype + 's/' + param + '-' + '-'.join(r))
    plt.close(fig)

def plot_param_traces(path, param_dict, params2learn, true_vals, net, fold):
    fig_dict, ax_dict = {},{}
    for name, plist in param_dict.items():
        if len(plist[0].squeeze().shape)==1:
            fig_dict[name], ax_dict[name] = plt.subplots(plist[0].squeeze().shape[0],
                                                         figsize=(5, 4 * plist[0].squeeze().shape[0]))
        else:
            fig_dict[name], ax_dict[name] = plt.subplots(plist[0].squeeze().shape[0], plist[0].squeeze().shape[1],
                                                         figsize = (5*plist[0].squeeze().shape[1], 4*plist[0].squeeze().shape[0]))
        if name in params2learn or 'all' in params2learn:
            for k in range(plist[0].squeeze().shape[0]):
                if len(plist[0].squeeze().shape) == 1:
                    trace = [p.squeeze()[k] for p in plist]
                    ax_dict[name][k].plot(trace, label='Trace')
                    if not isinstance(true_vals[name], list):
                        true_vals[name] = true_vals[name].squeeze()
                    ax_dict[name][k].plot([true_vals[name][k]] * len(trace), c='r', label='True')
                    ax_dict[name][k].set_title(name + ', ' + str(k))
                    ax_dict[name][k].set_ylim(net.range_dict[name])
                    ax_dict[name][k].legend()
                    ax_dict[name][k].set_xlabel('Iterations')
                    ax_dict[name][k].set_ylabel('Parameter Values')
                else:
                    for j in range(plist[0].squeeze().shape[1]):
                        trace = [p.squeeze()[k, j] for p in plist]
                        ax_dict[name][k, j].plot(trace, label='Trace')
                        ax_dict[name][k, j].plot([true_vals[name][k, j]] * len(trace), c='r', label='True')
                        ax_dict[name][k, j].set_title(name + ', ' + str(k) + ', ' + str(j))
                        ax_dict[name][k, j].set_ylim(net.range_dict[name])
                        ax_dict[name][k, j].legend()
                        ax_dict[name][k, j].set_xlabel('Iterations')
                        ax_dict[name][k, j].set_ylabel('Parameter Values')
            fig_dict[name].tight_layout()
            fig_dict[name].savefig(path + 'fold' + str(fold) + '_' + name + '_parameter_trace.pdf')
            plt.close(fig_dict[name])

def plot_training_output(path, loss, out_vec, targets, gen_z, gen_w, param_dict, fig_dict2, ax_dict2, fig_dict3, ax_dict3, fold):
    best_mod = np.argmin(loss)
    preds = out_vec[best_mod]
    total = np.concatenate([targets, preds.detach().numpy()])

    pred_z = param_dict['z'][best_mod]
    df_dict= {}
    for col in np.arange(pred_z.shape[1]):
        df_dict['Cluster ' + str(col) + ' Prediction'] = pred_z[:,col]
    df_dict['True'] = np.where(gen_z==1)[1]
    index_names = ['Metabolite ' + str(i) for i in np.arange(gen_z.shape[0])]
    df = pd.DataFrame(df_dict, index = index_names)
    df.to_csv(path + 'fold' + str(fold) + '_metab_cluster_fit.csv')

    pred_w = param_dict['w'][best_mod]
    df_dict= {}
    for col in np.arange(pred_w.shape[1]):
        df_dict['Cluster ' + str(col) + ' Prediction'] = pred_w[:,col]
    df_dict['True'] = np.where(gen_w==1)[1]
    index_names = ['Microbe ' + str(i) for i in np.arange(gen_w.shape[0])]
    df = pd.DataFrame(df_dict, index = index_names)
    df.to_csv(path + 'fold' + str(fold) + '_bug_cluster_fit.csv')

    for met in range(targets.shape[1]):
        cluster_id = np.where(gen_z[met, :] == 1)[0]
        range_ratio = (preds[:, met].max() - preds[:, met].min()) / (total.max() - total.min())
        bins = int((total.max() - total.min())/5)
        if bins<10:
            bins = 10
        ax_dict2[fold][met].hist(preds[:, met].detach().numpy(), range=(total.min(), total.max()), label='guess',
                                 alpha=0.5, bins = bins)
        ax_dict2[fold][met].hist(targets[:, met], range=(total.min(), total.max()),
                                 label='true', alpha=0.5, bins = bins)
        ax_dict2[fold][met].set_title('Metabolite ' + str(met) + ', Cluster ' + str(cluster_id))
        ax_dict2[fold][met].legend()
        ax_dict2[fold][met].set_xlabel('Metabolite Levels')
        ax_dict2[fold][met].set_ylabel('# Samples Per Metabolite')
    fig_dict2[fold].tight_layout()
    fig_dict2[fold].savefig(path + 'fold' + str(fold) + '_training_histograms.pdf')
    plt.close(fig_dict2[fold])

    for cluster in range(gen_z.shape[1]):
        met_ids = np.where(gen_z[:, cluster] == 1)[0]
        range_ratio = (preds[:, met_ids].max() - preds[:, met_ids].min()) / (total.max() - total.min())
        bins = int((total.max() - total.min()) / 5)
        if bins<10:
            bins = 10
        ax_dict3[fold][cluster].hist(preds[:, met_ids].flatten().detach().numpy(), range=(total.min(), total.max()),
                                     label='guess', alpha=0.5, bins = bins)
        ax_dict3[fold][cluster].hist(targets[:, met_ids].flatten(), range=(total.min(), total.max()), label='true',
                                     alpha=0.5, bins = bins)
        ax_dict3[fold][cluster].set_title('Cluster ' + str(cluster))
        ax_dict3[fold][cluster].legend()
        ax_dict3[fold][cluster].set_xlabel('Metabolite Levels')
        ax_dict3[fold][cluster].set_ylabel('# Metabolites in Cluster x\n# Samples Per Metabolite')
    fig_dict3[fold].tight_layout()
    fig_dict3[fold].savefig(path + 'fold' + str(fold) + '_training_cluster_histograms.pdf')
    plt.close(fig_dict3[fold])

def plot_output(path, test_loss, test_out_vec, test_targets, gen_z, gen_w, param_dict, fig_dict2, ax_dict2, fig_dict3, ax_dict3, fold):
    best_mod = np.argmin(test_loss)
    preds = test_out_vec[best_mod]
    total = np.concatenate([test_targets, preds])

    pred_z = param_dict['z'][int(best_mod*100)]
    df_dict= {}
    for col in np.arange(pred_z.shape[1]):
        df_dict['Cluster ' + str(col) + ' Prediction'] = pred_z[:,col]
    df_dict['True'] = np.where(gen_z==1)[1]
    index_names = ['Metabolite ' + str(i) for i in np.arange(gen_z.shape[0])]
    df = pd.DataFrame(df_dict, index = index_names)
    df.to_csv(path + 'fold' + str(fold) + '_metab_cluster_predictions.csv')

    pred_w = param_dict['w'][int(best_mod*100)]
    df_dict= {}
    for col in np.arange(pred_w.shape[1]):
        df_dict['Cluster ' + str(col) + ' Prediction'] = pred_w[:,col]
    df_dict['True'] = np.where(gen_w==1)[1]
    index_names = ['Microbe ' + str(i) for i in np.arange(gen_w.shape[0])]
    df = pd.DataFrame(df_dict, index = index_names)
    df.to_csv(path + 'fold' + str(fold) + '_bug_cluster_predictions.csv')

    for met in range(test_targets.shape[1]):
        cluster_id = np.where(gen_z[met, :] == 1)[0]
        range_ratio = (preds[:, met].max() - preds[:, met].min()) / (total.max() - total.min())
        bins = int((total.max() - total.min())/5)
        if bins<10:
            bins = 10
        ax_dict2[fold][met].hist(preds[:, met].detach().numpy(), range=(total.min(), total.max()), label='guess',
                                 alpha=0.5, bins = bins)
        ax_dict2[fold][met].hist(test_targets[:, met], range=(total.min(), total.max()),
                                 label='true', alpha=0.5, bins = bins)
        ax_dict2[fold][met].set_title('Metabolite ' + str(met) + ', Cluster ' + str(cluster_id))
        ax_dict2[fold][met].legend()
        ax_dict2[fold][met].set_xlabel('Metabolite Levels')
        ax_dict2[fold][met].set_ylabel('# Samples Per Metabolite')
    fig_dict2[fold].tight_layout()
    fig_dict2[fold].savefig(path + 'fold' + str(fold) + '_output_histograms.pdf')
    plt.close(fig_dict2[fold])

    for cluster in range(gen_z.shape[1]):
        met_ids = np.where(gen_z[:, cluster] == 1)[0]
        range_ratio = (preds[:, met_ids].max() - preds[:, met_ids].min()) / (total.max() - total.min())
        bins = int((total.max() - total.min()) / 5)
        if bins<10:
            bins = 10
        ax_dict3[fold][cluster].hist(preds[:, met_ids].flatten().detach().numpy(), range=(total.min(), total.max()),
                                     label='guess', alpha=0.5, bins = bins)
        ax_dict3[fold][cluster].hist(test_targets[:, met_ids].flatten(), range=(total.min(), total.max()), label='true',
                                     alpha=0.5, bins = bins)
        ax_dict3[fold][cluster].set_title('Cluster ' + str(cluster))
        ax_dict3[fold][cluster].legend()
        ax_dict3[fold][cluster].set_xlabel('Metabolite Levels')
        ax_dict3[fold][cluster].set_ylabel('# Metabolites in Cluster x\n# Samples Per Metabolite')
    fig_dict3[fold].tight_layout()
    fig_dict3[fold].savefig(path + 'fold' + str(fold) + '_output_cluster_histograms.pdf')
    plt.close(fig_dict3[fold])

def plot_loss(fig3, ax3, fold, iterations, loss_vec, test_loss):
    ax3.set_title('Fold ' + str(fold))
    ax3.plot(np.arange(iterations), loss_vec, label='training loss')
    if iterations%100 != 1:
        xvals = np.append(np.arange(0, iterations, 100), iterations)
    else:
        xvals = np.arange(0, iterations, 100)
    ax3.plot(xvals, test_loss, label='test loss')
    ax3.legend()
    return fig3, ax3