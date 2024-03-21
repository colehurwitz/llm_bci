import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering


"""
This script generates a plot showing the (single-trial) fitting of a single neuron.
:X: behavior matrix of the shape [n_trials, n_timesteps, n_variables]. 
:y: true neural activity matrix of the shape [n_trials, n_timesteps] 
:ypred: predicted activity matrix of the shape [n_trials, n_timesteps] 
:var_name2idx: dictionary mapping feature names to their corresponding index of the 3-rd axis of the behavior matrix X. e.g.: {"choice": [0], "wheel": [1]}
:var_tasklist: *static* task variables used to form the task condition and compute the psth. e.g.: ["choice"]
:var_value2label: dictionary mapping values in X to their corresponding readable labels (only required for static task variables). e.g.: {"choice": {1.: "left", -1.: "right"}}
:subtract_psth: 
    - None: no subtraction
    - "task": subtract task-condition-averaged psth
    - "global": subtract global-averaged psth
:algined_tbins: reference time steps to annotate in the plot. 
"""
def viz_single_cell(X,y,y_pred, var_name2idx, var_tasklist, var_value2label,
                    subtract_psth="task", aligned_tbins=[], save_name="plot.png"):
    nrows = 8
    plt.figure(figsize=(8,2*nrows))
    
    ### plot psth
    axes_psth = [plt.subplot(nrows,4,k) for k in range(1,5)]
    plot_psth(X, y, y_pred, 
              var_tasklist=var_tasklist, 
              var_name2idx=var_name2idx, 
              var_value2label=var_value2label,
              aligned_tbins=aligned_tbins,
              axes=axes_psth, legend=True)
    
    ### plot the psth-subtracted activity
    axes_single = [plt.subplot(nrows,1,k) for k in range(2,2+2+2)]
    plot_single_trial_activity(X, y, y_pred,
                               var_name2idx,
                               var_tasklist, subtract_psth=subtract_psth,
                               aligned_tbins=aligned_tbins,
                               axes=axes_single)

    plt.tight_layout()
    plt.savefig(save_name)

"""
:X: [n_trials, n_timesteps, n_variables]
:y: [n_trials, n_timesteps] 
:y_pred: [n_trials, n_timesteps] 
:var_tasklist: for each task variable in var_tasklists, compute PSTH
:var_name2idx: for each task variable in var_tasklists, the corresponding index of X
:var_value2label:
:aligned_tbins: reference time steps to annotate. 
"""
def plot_psth(X, y, y_pred, var_tasklist, var_name2idx, var_value2label,
              aligned_tbins=[],
              axes=None, legend=False):
    if axes is None:
        nrows = 1; ncols = len(var_tasklist)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))

    for ci, var in enumerate(var_tasklist):
        ax = axes[ci]
        _var_unique = np.unique(X[:, 0, var_name2idx[var]], axis=0)
        for _i, _x in enumerate(_var_unique):
            psth = compute_PSTH(X, y, axis=var_name2idx[var], value=_x)
            psth_pred = compute_PSTH(X, y_pred, axis=var_name2idx[var_tasklist[ci]], value=_x)
            ax.plot(psth,
                    color=plt.get_cmap('tab10')(_i),
                    linewidth=3, alpha=0.3, label=f"{var_value2label[var][tuple(_x)]}")
            ax.plot(psth_pred,
                    color = plt.get_cmap('tab10')(_i),
                    linestyle='--')
            ax.set_xlabel("Time bin")
            if ci == 0:
                ax.set_ylabel("Neural activity")
            else:
                ax.sharey(axes[0])
        _add_baseline(ax, aligned_tbins=aligned_tbins)
        if legend:
            ax.legend()
            ax.set_title(f"{var}")
    for ax in axes:
        ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()


"""
:X: [n_trials, n_timesteps, n_variables]
:y: [n_trials, n_timesteps] 
:y_pred: [n_trials, n_timesteps] 
:var_tasklist: variables used for computing the task-condition-averaged psth if subtract_psth=='task'
:var_name2idx:
:var_tasklist: variables to be plotted in the single-trial behavior
:subtract_psth: 
    - None: no subtraction
    - "task": subtract task-condition-averaged psth
    - "global": subtract global-averaged psth
:aligned_tbins: reference time steps to annotate. 
:nclus, n_neighbors: hyperparameters for spectral_clustering
:cmap, vmax_perc, vmin_perc: parameters used when plotting the activity and behavior
"""
def plot_single_trial_activity(X, y, y_pred,
                               var_name2idx,
                               var_tasklist, subtract_psth="task",
                               aligned_tbins=[],
                               n_clus=8, n_neighbors=5,
                               cmap='bwr', vmax_perc=90, vmin_perc=10,
                               axes=None):
    if axes is None:
        nrows = 1; ncols = 2+len(var_name2idx)+1+1
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))

    ### get the psth-subtracted y
    if subtract_psth is None:
        pass
    elif subtract_psth == "task":
        idxs_psth = np.concatenate([var_name2idx[var] for var in var_tasklist])
        uni_vs = np.unique(X[:, 0, idxs_psth], axis=0)  # get all the unique task-conditions
        psth_vs = {};
        psth_pred_vs = {}  # make dictionary with task-conditions as keys and psth as values
        for v in uni_vs:
            # compute separately for true y and predicted y
            _psth = compute_PSTH(X, y,
                                 axis=idxs_psth, value=v)  # (T)
            psth_vs[tuple(v)] = _psth
            _psth_pred = compute_PSTH(X, y_pred,
                                      axis=idxs_psth, value=v)  # (T)
            psth_pred_vs[tuple(v)] = _psth_pred
        y_predpsth = np.asarray(
            [psth_vs[tuple(x)] for x in X[:, 0, idxs_psth]])  # (K, T) predict the neural activity with psth
        y = y - y_predpsth  # (K, T)
        y_pred = y_pred - y_predpsth  # (K, T)
    elif subtract_psth == "global":
        y_predpsth = np.mean(y, 0)
        y -= y_predpsth  # (K, T)
        y_pred -= y_predpsth  # (K, T)
    else:
        assert False, "Unknown subtract_psth, has to be one of: task, global. \'\'"
    y_residual = (y_pred - y)  # (K, T), residuals of prediction

    ### plot single-trial activity
    # arange the trials by unsupervised clustering labels
    clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                    affinity='nearest_neighbors',
                                    assign_labels='discretize',
                                    random_state=0).fit(y_pred)
    t_sort = np.argsort(clustering.labels_)

    for ri, (toshow, label, ax) in enumerate(zip([y, y_pred, y_residual],
                                                 [f"obs. act. \n (subtract_psth={subtract_psth})",
                                                  f"pred. act. \n (subtract_psth={subtract_psth})",
                                                  "residual act."],
                                                 [axes[0], axes[1], axes[-2]])):
        if ri <= 1:
            # plot obs./ predicted activity
            vmax = np.percentile(y_pred, vmax_perc)
            vmin = np.percentile(y_pred, vmin_perc)
            raster_plot(toshow[t_sort], vmax, vmin, True, label, ax,
                        cmap=cmap,
                        aligned_tbins=aligned_tbins)
        elif ri == 2:
            # plot residual activity
            vmax = np.percentile(toshow, vmax_perc)
            vmin = np.percentile(toshow, vmin_perc)
            raster_plot(toshow[t_sort], vmax, vmin, True, label, ax,
                        cmap=cmap,
                        aligned_tbins=aligned_tbins)

    ### plot single-trial activity
    # re-arrange the trials
    clustering = SpectralClustering(n_clusters=n_clus,n_neighbors=n_neighbors,
                                    affinity='nearest_neighbors',
                                    assign_labels='discretize',
                                    random_state=0).fit(y_residual)
    t_sort_rd = np.argsort(clustering.labels_)
    raster_plot(y_residual[t_sort_rd], np.percentile(y_residual, vmax_perc), np.percentile(y_residual, vmin_perc), True, "residual act. (re-clustered)", axes[-1])

    plt.tight_layout()


def _add_baseline(ax, aligned_tbins=[40]):
    for tbin in aligned_tbins:
        ax.axvline(x=tbin-1, c='k', alpha=0.2)
    # ax.axhline(y=0., c='k', alpha=0.2)


def raster_plot(ts_, vmax, vmin, whether_cbar, ylabel, ax,
                cmap='bwr',
                aligned_tbins=[40]):
    N, T = ts_.shape
    im = ax.imshow(ts_, aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin)
    for tbin in aligned_tbins:
        ax.annotate('',
            xy=(tbin-1, N),
            xytext=(tbin-1, N+10),
            ha='center',
            va='center',
            arrowprops={'arrowstyle': '->', 'color': 'r'})
    if whether_cbar:
        cbar = plt.colorbar(im, pad=0.01, shrink=.6)
        cbar.ax.tick_params(rotation=90)
    if not (ylabel is None):
        ax.set_ylabel(f"{ylabel}"+f"\n(#trials={N})")
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.spines[['left','bottom', 'right', 'top']].set_visible(False)
        pass
    else:
        ax.axis('off')

"""
- X, y should be nparray with
    - X: [K,T,ncoef]
    - y: [K,T,N] or [K,T]
- axis and value should be list
- return: nparray [T, N] or [T]
"""
def compute_PSTH(X, y, axis, value):
    trials = np.all(X[:, 0, axis] == value, axis=-1)
    return y[trials].mean(0)




# _temp = np.load("example_X_y_ypred.npz", allow_pickle=True)
# X = _temp["X"] # [#trials, #timesteps, #variables]
# ys = _temp["ys"] # [#trials, #timesteps, #neurons]
# y_preds = _temp["y_preds"] # [#trials, #timesteps, #neurons]


# var_name2idx = {'choice': [0]}
        

# var_value2label = {'choice': {(-1.0,): "right", (1.0,): "left"}}

# var_tasklist = ['choice']

# for ni in range(ys.shape[-1]):
#     viz_single_cell(X,ys[:,:,ni],y_preds[:,:,ni], 
#                     var_name2idx, var_tasklist, var_value2label, var_behlist,
#                     subtract_psth="task", aligned_tbins=[40])