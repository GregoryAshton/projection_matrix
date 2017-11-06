import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.misc import logsumexp


def log_mean(loga, axis):
    """ Calculate the log(<a>) mean

    Given `N` logged value `log`, calculate the log_mean
    `log(<loga>)=log(sum(np.exp(loga))) - log(N)`. Useful for marginalizing
    over logged likelihoods for example.

    Parameters
    ----------
    loga: array_like
        Input_array.
    axies: None or int or type of ints, optional
        Axis or axes over which the sum is taken. By default axis is None, and
        all elements are summed.
    Returns
    -------
    log_mean: ndarry
        The logged average value (shape loga.shape)
    """
    loga = np.array(loga)
    N = np.prod([loga.shape[i] for i in axis])
    return logsumexp(loga, axis) - np.log(N)


def max_slice(D, axis):
    """ Return the slice along the given axis """
    idxs = [range(D.shape[j]) for j in range(D.ndim)]
    max_idx = list(np.unravel_index(D.argmax(), D.shape))
    for k in np.atleast_1d(axis):
        idxs[k] = [max_idx[k]]
    res = np.squeeze(D[np.ix_(*tuple(idxs))])
    return res


def idx_array_slice(D, axis, slice_idx):
    """ Return the slice along the given axis """
    idxs = [range(D.shape[j]) for j in range(D.ndim)]
    for k in np.atleast_1d(axis):
        idxs[k] = [slice_idx[k]]
    res = np.squeeze(D[np.ix_(*tuple(idxs))])
    return res


def projection_matrix(D, xyz, labels=None, projection='max_slice',
                      max_n_ticks=4, factor=3, whspace=0.05, **kwargs):
    """ Generate a projection matrix plot

    Parameters
    ----------
    D: array_like
        N-dimensional data to plot, `D.shape` should be  `(n1, n2,..., nn)`,
        where `ni`, is the number of grid points along dimension `i`.
    xyz: list
        List of 1-dimensional arrays of coordinates. `xyz[i]` should have
        length `ni` (see help for `D`).
    labels: list
        N+1 length list of labels; the first N correspond to the coordinates
        labels, the final label is for the dependent variable.
    projection: str or func
        If a string, one of `{"log_mean", "max_slice"} to use inbuilt functions
        to calculate either the logged mean or maximum slice projection. Else
        a function to use for projection, must take an `axis` argument. Default
        is `projection_matrix.max_slice()`, to project out a slice along the
        maximum.
    max_n_ticks: int
        Number of ticks for x and y axis of the `pcolormesh` plots
    factor: float
        Controls the size of one window

    Returns
    -------
    fig, axes:
        The figure and NxN set of axes

    """
    ndim = D.ndim
    lbdim = 0.4 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    plotdim = factor * ndim + factor * (ndim - 1.) * whspace
    dim = lbdim + plotdim + trdim

    if len(labels) == ndim:
        labels[-1] = ''

    if type(projection) == str:
        if projection in ['log_mean']:
            projection = log_mean
        elif projection in ['max_slice']:
            projection = max_slice
        else:
            raise ValueError("Projection {} not understood".format(projection))

    fig, axes = plt.subplots(ndim, ndim, figsize=(dim, dim))

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=0.98*tr, top=tr,
                        wspace=whspace, hspace=whspace)
    for i in range(ndim):
        projection_1D(
            axes[i, i], xyz[i], D, i, projection=projection, **kwargs)
        for j in range(ndim):
            ax = axes[i, j]

            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ax.get_shared_x_axes().join(axes[ndim-1, j], ax)
            if i < ndim - 1:
                ax.set_xticklabels([])
            if j < i:
                ax.get_shared_y_axes().join(axes[i, i-1], ax)
                if j > 0:
                    ax.set_yticklabels([])
            if j == i:
                continue

            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="upper"))
            ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="upper"))

            ax, pax = projection_2D(ax, xyz[i], xyz[j], D, i, j,
                                    projection=projection, **kwargs)

    if labels:
        for i in range(ndim):
            axes[-1, i].set_xlabel(labels[i])
            if i > 0:
                axes[i, 0].set_ylabel(labels[i])
            axes[i, i].set_ylabel(labels[-1])
    return fig, axes


def projection_2D(ax, x, y, D, xidx, yidx, projection, **kwargs):
    flat_idxs = range(D.ndim)
    flat_idxs.remove(xidx)
    flat_idxs.remove(yidx)
    D2D = projection(D, axis=tuple(flat_idxs), **kwargs)
    X, Y = np.meshgrid(x, y, indexing='ij')
    pax = ax.pcolormesh(Y, X, D2D.T, vmin=D.min(), vmax=D.max())
    return ax, pax


def projection_1D(ax, x, D, xidx, projection, **kwargs):
    flat_idxs = range(D.ndim)
    flat_idxs.remove(xidx)
    D1D = projection(D, axis=tuple(flat_idxs), **kwargs)
    ax.plot(x, D1D)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    return ax



