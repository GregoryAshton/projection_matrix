import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def projection_matrix(D, xyz, labels=None, projection=np.max, max_n_ticks=4,
                      factor=3):
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
    projection: func
        Function to use for projection, must take an `axis` argument. Default
        is `np.max()`, to project out a slice along the maximum.
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
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * ndim + factor * (ndim - 1.) * whspace
    dim = lbdim + plotdim + trdim

    fig, axes = plt.subplots(ndim, ndim, figsize=(dim, dim))

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)
    for i in range(ndim):
        projection_1D(axes[i, i], xyz[i], D, i, projection=projection)
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
                                    projection=projection)

    if labels:
        for i in range(ndim):
            axes[-1, i].set_xlabel(labels[i])
            if i > 0:
                axes[i, 0].set_ylabel(labels[i])
            axes[i, i].set_ylabel(labels[-1])
    return fig, axes


def projection_2D(ax, x, y, D, xidx, yidx, projection):
    flat_idxs = range(D.ndim)
    flat_idxs.remove(xidx)
    flat_idxs.remove(yidx)
    D2D = projection(D, axis=tuple(flat_idxs))
    X, Y = np.meshgrid(x, y, indexing='ij')
    pax = ax.pcolormesh(Y, X, D2D.T, vmin=D.min(), vmax=D.max())
    return ax, pax


def projection_1D(ax, x, D, xidx, projection):
    flat_idxs = range(D.ndim)
    flat_idxs.remove(xidx)
    D1D = projection(D, axis=tuple(flat_idxs))
    ax.plot(x, D1D)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    return ax
