import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_covariances(p_init, r):
    grid = plt.GridSpec(ncols=2, nrows=2)

    plt.subplot(grid[0, :])
    im = plt.imshow(p_init, interpolation="none", cmap=plt.get_cmap('binary'))
    plt.title('Initial Covariance Matrix $P$')
    # set the locations of the yticks
    plt.yticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.yticks(np.arange(6), ('$x$', '$\dot x$', '$y$', '$\dot y$'), fontsize=22)

    # set the locations of the yticks
    plt.xticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.xticks(np.arange(6), ('$x$', '$\dot x$', '$y$', '$\dot y$'), fontsize=22)

    plt.xlim([-0.5, 3.5])
    plt.ylim([3.5, -0.5])

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(im, cax=cax)

    xpdf = np.arange(-1, 1, 0.001)
    plt.subplot(grid[1, 0])
    plt.plot(xpdf, norm.pdf(xpdf, 0, r[0, 0]))
    plt.title('x observation noise distribution')

    plt.subplot(grid[1, 1])
    plt.plot(xpdf, norm.pdf(xpdf, 0, r[1, 1]))
    plt.title('y observation noise distribution')
    plt.tight_layout()

    plt.show()
