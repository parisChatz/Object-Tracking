import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm


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


def plot_xy(x_noise, y_noise, x_real, y_real, x_predicted=None, y_predicted=None):
    plt.figure(1)
    noisy = plt.scatter(x_noise, y_noise, label='Noisy Signal', c='red')
    realy, = plt.plot(x_real, y_real, label='Real Signal', c='green', linewidth=5)
    predicty, = plt.plot(x_predicted, y_predicted, label='Estimated Signal', linestyle='dashed', c='black', linewidth=2)
    plt.legend(handles=[noisy, realy, predicty])
    plt.show()


def plot_k(measurements, Kx, Ky, Kdx, Kdy):
    plt.plot(range(len(measurements[0])), Kx, label='Kalman Gain for $x$', linewidth=4)
    plt.plot(range(len(measurements[0])), Ky, label='Kalman Gain for $y$', linewidth=8)
    plt.plot(range(len(measurements[0])), Kdx, label='Kalman Gain for $\dot x$', linewidth=4)
    plt.plot(range(len(measurements[0])), Kdy, label='Kalman Gain for $\dot y$', linewidth=4)

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.title('Kalman Gain')  # (the lower, the more the measurement fullfill the prediction)
    plt.legend(loc='best', prop={'size': 22})
    plt.show()


def plot_p(measurements, Px, Py, Pdx, Pdy):
    plt.plot(range(len(measurements[0])), Px, label='$x$', linewidth=5)
    plt.plot(range(len(measurements[0])), Py, label='$y$', linewidth=2)
    plt.plot(range(len(measurements[0])), Pdx, label='$\dot x$', linewidth=5)
    plt.plot(range(len(measurements[0])), Pdy, label='$\dot y$', linewidth=2)

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.title('Uncertainty (Elements from Matrix $P$)')
    plt.legend(loc='best', prop={'size': 22})
    plt.show()


def plot_x(measurements, dxt, dyt):
    plt.plot(range(len(measurements[0])), dxt, label='$\dot x$')
    plt.plot(range(len(measurements[0])), dyt, label='$\dot y$')

    # plt.axhline(5, color='#999999', label='$\dot x_{real}$')
    # plt.axhline(5, color='#999999', label='$\dot y_{real}$')

    plt.xlabel('Filter Step')
    plt.title('Estimate (Elements from State Vector $x$)')
    plt.legend(loc='best', prop={'size': 22})
    plt.ylim([0, 15])
    plt.ylabel('Velocity')
    plt.show()
