import pandas as pd

from histograms_of_observations import *
from plots import *
from save_vectors import *
from tracking_evaluation import *

X = np.array(pd.read_csv('Data/x.csv'))
Y = np.array(pd.read_csv('Data/y.csv'))
X = np.reshape(X, [len(X), ])
Y = np.reshape(Y, [len(Y), ])

a = np.array(pd.read_csv('Data/a.csv'))
b = np.array(pd.read_csv('Data/b.csv'))
a = np.reshape(a, [len(a), ])
b = np.reshape(b, [len(a), ])
measurements = np.stack((a, b), axis=0)
gate_arr = []

# Initial state initialisation
# state_vector = [x, vx, y, vy]
x = np.transpose(np.array([[0.0, 0.0, 0.0, 0.0]]))

# P_init = Q
P = np.diag([0.1, 0.2, 0.1, 0.2])

dt = 0.1  # Time Step between Filter Steps
# Matrix of motion model
F = np.array([[1.0, dt, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, dt],
              [0.0, 0.0, 0.0, 1.0]])

std_x, std_vx = 0.1, 0.1
std_y, std_vy = 0.2, 0.2
# Covariance of (motion) noise
Q = np.array([[std_x, 0.0, 0.0, 0.0],
              [0.0, std_y, 0.0, 0.0],
              [0.0, 0.0, std_vx, 0.0],
              [0.0, 0.0, 0.0, std_vy]])

# Observation model assuming we get measurements from positions
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

var_u, var_v = 0.3, 0.3
R = np.array([[var_u, 0.0],
              [0.0, var_v]])

# Plot covariance Matrix
# plot_covariances(P, R)

for n in range(len(measurements[0])):
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    x = np.dot(F, x)

    # Project the error covariance ahead
    P = np.dot(np.dot(F, P), np.transpose(F)) + Q

    # Measurement Update (Correction)
    # ===============================
    # Compute the Kalman Gain and Innovation matrix
    S = np.dot(np.dot(H, P), np.transpose(H)) + R
    K = np.dot(np.dot(P, np.transpose(H)), np.linalg.inv(S))

    # Testing filtering with no measurments for steps 40 to 60.
    if 40 < n < 60 and False:
        Z = np.array([0, 0]).reshape(2, 1)
    else:
        # Update the estimate via z
        Z = measurements[:, n].reshape(2, 1)

    Z_pred = Z - np.dot(H, x)

    gate = np.dot(np.dot(np.transpose(Z_pred), np.linalg.inv(S)), Z_pred)

    gate_arr.append(gate)
    if gate > 9.21 and True:
        print('Observation outside validation gate!', n - 1)
        x = x
        P = P
    else:
        # Update the states
        x = x + np.dot(K, Z_pred)
        # Update the error covariance
        P = P - np.dot(np.dot(K, S), np.transpose(K))

    # Save states (for Plotting)
    append_states(x, Z, P, R, K)

# Plots
plot_xy(a, b, X, Y, x_pred, y_pred)
plot_x(measurements, vx_pred, vy_pred)
plot_k(measurements, Kx, Ky, Kvx, Kvy)
plot_p(measurements, Px, Py, Pvx, Pvy)

# Metrics
absolute_error(x_pred, X, y_pred, Y, show_plot=False)
absolute_error(a, X, b, Y, show_plot=False)

# Observation variances
obs_analysis(a, X)
