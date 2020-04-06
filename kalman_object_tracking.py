import numpy as np
import pandas as pd
from plots import *

X = np.array(pd.read_csv('x.csv'))
Y = np.array(pd.read_csv('y.csv'))
a = np.array(pd.read_csv('a.csv'))
b = np.array(pd.read_csv('b.csv'))
a = np.reshape(a, [len(a), ])
b = np.reshape(b, [len(a), ])
measurements = np.stack((a, b), axis=0)

# Initial state initialisation
# state_vector = [xk, vxk, yk, vyk]
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
# Matrix of motion noise
Q = np.array([[std_x, 0.0, 0.0, 0.0],
              [0.0, std_y, 0.0, 0.0],
              [0.0, 0.0, std_vx, 0.0],
              [0.0, 0.0, 0.0, std_vy]])

# Observation model assuming we get measurements from speed
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

std_u, std_v = 0.3, 0.3
R = np.array([[std_u, 0.0],
              [0.0, std_v]])

I = np.eye(4)

# Plot covariance Matrix
plot_covariances(P,R)

