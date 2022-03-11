import numpy as np

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils.kalman import KalmanFilter, KalmanState, KalmanObservation
from utils.kalman.models import NDCWPA
from utils.helpers import *

import matplotlib.pyplot as plt

positions = load_trajectory(1, 1)
n = positions.shape[0]

# Kalman Filter
state = KalmanState(mean=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                    covariance=10.0 * np.eye(6))
model = NDCWPA(dt=1.0, q=2e-2, r=10.0, n_dim=2)
kalman = KalmanFilter(model=model)

filtered_states = [state] * n
for i in range(n):
    observation = positions[i]
    state = kalman.filter(state, observation)
    filtered_states[i] = state

smoothed_states = [state] * n
for i in range(1, n):
    j = n - 1 - i
    state = kalman.smoother(filtered_states[j], state)
    smoothed_states[j] = state

# Visualization
subplot_shape = (3, 2)

# smoothed = np.asarray(map(lambda state: state.x(), smoothed_states))
smoothed = np.asarray([state.x() for x in smoothed_states])
print("smoothed: ", smoothed)
ax1 = plt.subplot2grid(subplot_shape, (0, 0), rowspan=3)
ax1.plot(positions[:,0], positions[:,1])
ax1.plot(smoothed[:,0], smoothed[:,1])
i = 0
markers = np.zeros((int(n / 100), 2))
print("n:", n)
for position in smoothed_states:
    mean = position.m
    if i % 5 == 0:
        ax1.plot([mean[0], mean[0] + mean[2]], [mean[1], mean[1] + mean[3]], 'r-', lw=2)
    if i > 0 and i % 100 == 0:
        markers[int(i / 100) - 1,:] = mean[0:2]
    i += 1
ax1.plot(markers[:,0], markers[:,1], 'ko')

ax2 = plt.subplot2grid(subplot_shape, (0, 1))
speed = np.asarray([state.dx() for state in smoothed_states])
speed_norm = speed.T / np.linalg.norm(speed, axis=1)
speed_norm = speed_norm.T
accl = np.asarray([state.ddx() for state in smoothed_states])
print("speed norm shape: ", speed_norm.shape)
print("accl shape: ", accl.shape)
speed_and_accl = np.hstack((speed_norm, accl))

ax2.plot(range(n), [np.vdot(x[0:2], x[2:4]) for x in speed_and_accl], color='black')

ax3 = plt.subplot2grid(subplot_shape, (1, 1))
covar_speed = np.array([np.diag(state.P[2:4, 2:4]) for state in smoothed_states])
cov = np.linalg.norm(covar_speed, axis=1)
norm_speed = np.linalg.norm(speed, axis=1)
# ax3.fill_between(range(n), np.linalg.norm(speed - covar_speed, axis=1), np.linalg.norm(speed + covar_speed, axis=1), facecolor='0.80', interpolate=True, linewidth=0)
# ax3.fill_between(range(n), norm_speed - cov, norm_speed + cov, facecolor='0.80', interpolate=True, linewidth=0)
ax3.plot(range(n), norm_speed, color='black')

ax4 = plt.subplot2grid(subplot_shape, (2, 1))
directions = np.arccos(speed_norm[:-1,0]) - np.arccos(speed_norm[1:,0])
ax4.plot(range(n - 1), directions, color='black')

plt.show()