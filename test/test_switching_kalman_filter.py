import csv, time
import numpy as np

from utils.kalman import SwitchingKalmanState, SwitchingKalmanFilter
from utils.kalman.models import CWPA2D, Brownian2D
from utils.helpers import *

import matplotlib.pyplot as plt

# positions = load_trajectory(1080, 100)
# positions = load_trajectory(1462, 79)
positions = load_trajectory(3331, 100)
# positions = load_random_trajectory()
n = positions.shape[0]

models = [
    CWPA2D(dt=1.0, q=2e-1, r=10.0),
    Brownian2D(dt=1.0, q=2e-1, r=10.0)
]
Z = np.log(np.asarray([
    [0.99, 0.01],
    [0.01, 0.99]
]))

kalman = SwitchingKalmanFilter(n_obs=2, n_hid=6, models=models, log_transmat=Z)


start_time = time.time()
# Kalman Filter
state = SwitchingKalmanState(mean=np.zeros((6,2)), covariance=np.zeros((6,6,2)))
state.P[:,:,0] = 10.0 * np.eye(6)
state.P[:,:,1] = 10.0 * np.eye(6)
state.M = np.ones(2) / 2.0

filtered_states = [state] * n
for i in xrange(n):
    observation = positions[i]
    state = kalman.filter(state, observation)
    filtered_states[i] = state

smoothed_states = [state] * n
for i in xrange(1, n):
    j = n - 1 - i
    state = kalman.smoother(state, filtered_states[j])
    smoothed_states[j] = state

print '---- %.3f seconds ----' % (time.time() - start_time,)

# output_states = filtered_states
output_states = smoothed_states

smoothed = np.asarray(map(lambda state: np.dot(state.m[0:2,:], np.exp(state.M)), output_states))
stops = np.asarray(map(lambda state: np.exp(state.M[1]), output_states))

subplot_shape = (2,2)
plt.subplot2grid(subplot_shape, (0,0), rowspan=2)
plt.plot(positions[:,0], positions[:,1], 'b-')
plt.plot(smoothed[:,0], smoothed[:,1], 'g-')
plt.plot(smoothed[stops>0.50,0], smoothed[stops>0.50,1], 'ro')

plt.subplot2grid(subplot_shape, (0,1))
plt.plot(range(n), stops)
plt.plot(range(n), 0.5 * np.ones(n), 'r--')

plt.subplot2grid(subplot_shape, (1,1))
velocity = np.asarray(map(lambda state: np.dot(state.m[2:4], np.exp(state.M)), output_states))
plt.plot(range(n), map(lambda v: 3.6 * np.linalg.norm(v), velocity))

plt.show()