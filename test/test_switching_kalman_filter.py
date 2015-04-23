import csv, time
import numpy as np

from utils.kalman import SwitchingKalmanState, SwitchingKalmanFilter, KalmanState
from utils.kalman.models import NDCWPA, NDBrownian
from utils.helpers import *

import matplotlib.pyplot as plt

# positions = load_trajectory(1080, 100)
# positions = load_trajectory(1462, 79)
positions = load_trajectory(3331, 100)
# positions = load_random_trajectory()
n = positions.shape[0]

models = [
    NDCWPA(dt=1.0, q=2e-1, r=10.0, n_dim=2),
    NDBrownian(dt=1.0, q=2e-1, r=10.0, n_dim=2)
]
Z = np.log(np.asarray([
    [0.99, 0.01],
    [0.01, 0.99]
]))
masks = [
    np.asarray([
        np.diag([1, 0, 1, 0, 1, 0]),
        np.diag([0, 1, 0, 1, 0, 1])
    ]),
    np.asarray([
        np.diag([1, 0]),
        np.diag([0, 1])
    ])
]

T = np.asarray([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]
])
embeds = [
    [np.eye(6), T],
    [T.T, np.eye(2)]
]

kalman = SwitchingKalmanFilter(models=models, log_transmat=Z, masks=masks, embeds=embeds)


start_time = time.time()
# Kalman Filter
state = SwitchingKalmanState(n_models=2)
state._states[0] = KalmanState(mean=np.zeros(6), covariance=10.0 * np.eye(6))
state._states[1] = KalmanState(mean=np.zeros(2), covariance=10.0 * np.eye(2))
state.M = np.ones(2) / 2.0

filtered_states = [state] * n
for i in xrange(n):
    observation = positions[i]
    state = kalman.filter(state, observation)
    filtered_states[i] = state

# smoothed_states = [state] * n
# for i in xrange(1, n):
#     j = n - 1 - i
#     state = kalman.smoother(state, filtered_states[j])
#     smoothed_states[j] = state

print '---- %.3f seconds ----' % (time.time() - start_time,)

output_states = filtered_states
# output_states = smoothed_states

# smoothed = np.asarray(map(lambda state: np.dot(state.m[0:2,:], np.exp(state.M)), output_states))
embeds_f = [np.eye(6), T.T]
smoothed_collapsed = map(lambda state: state.collapse(embeds_f), output_states)
smoothed = np.asarray(map(lambda state: state.m, smoothed_collapsed))
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
velocity = np.asarray(map(lambda state: state.m[2:4], smoothed_collapsed))
plt.plot(range(n), map(lambda v: 3.6 * np.linalg.norm(v), velocity))

# for state in output_states:
#     # print state._states[1].P[0:2,0:2]
#     # print np.linalg.det(state.collapse(embeds_f).P)
#     print state.collapse(embeds_f).P[0:2,0:2]

plt.show()