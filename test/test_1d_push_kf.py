import numpy as np

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils.kalman import SwitchingKalmanState, SwitchingKalmanFilter, KalmanFilter, KalmanState
from utils.kalman.models import NDCWPA, NDBrownian, RandAcc

import matplotlib.pyplot as plt

# Generate toyexample data
K_pos = 7.0

n_pts = 1000
pos_min, pos_max = 0.1, 20.0
dx = (pos_max-pos_min)/n_pts

pos_ary = np.linspace(pos_min, pos_max, n_pts)
f_ary = 1000*(pos_ary-K_pos)
f_ary[f_ary<0] = 0

f_noise = f_ary + 5*np.random.randn(*f_ary.shape)

# evaluate single Kalman filter
model_name = 'RandAcc'
if model_name == 'NDCWPA':
    state = KalmanState(mean=np.zeros(3), covariance=1.0 * np.eye(3), ord=3)
    model = NDCWPA(dt=dx, q=2e-2, r=10.0, n_dim=1)
elif model_name == 'RandAcc': 
    state = KalmanState(mean=np.zeros(2), covariance=1.0 * np.eye(2), ord=2)
    model = RandAcc(dt=dx, q=2e-2, r=10.0)
kalman = KalmanFilter(model=model)

filtered_states_kf = [state] * n_pts
for i in range(n_pts):
    observation = f_noise[i]
    state = kalman.filter(state, observation)
    filtered_states_kf[i] = state

smoothed_states_kf = [state] * n_pts
for i in range(1, n_pts):
    j = n_pts - 1 - i
    state = kalman.smoother(filtered_states_kf[j], state)
    smoothed_states_kf[j] = state

filtered_kf = np.asarray([state.x()[0] for state in filtered_states_kf])
filtered_df_kf = np.asarray([state.x()[1] for state in filtered_states_kf])
smoothed_kf = np.asarray([state.x()[0] for state in smoothed_states_kf])

print("smoothed kf shape:", smoothed_kf.shape)

np.save("f_ary.npy", f_ary)
np.save("pos_ary.npy", pos_ary)
np.save(model_name+"_f.npy", filtered_kf)
np.save(model_name+"_K.npy", filtered_df_kf)

sys.exit(0)
# evaluate switching Kalman filter
models = [
    NDCWPA(dt=1.0, q=2e-2, r=10.0, n_dim=2),
    NDBrownian(dt=1.0, q=2e-2, r=10.0, n_dim=2)
]
Z = np.log(np.asarray([
    [0.99, 0.01],
    [0.01, 0.99]
]))
masks = [
    np.array([
        np.diag([1, 0, 1, 0, 1, 0]),
        np.diag([0, 1, 0, 1, 0, 1])
    ]),
    np.array([
        np.diag([1, 0]),
        np.diag([0, 1])
    ])
]

T = np.kron(np.array([1, 0, 0]), np.eye(2))
embeds = [
    [np.eye(6), T],
    [T.T, np.eye(2)]
]

kalman = SwitchingKalmanFilter(models=models, log_transmat=Z, masks=masks, embeds=embeds)

state = SwitchingKalmanState(n_models=2)
state._states[0] = KalmanState(mean=np.zeros(6), covariance=10.0 * np.eye(6))
state._states[1] = KalmanState(mean=np.zeros(2), covariance=10.0 * np.eye(2))
state.M = np.ones(2) / 2.0

filtered_states_skf = [state] * n
for i in range(n):
    observation = positions[i]
    state = kalman.filter(state, observation)
    filtered_states_skf[i] = state

smoothed_states_skf = [state] * n
for i in range(1, n):
    j = n - 1 - i
    state = kalman.smoother(state, filtered_states_skf[j])
    smoothed_states_skf[j] = state

display_smoothed = True
if display_smoothed:
    output_states_skf = smoothed_states_skf
    output_states_kf = smoothed_states_kf
else:
    output_states_skf = filtered_states_skf
    output_states_kf = filtered_states_kf

smoothed_collapsed = [ state.collapse([np.eye(6), T.T]) for state in output_states_skf ]
smoothed_skf = np.asarray([state.m for state in smoothed_collapsed])
smoothed_kf = np.asarray([state.x() for state in output_states_kf])
stops = np.asarray([np.exp(state.M[1]) for state in output_states_skf])

subplot_shape = (2,2)

plt.subplot2grid(subplot_shape, (0,0))
plt.plot(positions[:,0], positions[:,1], 'b-')
plt.plot(smoothed_skf[:,0], smoothed_skf[:,1], 'g-')
plt.plot(smoothed_skf[stops>0.50,0], smoothed_skf[stops>0.50,1], 'ro')

plt.subplot2grid(subplot_shape, (1,0))
plt.plot(positions[:,0], positions[:,1], 'b-')
plt.plot(smoothed_kf[:,0], smoothed_kf[:,1], 'g-')

plt.subplot2grid(subplot_shape, (0,1), rowspan=2)
plt.plot(range(n), stops)
plt.plot(range(n), 0.5 * np.ones(n), 'r--')

plt.show()