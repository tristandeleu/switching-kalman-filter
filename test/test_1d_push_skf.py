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

n_repeat = 3
n_pts = 200
pos_min, pos_max = 0.1, 40.0
dx = (pos_max-pos_min)/n_pts

pos_ary = np.linspace(pos_min, pos_max, n_pts)
f_ary = 20*(pos_ary-K_pos)
f_ary[f_ary<0] = 0

f_noise = f_ary + 2*np.random.randn(*f_ary.shape)
plt.plot(pos_ary, f_ary)
plt.xlabel("Position(m)")
plt.ylabel("Force(N)")
plt.savefig("gt_f.png")
plt.show()

models = [
    NDCWPA(dt=dx, q=2e-2, r=10.0, n_dim=1),
    NDBrownian(dt=dx, q=2e-2, r=10.0, n_dim=1)
]
Z = np.log(np.asarray([
    [0.999, 0.001],
    [0.001, 0.999]
]))
masks = [
    np.array([
        np.diag([1, 1, 1])
    ]),
    np.array([
        np.diag([1])
    ])
]

T = np.kron(np.array([1, 0, 0]), np.eye(1))
embeds = [
    [np.eye(3), T],
    [T.T, np.eye(1)]
]

kalman = SwitchingKalmanFilter(models=models, log_transmat=Z, masks=masks, embeds=embeds)

state = SwitchingKalmanState(n_models=2)
state._states[0] = KalmanState(mean=np.zeros(3), covariance=10.0 * np.eye(3))
state._states[1] = KalmanState(mean=np.zeros(1), covariance=10.0 * np.eye(1))
state.M = np.ones(2) / 2.0

print("INFO: initialization is completed")

# filtered_states_skf = [state] * n_pts
filtered_states_skf = []
for i in list(range(n_pts)) + list(range(n_pts)):
    observation = f_noise[i]
    state = kalman.filter(state, observation)
    filtered_states_skf.append(state)

print("INFO: filtering is completed")

# smoothed_states_skf = [state] * n_pts
# for i in range(1, n_pts):
#     j = n_pts - 1 - i
#     state = kalman.smoother(state, filtered_states_skf[j])
#     smoothed_states_skf[j] = state

# display_smoothed = False
# if display_smoothed:
#     output_states_skf = smoothed_states_skf
# else:
output_states_skf = filtered_states_skf

smoothed_collapsed = [ state.collapse([np.eye(3), T.T]) for state in output_states_skf ]
smoothed_skf = np.asarray([state.m for state in smoothed_collapsed])
stops = np.asarray([np.exp(state.M[0]) for state in output_states_skf])

subplot_shape = (1,2)

print("smoothed_skf shape:", smoothed_skf.shape)
print("stops shape:", stops.shape)
# plt.plot(pos_ary, smoothed_skf[:, 0], label="force")
plt.plot(list(pos_ary)+list(pos_ary+pos_max), 
         smoothed_skf[:, 1], label="stiffness")
plt.xlabel("Position(m)")
plt.ylabel("Stiffness(N/m)")
plt.legend()
plt.savefig("ps.png")
# plt.plot(pos_ary, f_ary)
plt.show()

plt.clf()
plt.plot(list(pos_ary)+list(pos_ary+pos_max), stops)
plt.xlabel("Position(m)")
plt.ylabel("Probability in contact")
plt.legend()
plt.savefig("prob_contact.png")
plt.show()

# plt.subplot2grid(subplot_shape, (0,0))
# plt.plot(pos_ary, smoothed_skf[:, 0], 'g-')
# plt.plot(smoothed_skf[stops>0.50, 0], smoothed_skf[stops>0.50, 1], 'ro')

# plt.subplot2grid(subplot_shape, (0,1))
# plt.plot(pos_ary, stops)
# plt.plot(pos_ary, 0.5 * np.ones(n_pts), 'r--')

# plt.show()
