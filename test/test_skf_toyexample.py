import numpy as np

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils.kalman import SwitchingKalmanState, SwitchingKalmanFilter, KalmanFilter, KalmanState
from utils.kalman.models import NDCWPA, NDBrownian

import matplotlib.pyplot as plt


def angle_between(x,y):
  return min(y-x, y-x+2*np.pi, y-x-2*np.pi, key=np.abs)

class ManeuveringTarget(object): 
    def __init__(self, x0, y0, v0, heading):
        self.x = x0
        self.y = y0
        self.vel = v0
        self.hdg = heading
        
        self.cmd_vel = v0
        self.cmd_hdg = heading
        self.vel_step = 0
        self.hdg_step = 0
        self.vel_delta = 0
        self.hdg_delta = 0
        self.stop_step = 0
        
        
    def update(self):
        if self.stop_step > 0:
            self.stop_step -= 1
            # return np.array([self.x, self.y])
        else:
            vx = self.vel * np.cos(self.hdg)
            vy = self.vel * np.sin(self.hdg)
            self.x += vx
            self.y += vy
        
        if self.hdg_step > 0:
            self.hdg_step -= 1
            self.hdg += self.hdg_delta

        if self.vel_step > 0:
            self.vel_step -= 1
            self.vel += self.vel_delta

        return np.array([self.x, self.y])
        

    def set_commanded_heading(self, hdg_degrees, steps):
        self.cmd_hdg = hdg_degrees
        self.hdg_delta = angle_between(self.cmd_hdg, self.hdg) / steps
        if abs(self.hdg_delta) > 0:
            self.hdg_step = steps
        else:
            self.hdg_step = 0
            
            
    def set_commanded_speed(self, speed, steps):
        self.cmd_vel = speed
        self.vel_delta = (self.cmd_vel - self.vel) / steps
        if abs(self.vel_delta) > 0:
            self.vel_step = steps
        else:
            self.vel_step = 0

    def set_commanded_stop(self, steps):
        self.stop_step = steps

n = 200
t = ManeuveringTarget(x0=0, y0=0, v0=3, heading=np.pi/4)
positions = np.zeros((n, 2))
Q = np.random.randn(n, 2) * 0.2

for i in range(100):
    positions[i, :] = t.update()
t.set_commanded_stop(50)
t.set_commanded_heading(np.pi / 2, 50)
for i in range(100):
    positions[100 + i,:] = t.update()

positions += Q

state = KalmanState(mean=np.zeros(6), covariance=10.0 * np.eye(6))
model = NDCWPA(dt=1.0, q=2e-2, r=10.0, n_dim=2)
kalman = KalmanFilter(model=model)

filtered_states_kf = [state] * n
for i in range(n):
    observation = positions[i]
    state = kalman.filter(state, observation)
    filtered_states_kf[i] = state

smoothed_states_kf = [state] * n
for i in range(1, n):
    j = n - 1 - i
    state = kalman.smoother(filtered_states_kf[j], state)
    smoothed_states_kf[j] = state

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