import numpy as np


class _KalmanModel(object):

    @property
    def observation(self):
        return {'H': self.H, 'R': self.R}

    @observation.setter
    def observation(self, value):
        print 'ok'
        self.H = value['H']
        self.R = value['R']

    # def get_dynamic(self):
    #     return {'A': self.A, 'Q': self.Q}

    # def set_dynamic(self, value):
    #     self.A = value['A']
    #     self.Q = value['Q']

    # dynamic = property(get_dynamic, set_dynamic)
    


class CWPA(_KalmanModel):
    # Continuous Wiener Process Acceleration

    def __init__(self, dt=1.0, q=2e-2, r=10.0):
        dt2 = dt * dt / 2.0
        dt3 = dt * dt2 / 6.0
        dt4 = dt * dt3 / 8.0
        dt5 = dt * dt4 / 20.0

        self.A = np.asarray([
            [1.0, dt, dt2 ],
            [0.0, 1.0, dt ],
            [0.0, 0.0, 1.0]
        ])
        self.Q = q * np.asarray([
            [dt5, dt4, dt3],
            [dt4, dt3, dt2],
            [dt3, dt2, dt ]
        ])
        self.R = r
        self.H = np.asarray([1.0, 0.0, 0.0])
        # self.T = np.eye(3)


class NDCWPA(CWPA):

    def __init__(self, dt=1.0, q=2e-2, r=10.0, n_dim=1):
        dt2 = dt * dt / 2.0
        dt3 = dt * dt2 / 6.0
        dt4 = dt * dt3 / 8.0
        dt5 = dt * dt4 / 20.0
        I = np.eye(n_dim)

        self.n_obs = n_dim
        self.n_hid = 3 * n_dim
        self.A = np.kron(np.asarray([
            [1.0, dt, dt2 ],
            [0.0, 1.0, dt ],
            [0.0, 0.0, 1.0]
        ]), I)
        self.Q = q * np.kron(np.asarray([
            [dt5, dt4, dt3],
            [dt4, dt3, dt2],
            [dt3, dt2, dt ]
        ]), I)
        self.R = np.kron(r, I)
        self.H = np.kron(np.asarray([1.0, 0.0, 0.0]), I)

    # def __init__(self, dt=1.0, q=2e-2, r=10.0, n_dim=1):
    #     self.n_obs = n_dim
    #     self.n_hid = 3 * n_dim
    #     CWPA.__init__(self, dt, q, r)

    # def __setattr__(self, key, value):
    #     if key in 'AQHRT' and hasattr(self, 'n_obs'):
    #         self.__dict__[key] = np.kron(value, np.eye(self.n_obs))
    #     else:
    #         self.__dict__[key] = value


class Brownian(_KalmanModel):

    def __init__(self, dt=1.0, q=2e-2, r=10.0):
        self.A = 1.0
        self.Q = q * dt
        self.R = r
        self.H = 1.0
        # self.T = np.asarray([1.0, 0.0, 0.0])


class NDBrownian(Brownian):

    def __init__(self, dt=1.0, q=2e-2, r=10.0, n_dim=1):
        I = np.eye(n_dim)

        self.n_obs = n_dim
        self.n_hid = n_dim
        self.A = np.kron(1.0, I)
        self.Q = q * np.kron(dt, I)
        self.R = np.kron(r, I)
        self.H = np.kron(1.0, I)

    # def __init__(self, dt=1.0, q=2e-2, r=10.0, n_dim=1):
    #     self.n_obs = n_dim
    #     self.n_hid = n_dim
    #     Brownian.__init__(self, dt, q, r)

    # def __setattr__(self, key, value):
    #     if key in 'AQHRT' and hasattr(self, 'n_obs'):
    #         self.__dict__[key] = np.kron(value, np.eye(self.n_obs))
    #     else:
    #         self.__dict__[key] = value


if __name__ == '__main__':
    model = NDCWPA(dt=1.0, q=2e-2, r=10.0, n_dim=2)
    print model.__dict__
    obs = model.observation
    obs['R'] = 3
    model.observation = obs
    print model.__dict__