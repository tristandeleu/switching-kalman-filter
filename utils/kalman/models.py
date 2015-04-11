import numpy as np


class CWPA(object):
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
        self.T = np.eye(3)


class NDCWPA(CWPA):

    def __init__(self, dt=1.0, q=2e-2, r=10.0, n_dim=1):
        self.n_dim = n_dim
        super(NDCWPA, self).__init__(dt, q, r)

    def __setattr__(self, key, value):
        if key in 'AQHRT' and hasattr(self, 'n_dim'):
            self.__dict__[key] = np.kron(value, np.eye(self.n_dim))
        else:
            self.__dict__[key] = value


class Brownian(object):

    def __init__(self, dt=1.0, q=2e-2, r=10.0):
        self.A = 1.0
        self.Q = q * dt
        self.R = r
        self.H = 1.0
        self.T = np.asarray([1.0, 0.0, 0.0])


class NDBrownian(Brownian):

    def __init__(self, dt=1.0, q=2e-2, r=10.0, n_dim=1):
        self.n_dim = n_dim
        super(NDBrownian, self).__init__(dt, q, r)

    def __setattr__(self, key, value):
        if key in 'AQHRT' and hasattr(self, 'n_dim'):
            self.__dict__[key] = np.kron(value, np.eye(self.n_dim))
        else:
            self.__dict__[key] = value


if __name__ == '__main__':
    model = NDCWPA(dt=1.0, q=2e-2, r=10.0, n_dim=2)
    print model.A