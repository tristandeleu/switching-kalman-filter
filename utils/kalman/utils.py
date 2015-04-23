import numpy as np

def dot3(A, B, C):
    return np.dot(A, np.dot(B, C))


class KalmanState(object):

    def __init__(self, mean, covariance):
        self.m = mean
        self.P = covariance

    def x(self): return self.m[0:2]

    def dx(self): return self.m[2:4]

    def ddx(self): return self.m[4:6]


class KalmanObservation:

    def __init__(self, mean, covariance):
        self.y = mean
        self.R = covariance


class SwitchingKalmanState(object):

    def __init__(self, n_models):
        self.n_models = n_models
        self._states = np.empty(n_models, dtype=KalmanState)

    def model(self, i): 
        return self._states[i]

    def collapse(self, embeds):
        m = 0.0
        P = 0.0
        for i in xrange(self.n_models):
            T = embeds[i]
            m += np.exp(self.M[i]) * np.dot(T, self._states[i].m)

        for i in xrange(self.n_models):
            T = embeds[i]
            m_c = np.dot(T, self._states[i].m) - m
            P += np.exp(self.M[i]) * (dot3(T, self._states[i].P, T.T) + \
                np.outer(m_c, m_c))

        return KalmanState(mean=m, covariance=P)