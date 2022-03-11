import numpy as np

def dot3(A, B, C):
    return np.dot(A, np.dot(B, C))


class KalmanState(object):

    def __init__(self, mean, covariance, ord=3):
        self.m = mean
        self.P = covariance

        self.ord = ord
        self.dim = int(self.m.shape[0]/self.ord)

    def x(self): 
        return self.m[0:self.dim]

    def dx(self): 
        if self.ord < 2:
            raise ValueError("No velocity in Kalman state")
        return self.m[self.dim:2*self.dim]

    def ddx(self): 
        if self.ord < 3: 
            raise ValueError("No acceleration in Kalman state")
        return self.m[2*self.dim:3*self.dim]

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
        for i in range(self.n_models):
            T = embeds[i]
            m += np.exp(self.M[i]) * np.dot(T, self._states[i].m)

        for i in range(self.n_models):
            T = embeds[i]
            m_c = np.dot(T, self._states[i].m) - m
            P += np.exp(self.M[i]) * (dot3(T, self._states[i].P, T.T) + \
                np.outer(m_c, m_c))

        return KalmanState(mean=m, covariance=P)