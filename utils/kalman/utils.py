import numpy as np

def dot3(A, B, C):
    return np.dot(A, np.dot(B, C))


class KalmanState:

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

class SwitchingKalmanState:

    def __init__(self, mean, covariance, switching=None):
        self.m = mean
        self.P = covariance
        self.M = switching

    def model(self, i): 
        return KalmanState(mean=self.m[:,i], covariance=self.P[:,:,i])