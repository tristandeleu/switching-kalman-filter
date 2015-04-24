import numpy as np
from .utils import KalmanState, dot3
from scipy.stats import multivariate_normal


class KalmanFilter:

    def __init__(self, model):
        self.model = model

    def _filter_prediction(self, prev_state, T=None):
        prev_state_m = prev_state.m if T is None else np.dot(T, prev_state.m)
        prev_state_P = prev_state.P if T is None else dot3(T, prev_state.P, T.T)
        # m_k^- = A_k-1 * m_k-1
        m = np.dot(self.model.A, prev_state_m)
        # P_k^- = A_k-1 * P_k-1 * A_k-1^T + Q_k-1
        P = dot3(self.model.A, prev_state_P, self.model.A.T) + self.model.Q

        return KalmanState(mean=m, covariance=P)

    def _filter_update(self, pred_state, observation):
        # v_k = y_k - H_k * m_k^-
        v = observation - np.dot(self.model.H, pred_state.m)
        # S_k = H_k * P_k^- * H_k^T + R_k
        S = dot3(self.model.H, pred_state.P, self.model.H.T) + self.model.R
        # K_k = P_k^- * H_k^T * S_k^-1
        K = dot3(pred_state.P, self.model.H.T, np.linalg.inv(S))
        # m_k = m_k^- + K_k * v_k
        m = pred_state.m + np.dot(K, v)
        # P_k = P_k^- - K_k * S_k * K_k^T
        P = pred_state.P - dot3(K, S, K.T)
        # L_t = N(v_t | 0, S_t)
        dist = multivariate_normal(mean=np.zeros_like(v), cov=S)
        L = dist.logpdf(v)

        return (m, P, L)

    def filter(self, prev_state, observation, T=None):
        # Prediction step
        pred_state = self._filter_prediction(prev_state, T)
        # Update step
        m, P, _ = self._filter_update(pred_state, observation)

        return KalmanState(mean=m, covariance=P)

    # TODO: store the updated states instead of recomputing them
    def _smoother(self, filtered_state, next_state, T=None):
        pred_next_state = self._filter_prediction(filtered_state, T)

        A = self.model.A if T is None else np.dot(self.model.A, T)
        next_state_m = next_state.m if T is None else np.dot(T, next_state.m)
        next_state_P = next_state.P if T is None else dot3(T, next_state.P, T.T)
        # [P_k+1^-]^-1
        Pm1 = np.linalg.inv(pred_next_state.P)
        # dist = multivariate_normal(mean=pred_next_state.m, cov=pred_next_state.P)
        # C_k = P_k * A_k^T * [P_k+1^-]^-1
        C = dot3(filtered_state.P, A.T, Pm1)
        # m_k^s = m_k + C_k * [m_k+1^s - m_k+1^-]
        m = filtered_state.m + np.dot(C, next_state_m - pred_next_state.m)
        # P_k^s = P_k + C_k * [P_k+1^s - P_k+1^-] * C_k^T
        P = filtered_state.P + dot3(C, next_state_P - pred_next_state.P, C.T)
        # L_t = N(m_k+1^s | m_k+1^-, P_k+1^-)
        # L = dist.logpdf(next_state_m)
        L = 0.0 # TODO

        m = m if T is None else np.dot(T, m)
        P = P if T is None else dot3(T, P, T.T)
        return (m, P, L)

    def smoother(self, filtered_state, next_state, T=None):
        m, P, _ = self._smoother(filtered_state, next_state, T)

        return KalmanState(mean=m, covariance=P)