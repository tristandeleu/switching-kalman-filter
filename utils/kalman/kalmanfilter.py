import numpy as np
from .utils import KalmanState, dot3
from scipy.stats import multivariate_normal


class KalmanFilter:

    @staticmethod
    def _filter_prediction(prev_state, A, Q, T=None):
        # Latent space compatibility
        if T is not None:
            A = dot3(T.T, A, T)
            Q = dot3(T.T, Q, T)
        # m_k^- = A_k-1 * m_k-1
        m = np.dot(A, prev_state.m)
        # P_k^- = A_k-1 * P_k-1 * A_k-1^T + Q_k-1
        P = dot3(A, prev_state.P, A.T) + Q

        return KalmanState(mean=m, covariance=P)

    @staticmethod
    def _filter_update(pred_state, observation, H, R, T=None):
        # Latent space compatibility
        if T is not None:
            H = np.dot(H, T)
        # v_k = y_k - H_k * m_k^-
        v = observation - np.dot(H, pred_state.m)
        # S_k = H_k * P_k^- * H_k^T + R_k
        S = dot3(H, pred_state.P, H.T) + R
        # K_k = P_k^- * H_k^T * S_k^-1
        K = dot3(pred_state.P, H.T, np.linalg.inv(S))
        # m_k = m_k^- + K_k * v_k
        m = pred_state.m + np.dot(K, v)
        # P_k = P_k^- - K_k * S_k * K_k^T
        P = pred_state.P - dot3(K, S, K.T)
        # L_t = N(v_t | 0, S_t)
        dist = multivariate_normal(mean=np.zeros(S.shape[0]), cov=S)
        L = dist.logpdf(v)

        return (m, P, L)

    @staticmethod
    def filter(prev_state, observation, A, Q, H, R, T=None):
        # Prediction step
        pred_state = KalmanFilter._filter_prediction(prev_state, A, Q, T)
        # Update step
        m, P, _ = KalmanFilter._filter_update(pred_state, observation, H, R, T)

        return KalmanState(mean=m, covariance=P)

    # TODO: store the updated states instead of recomputing them
    @staticmethod
    def _smoother(filtered_state, next_state, A, Q, T=None):
        pred_next_state = KalmanFilter._filter_prediction(filtered_state, A, Q, T)
        # [P_k+1^-]^-1
        if T is None:
            Pm1 = np.linalg.inv(pred_next_state.P)
            dist = multivariate_normal(mean=pred_next_state.m, cov=pred_next_state.P)
        else:
            Pm1 = dot3(T.T, np.linalg.inv(dot3(T, pred_next_state.P, T.T)), T)

            dist = multivariate_normal(mean=np.dot(T, pred_next_state.m), cov=dot3(T, pred_next_state.P, T.T))
            A = dot3(T.T, A, T) # TODO: Simplify T * T.T
        # C_k = P_k * A_k^T * [P_k+1^-]^-1
        C = dot3(filtered_state.P, A.T, Pm1)
        # m_k^s = m_k + C_k * [m_k+1^s - m_k+1^-]
        m = filtered_state.m + np.dot(C, next_state.m - pred_next_state.m)
        # P_k^s = P_k + C_k * [P_k+1^s - P_k+1^-] * C_k^T
        P = filtered_state.P + dot3(C, next_state.P - pred_next_state.P, C.T)
        # L_t = N(m_k+1^s | m_k+1^-, P_k+1^-)
        # L = dist.logpdf(next_state.m)
        L = 0.0 # TODO

        return (m, P, L)

    @staticmethod
    def smoother(filtered_state, next_state, A, Q, T=None):
        m, P, _ = KalmanFilter._smoother(filtered_state, next_state, A, Q, T)

        return KalmanState(mean=m, covariance=P)