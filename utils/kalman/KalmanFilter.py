import numpy as np
from .utils import KalmanState
from scipy.stats import multivariate_normal


class KalmanFilter:

    @staticmethod
    def _filter_prediction(prev_state, A, Q):
        # m_k^- = A_k-1 * m_k-1
        m = np.dot(A, prev_state.m)
        # P_k^- = A_k-1 * P_k-1 * A_k-1^T + Q_k-1
        P = np.dot(A, np.dot(prev_state.P, A.T)) + Q

        return KalmanState(mean=m, covariance=P)

    @staticmethod
    def _filter_update(pred_state, observation, H, R, latent_variables=None):
        # By default, every latent variable is of interest
        if latent_variables is None:
            latent_variables = np.ones(H.shape[1], dtype=bool)

        # v_k = y_k - H_k * m_k^-
        v = observation - np.dot(H, pred_state.m)
        # S_k = H_k * P_k^- * H_k^T + R_k
        S = np.dot(H, np.dot(pred_state.P, H.T)) + R
        # K_k = P_k^- * H_k^T * S_k^-1
        K = np.dot(pred_state.P, np.dot(H.T, np.linalg.inv(S)))
        # m_k = m_k^- + K_k * v_k
        m = pred_state.m + np.dot(K, v)
        # P_k = P_k^- - K_k * S_k * K_k^T
        P = pred_state.P - np.dot(K, np.dot(S, K.T))
        # L_t = N(v_t | 0, S_t)
        dist = multivariate_normal(mean=np.zeros(S.shape[0]), cov=S)
        L = dist.pdf(v)

        return (m, P, L)

    @staticmethod
    def filter(prev_state, observation, A, Q, H, R):
        # Prediction step
        pred_state = KalmanFilter._filter_prediction(prev_state, A, Q)
        # Update step
        m, P, _ = KalmanFilter._filter_update(pred_state, observation, H, R)

        return KalmanState(mean=m, covariance=P)

    # TODO: store the updated states instead of recomputing them
    @staticmethod
    def _smoother(filtered_state, next_state, A, Q, latent_slice):

        pred_next_state = KalmanFilter._filter_prediction(filtered_state, A, Q)

        # [P_k+1^-]^-1
        Pm1 = np.zeros(pred_next_state.P.shape)
        Pm1[latent_slice, latent_slice] = np.linalg.inv(pred_next_state.P[latent_slice, latent_slice])

        # C_k = P_k * A_k^T * [P_k+1^-]^-1
        C = np.dot(filtered_state.P, np.dot(A.T, Pm1))
        # m_k^s = m_k + C_k * [m_k+1^s - m_k+1^-]
        m = filtered_state.m + np.dot(C, next_state.m - pred_next_state.m)
        # P_k^s = P_k + C_k * [P_k+1^s - P_k+1^-] * C_k^T
        P = filtered_state.P + np.dot(C, np.dot(next_state.P - pred_next_state.P, C.T))

        return (m, P)

    @staticmethod
    def smoother(filtered_state, next_state, A, Q, latent_slice=None):
        if latent_slice is None:
            latent_slice = slice(0, A.shape[0])
        m, P = KalmanFilter._smoother(filtered_state, next_state, A, Q, latent_slice)

        return KalmanState(mean=m, covariance=P)