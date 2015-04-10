import numpy as np
from scipy.misc import logsumexp
from .utils import SwitchingKalmanState
from .kalmanfilter import KalmanFilter

# See: K. P. Murphy, Switching Kalman Filters
# <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.32.5379>

class SwitchingKalmanFilter:

    def __init__(self, n_obs, n_hid, models, log_transmat):
        # self.n_models = n_models
        self.n_obs = n_obs
        self.n_hid = n_hid
        self.models = models
        self.n_models = len(models)
        self.log_transmat = log_transmat

    def _collapse(self, mu_X, V_XX, W):
        V = np.zeros((self.n_hid, self.n_hid))

        mu = np.dot(mu_X, W)

        for i in xrange(self.n_models):
            # TODO: generalize this to any number of observations
            x_c = mu_X[:,i] - mu
            y_c = mu_X[:,i] - mu
            x_c[0::2] = 0.0
            y_c[1::2] = 0.0
            V += W[i] * (V_XX[:,:,i] + np.outer(x_c, x_c) + np.outer(y_c, y_c))

        return (mu, V)

    def filter(self, prev_state, observation):
        m_ = np.zeros((self.n_hid, self.n_models, self.n_models))
        P_ = np.zeros((self.n_hid, self.n_hid, self.n_models, self.n_models))
        state = SwitchingKalmanState(mean=np.zeros((self.n_hid, self.n_models)), \
            covariance=np.zeros((self.n_hid, self.n_hid, self.n_models)))

        L = np.zeros((self.n_models, self.n_models))

        for j in xrange(self.n_models):
            A = self.models[j].A
            Q = self.models[j].Q
            T = self.models[j].T
            for i in xrange(self.n_models):
                # Prediction step
                pred_state = KalmanFilter._filter_prediction(prev_state.model(i), A, Q, T)
                # Update step
                (m_[:,i,j], P_[:,:,i,j], L[i,j]) = KalmanFilter._filter_update(pred_state, observation, self.models[i].H, self.models[i].R, self.models[i].T)

        # Posterior Transition
        # p(s_t-1=i, s_t=j | y_1:t) \propto L_t(i,j) * p(s_t=j | s_t-1=i) * p(s_t-1=i | y_1:t-1)
        M = L.T + self.log_transmat.T + prev_state.M
        M = M.T - logsumexp(M)
        # p(s_t=j | y_1:t) = \sum_i p(s_t-1=i, s_t=j | y_1:t)
        state.M = logsumexp(M, axis=0)
        # p(s_t-1=i | s_t=j, y_1:t) = p(s_t-1=i, s_t=j | y_1:t) / p(s_t=j | y_1:t)
        W = np.exp(M - state.M)

        # Collapse step
        for j in xrange(self.n_models):
            (state.m[:,j], state.P[:,:,j]) = self._collapse(m_[:,:,j], P_[:,:,:,j], W[:,j])

        return state

    def smoother(self, next_state, filtered_state):
        m_ = np.zeros((self.n_hid, self.n_models, self.n_models))
        P_ = np.zeros((self.n_hid, self.n_hid, self.n_models, self.n_models))
        state = SwitchingKalmanState(mean=np.zeros((self.n_hid, self.n_models)), \
            covariance=np.zeros((self.n_hid, self.n_hid, self.n_models)))

        for k in xrange(self.n_models):
            A = self.models[k].A
            Q = self.models[k].Q
            T = self.models[k].T
            latent_slice = self.models[k].slice
            for j in xrange(self.n_models):
                # Smoothing step
                (m_[:,j,k], P_[:,:,j,k]) = KalmanFilter._smoother(filtered_state.model(j), next_state.model(j), A, Q, T)

        # Posterior Transition
        # p(s_t=j | s_t+1=k, y_1:T) \approx \propto p(s_t+1=k | s_t=j) * p(s_t=j | y_1:t)
        U = self.log_transmat.T + filtered_state.M
        U = U.T - logsumexp(U, axis=1)
        # p(s_t=j, s_t+1=k | y_1:T) = p(s_t=j | s_t+1=k, y_1:T) * p(s_t+1=k | y_1:T)
        M = U + next_state.M
        # p(s_t=j | y1:T) = \sum_k p(s_t=j, s_t+1=k | y_1:T)
        state.M = logsumexp(M, axis=1)
        # p(s_t+1=k | s_t=j, y_1:T) = p(s_t=j, s_t+1=k | y_1:T) / p(s_t=j | y_1:T)
        W = np.exp(M.T - state.M) # WARKING: W is W.T in Murphy's paper

        # Collapse step
        for j in xrange(self.n_models):
            (state.m[:,j], state.P[:,:,j]) = self._collapse(m_[:,:,j], P_[:,:,:,j], W[:,j])

        return state