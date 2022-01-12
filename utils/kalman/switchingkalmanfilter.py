import numpy as np
from scipy.special import logsumexp
from .utils import SwitchingKalmanState, KalmanState
from .kalmanfilter import KalmanFilter

# See: K. P. Murphy, Switching Kalman Filters
# <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.32.5379>

class SwitchingKalmanFilter:

    def __init__(self, models, log_transmat, masks, embeds):
        self.models = models
        self.n_models = len(models)
        self.log_transmat = log_transmat
        self.masks = masks
        self.embeds = embeds

    def _collapse(self, mu_X, V_XX, W, mask):
        mu = np.dot(mu_X, W)

        mu_X_c = mu_X.T - mu
        mu_X_m = np.dot(mask, mu_X_c.T)
        V = np.dot(V_XX, W)
        for i in range(self.n_models):
            V += W[i] * np.dot(mu_X_m[:,:,i].T, mu_X_m[:,:,i])

        return (mu, V)

    def _init_gpb(self):
        return [KalmanState(mean=np.zeros((model.n_hid, self.n_models)), \
            covariance=np.zeros((model.n_hid, model.n_hid, self.n_models))) \
            for model in self.models]

    def filter(self, prev_state, observation):
        gpb_ = self._init_gpb()
        state = SwitchingKalmanState(n_models=self.n_models)
        L = np.zeros((self.n_models, self.n_models))

        for j in range(self.n_models):
            kalman = KalmanFilter(model=self.models[j])
            for i in range(self.n_models):
                # Prediction step
                pred_state = kalman._filter_prediction(prev_state.model(i), self.embeds[i][j])
                # Update step
                (gpb_[j].m[:,i], gpb_[j].P[:,:,i], L[i,j]) = kalman._filter_update(pred_state, observation)

        # Posterior Transition
        # p(s_t-1=i, s_t=j | y_1:t) \propto L_t(i,j) * p(s_t=j | s_t-1=i) * p(s_t-1=i | y_1:t-1)
        M = L.T + self.log_transmat.T + prev_state.M
        M = M.T - logsumexp(M)
        # p(s_t=j | y_1:t) = \sum_i p(s_t-1=i, s_t=j | y_1:t)
        state.M = logsumexp(M, axis=0)
        # p(s_t-1=i | s_t=j, y_1:t) = p(s_t-1=i, s_t=j | y_1:t) / p(s_t=j | y_1:t)
        W = np.exp(M - state.M)

        # Collapse step
        for j in range(self.n_models):
            # (state.m[:,j], state.P[:,:,j]) = self._collapse(gpb_[j].m, gpb_[j].P, W[:,j])
            m, P = self._collapse(gpb_[j].m, gpb_[j].P, W[:,j], self.masks[j])
            state._states[j] = KalmanState(mean=m, covariance=P)

        return state

    def smoother(self, next_state, filtered_state):
        gpb_ = self._init_gpb()
        state = SwitchingKalmanState(n_models=self.n_models)

        for k in range(self.n_models):
            kalman = KalmanFilter(model=self.models[k])
            for j in range(self.n_models):
                # Smoothing step
                (gpb_[k].m[:,j], gpb_[k].P[:,:,j]) = kalman._smoother(\
                    filtered_state.model(j), next_state.model(j), self.embeds[j][k])

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
        for j in range(self.n_models):
            # (state.m[:,j], state.P[:,:,j]) = self._collapse(m_[:,:,j], P_[:,:,:,j], W[:,j])
            m, P = self._collapse(gpb_[j].m, gpb_[j].P, W[:,j], self.masks[j])
            state._states[j] = KalmanState(mean=m, covariance=P)

        return state
