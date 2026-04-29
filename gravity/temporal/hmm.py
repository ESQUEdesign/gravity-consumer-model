"""
Hidden Markov Model for consumer state transitions.

Models consumer lifecycle as a sequence of latent states
(Loyal, Exploring, Lapsing, Churned, Won_back) driven by
observable visit-frequency signals. Implements Baum-Welch
(EM) for parameter estimation and Viterbi for state decoding,
with no dependency on hmmlearn.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple


class ConsumerHMM:
    """Hidden Markov Model for consumer state transitions.

    States represent lifecycle positions (e.g. Loyal, Exploring,
    Lapsing, Churned, Won_back). Observations are discretised
    visit-frequency bins per time period (e.g. 'high', 'medium',
    'low', 'none').

    Parameters
    ----------
    states : list of str, optional
        Hidden state labels.  Defaults to
        ['Loyal', 'Exploring', 'Lapsing', 'Churned', 'Won_back'].
    observations : list of str, optional
        Observable symbols.  Defaults to
        ['high', 'medium', 'low', 'none'].
    random_state : int or None
        Seed for reproducibility.
    """

    DEFAULT_STATES = ["Loyal", "Exploring", "Lapsing", "Churned", "Won_back"]
    DEFAULT_OBSERVATIONS = ["high", "medium", "low", "none"]

    def __init__(
        self,
        states: Optional[List[str]] = None,
        observations: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ):
        self.states = list(states) if states is not None else list(self.DEFAULT_STATES)
        self.observations = (
            list(observations) if observations is not None else list(self.DEFAULT_OBSERVATIONS)
        )
        self.n_states = len(self.states)
        self.n_obs = len(self.observations)

        self._state_idx: Dict[str, int] = {s: i for i, s in enumerate(self.states)}
        self._obs_idx: Dict[str, int] = {o: i for i, o in enumerate(self.observations)}

        self.rng = np.random.default_rng(random_state)

        # Model parameters (initialised on fit)
        self.pi: Optional[np.ndarray] = None       # (n_states,)  initial state probs
        self.A: Optional[np.ndarray] = None         # (n_states, n_states)  transition matrix
        self.B: Optional[np.ndarray] = None         # (n_states, n_obs)  emission matrix
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences: List[List[str]],
        max_iter: int = 100,
        tol: float = 1e-6,
        n_init: int = 5,
    ) -> "ConsumerHMM":
        """Estimate model parameters via Baum-Welch (EM).

        Parameters
        ----------
        sequences : list of list of str
            Each inner list is one consumer's observation sequence
            over consecutive time periods, e.g.
            [['high', 'high', 'medium', 'low', 'none'], ...].
        max_iter : int
            Maximum EM iterations per initialisation.
        tol : float
            Convergence threshold on log-likelihood improvement.
        n_init : int
            Number of random restarts; the best (highest
            log-likelihood) solution is kept.

        Returns
        -------
        self
        """
        encoded = [self._encode_sequence(seq) for seq in sequences]

        best_ll = -np.inf
        best_params: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        for _ in range(n_init):
            pi, A, B = self._random_init()
            ll_prev = -np.inf

            for _ in range(max_iter):
                # E-step: collect sufficient statistics
                gamma_sum_0 = np.zeros(self.n_states)
                gamma_sum = np.zeros(self.n_states)
                xi_sum = np.zeros((self.n_states, self.n_states))
                emission_counts = np.zeros((self.n_states, self.n_obs))
                total_ll = 0.0

                for obs in encoded:
                    alpha, scale = self._forward(obs, pi, A, B)
                    beta = self._backward(obs, A, B, scale)
                    gamma = self._compute_gamma(alpha, beta)
                    xi = self._compute_xi(obs, alpha, beta, A, B, scale)

                    gamma_sum_0 += gamma[0]
                    gamma_sum += gamma.sum(axis=0)
                    xi_sum += xi.sum(axis=0)

                    for t, o in enumerate(obs):
                        emission_counts[:, o] += gamma[t]

                    total_ll += np.sum(np.log(scale + 1e-300))

                # M-step
                pi = gamma_sum_0 / gamma_sum_0.sum()

                # Transition: normalise xi by gamma (excluding last step)
                gamma_no_last = gamma_sum - np.zeros(self.n_states)
                # recompute gamma_no_last properly
                gamma_no_last = np.zeros(self.n_states)
                for obs_seq in encoded:
                    a, sc = self._forward(obs_seq, pi, A, B)
                    b = self._backward(obs_seq, A, B, sc)
                    g = self._compute_gamma(a, b)
                    gamma_no_last += g[:-1].sum(axis=0)

                denom = gamma_no_last[:, None]
                denom = np.where(denom == 0, 1.0, denom)
                A = xi_sum / denom

                # Emission
                denom_e = gamma_sum[:, None]
                denom_e = np.where(denom_e == 0, 1.0, denom_e)
                B = emission_counts / denom_e

                # Ensure rows sum to 1 (numerical safety)
                A = self._normalise_rows(A)
                B = self._normalise_rows(B)

                if total_ll - ll_prev < tol:
                    break
                ll_prev = total_ll

            if total_ll > best_ll:
                best_ll = total_ll
                best_params = (pi.copy(), A.copy(), B.copy())

        self.pi, self.A, self.B = best_params  # type: ignore[misc]
        self._fitted = True
        return self

    def decode(self, sequence: List[str]) -> List[str]:
        """Viterbi decoding: most-likely hidden state path.

        Parameters
        ----------
        sequence : list of str
            Observation sequence for a single consumer.

        Returns
        -------
        list of str
            Decoded state labels for each time step.
        """
        self._check_fitted()
        obs = self._encode_sequence(sequence)
        T = len(obs)
        log_pi = np.log(self.pi + 1e-300)
        log_A = np.log(self.A + 1e-300)
        log_B = np.log(self.B + 1e-300)

        # Viterbi
        V = np.zeros((T, self.n_states))
        backptr = np.zeros((T, self.n_states), dtype=int)

        V[0] = log_pi + log_B[:, obs[0]]
        for t in range(1, T):
            for j in range(self.n_states):
                scores = V[t - 1] + log_A[:, j]
                backptr[t, j] = int(np.argmax(scores))
                V[t, j] = scores[backptr[t, j]] + log_B[j, obs[t]]

        # Backtrace
        path_idx = np.zeros(T, dtype=int)
        path_idx[T - 1] = int(np.argmax(V[T - 1]))
        for t in range(T - 2, -1, -1):
            path_idx[t] = backptr[t + 1, path_idx[t + 1]]

        return [self.states[i] for i in path_idx]

    def predict_state(
        self, sequence: List[str]
    ) -> Dict[str, object]:
        """Current state estimate and one-step transition probabilities.

        Parameters
        ----------
        sequence : list of str
            Observation history for a consumer.

        Returns
        -------
        dict
            'current_state' : str — most likely current state.
            'current_probs' : dict — probability of being in each
                state at the current time step.
            'next_step_probs' : dict — probability of each state at
                the next time step.
        """
        self._check_fitted()
        obs = self._encode_sequence(sequence)
        alpha, scale = self._forward(obs, self.pi, self.A, self.B)

        # Current state distribution (last time step)
        current = alpha[-1] / (alpha[-1].sum() + 1e-300)
        current_state = self.states[int(np.argmax(current))]

        # One-step forecast
        next_probs = current @ self.A

        return {
            "current_state": current_state,
            "current_probs": {s: float(current[i]) for i, s in enumerate(self.states)},
            "next_step_probs": {s: float(next_probs[i]) for i, s in enumerate(self.states)},
        }

    def state_distribution(self) -> Dict[str, float]:
        """Stationary (equilibrium) distribution across states.

        Solves pi * A = pi, sum(pi) = 1 via eigen-decomposition
        of the transition matrix.

        Returns
        -------
        dict
            State label -> equilibrium probability.
        """
        self._check_fitted()
        eigenvalues, eigenvectors = np.linalg.eig(self.A.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = np.abs(stationary)
        stationary /= stationary.sum()
        return {s: float(stationary[i]) for i, s in enumerate(self.states)}

    def transition_matrix(self) -> np.ndarray:
        """Return the fitted transition probability matrix.

        Returns
        -------
        np.ndarray
            Shape (n_states, n_states).  Entry [i, j] is
            P(state_j | state_i).
        """
        self._check_fitted()
        return self.A.copy()

    def emission_matrix(self) -> np.ndarray:
        """Return the fitted emission probability matrix.

        Returns
        -------
        np.ndarray
            Shape (n_states, n_obs).  Entry [i, j] is
            P(obs_j | state_i).
        """
        self._check_fitted()
        return self.B.copy()

    def forecast_states(self, n_periods: int, initial_distribution: Optional[np.ndarray] = None) -> np.ndarray:
        """Project the state distribution forward n periods.

        Parameters
        ----------
        n_periods : int
            Number of future time steps.
        initial_distribution : np.ndarray, optional
            Starting distribution over states.  If None, uses the
            fitted initial-state distribution (pi).

        Returns
        -------
        np.ndarray
            Shape (n_periods + 1, n_states).  Row 0 is the initial
            distribution; row k is the projected distribution at
            period k.
        """
        self._check_fitted()
        if initial_distribution is None:
            dist = self.pi.copy()
        else:
            dist = np.asarray(initial_distribution, dtype=float).copy()
            dist /= dist.sum()

        trajectory = np.zeros((n_periods + 1, self.n_states))
        trajectory[0] = dist
        for k in range(1, n_periods + 1):
            dist = dist @ self.A
            trajectory[k] = dist
        return trajectory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

    def _encode_sequence(self, seq: List[str]) -> np.ndarray:
        """Map observation labels to integer indices."""
        try:
            return np.array([self._obs_idx[o] for o in seq], dtype=int)
        except KeyError as exc:
            raise ValueError(
                f"Unknown observation symbol {exc}. "
                f"Known symbols: {self.observations}"
            ) from exc

    def _random_init(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random Dirichlet initialisation of parameters."""
        pi = self.rng.dirichlet(np.ones(self.n_states))
        A = np.array([self.rng.dirichlet(np.ones(self.n_states)) for _ in range(self.n_states)])
        B = np.array([self.rng.dirichlet(np.ones(self.n_obs)) for _ in range(self.n_states)])
        return pi, A, B

    @staticmethod
    def _normalise_rows(M: np.ndarray) -> np.ndarray:
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return M / row_sums

    # ---- Forward / Backward with scaling ----

    @staticmethod
    def _forward(
        obs: np.ndarray, pi: np.ndarray, A: np.ndarray, B: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scaled forward algorithm.

        Returns
        -------
        alpha : (T, N) scaled forward variables
        scale : (T,) scaling factors  (product = P(O|model))
        """
        T = len(obs)
        N = len(pi)
        alpha = np.zeros((T, N))
        scale = np.zeros(T)

        alpha[0] = pi * B[:, obs[0]]
        scale[0] = alpha[0].sum()
        if scale[0] == 0:
            scale[0] = 1e-300
        alpha[0] /= scale[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
            scale[t] = alpha[t].sum()
            if scale[t] == 0:
                scale[t] = 1e-300
            alpha[t] /= scale[t]

        return alpha, scale

    @staticmethod
    def _backward(
        obs: np.ndarray, A: np.ndarray, B: np.ndarray, scale: np.ndarray
    ) -> np.ndarray:
        """Scaled backward algorithm."""
        T = len(obs)
        N = A.shape[0]
        beta = np.zeros((T, N))
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])
            if scale[t + 1] != 0:
                beta[t] /= scale[t + 1]

        return beta

    @staticmethod
    def _compute_gamma(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Posterior state probabilities gamma(t, i)."""
        raw = alpha * beta
        denom = raw.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1e-300, denom)
        return raw / denom

    @staticmethod
    def _compute_xi(
        obs: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        """Joint posterior xi(t, i, j) = P(s_t=i, s_{t+1}=j | O)."""
        T, N = alpha.shape
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            numer = (
                alpha[t][:, None]
                * A
                * (B[:, obs[t + 1]] * beta[t + 1])[None, :]
            )
            denom = numer.sum()
            if denom == 0:
                denom = 1e-300
            xi[t] = numer / denom
        return xi

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"ConsumerHMM(states={self.states}, "
            f"observations={self.observations}, {status})"
        )
