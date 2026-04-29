"""
Hawkes (self-exciting) point process for visit momentum.

Models how a consumer's recent visits increase the near-term
probability of another visit.  The conditional intensity is

    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

where
    mu    = baseline visit rate,
    alpha = excitation magnitude per event,
    beta  = exponential decay rate.

Stability requires alpha / beta < 1.
"""

import warnings
from typing import List, Optional, Tuple, Dict

import numpy as np
from scipy.optimize import minimize


class HawkesProcess:
    """Univariate Hawkes (self-exciting) point process.

    Parameters
    ----------
    mu : float
        Initial baseline intensity (events per unit time).
    alpha : float
        Excitation jump size per event.
    beta : float
        Exponential decay rate of excitation.

    Attributes
    ----------
    mu, alpha, beta : float
        Current parameter values (updated by ``fit``).
    fitted_ : bool
        Whether ``fit`` has been called successfully.
    """

    def __init__(
        self,
        mu: float = 0.1,
        alpha: float = 0.5,
        beta: float = 1.0,
    ):
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.fitted_ = False
        self._check_stability(warn=True)

    # ------------------------------------------------------------------
    # Core intensity
    # ------------------------------------------------------------------

    def intensity(self, t: float, history: np.ndarray) -> float:
        """Compute the conditional intensity at time *t*.

        Parameters
        ----------
        t : float
            Evaluation time.
        history : array-like of float
            Sorted event times strictly before *t*.

        Returns
        -------
        float
            lambda(t).
        """
        history = np.asarray(history, dtype=float)
        past = history[history < t]
        if len(past) == 0:
            return self.mu
        kernel = self.alpha * np.exp(-self.beta * (t - past))
        return self.mu + float(kernel.sum())

    # ------------------------------------------------------------------
    # Fitting (MLE)
    # ------------------------------------------------------------------

    def fit(
        self,
        event_times: np.ndarray,
        T: Optional[float] = None,
        method: str = "L-BFGS-B",
    ) -> "HawkesProcess":
        """Maximum-likelihood estimation of (mu, alpha, beta).

        Parameters
        ----------
        event_times : array-like of float
            Sorted event timestamps for a single realisation
            (one consumer's visit history).
        T : float, optional
            Observation window end time.  Defaults to the last event
            time.
        method : str
            Scipy optimiser method.

        Returns
        -------
        self
        """
        times = np.asarray(event_times, dtype=float)
        if len(times) < 2:
            raise ValueError("Need at least 2 events to fit a Hawkes process.")
        times = np.sort(times)
        if T is None:
            T = float(times[-1])

        def neg_log_lik(params: np.ndarray) -> float:
            mu_, alpha_, beta_ = params
            if mu_ <= 0 or alpha_ <= 0 or beta_ <= 0:
                return 1e15
            return -self._log_likelihood(times, mu_, alpha_, beta_, T)

        # Multiple random restarts
        best_result = None
        best_nll = np.inf
        rng = np.random.default_rng(42)

        for _ in range(10):
            x0 = np.array([
                rng.uniform(0.01, 1.0),
                rng.uniform(0.01, 2.0),
                rng.uniform(0.1, 5.0),
            ])
            try:
                result = minimize(
                    neg_log_lik,
                    x0,
                    method=method,
                    bounds=[(1e-8, None), (1e-8, None), (1e-8, None)],
                    options={"maxiter": 2000, "ftol": 1e-12},
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is None or not best_result.success:
            warnings.warn(
                "Optimisation did not converge; parameters may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
        if best_result is not None:
            self.mu, self.alpha, self.beta = best_result.x

        self._check_stability(warn=True)
        self.fitted_ = True
        return self

    def fit_multiple(
        self,
        sequences: List[np.ndarray],
        T: Optional[float] = None,
        method: str = "L-BFGS-B",
    ) -> "HawkesProcess":
        """Fit shared (mu, alpha, beta) across multiple consumers.

        Parameters
        ----------
        sequences : list of array-like
            Each element is one consumer's sorted event times.
        T : float, optional
            Common observation window end.  Defaults to the
            maximum event time across all sequences.
        method : str
            Scipy optimiser method.

        Returns
        -------
        self
        """
        all_times = [np.sort(np.asarray(s, dtype=float)) for s in sequences]
        if T is None:
            T = float(max(s[-1] for s in all_times if len(s) > 0))

        def neg_log_lik(params: np.ndarray) -> float:
            mu_, alpha_, beta_ = params
            if mu_ <= 0 or alpha_ <= 0 or beta_ <= 0:
                return 1e15
            total = 0.0
            for times in all_times:
                if len(times) >= 2:
                    total += self._log_likelihood(times, mu_, alpha_, beta_, T)
            return -total

        rng = np.random.default_rng(42)
        best_result = None
        best_nll = np.inf

        for _ in range(10):
            x0 = np.array([
                rng.uniform(0.01, 1.0),
                rng.uniform(0.01, 2.0),
                rng.uniform(0.1, 5.0),
            ])
            try:
                result = minimize(
                    neg_log_lik,
                    x0,
                    method=method,
                    bounds=[(1e-8, None), (1e-8, None), (1e-8, None)],
                    options={"maxiter": 2000, "ftol": 1e-12},
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is not None:
            self.mu, self.alpha, self.beta = best_result.x

        self._check_stability(warn=True)
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Prediction & simulation
    # ------------------------------------------------------------------

    def predict_next_visit(self, history: np.ndarray) -> Dict[str, float]:
        """Expected time until next event given observed history.

        Uses thinning-based Monte-Carlo estimation (10 000 draws)
        for robustness when alpha/beta is not negligible.

        Parameters
        ----------
        history : array-like of float
            Sorted past event times.

        Returns
        -------
        dict
            'expected_time' : expected waiting time from last event.
            'expected_timestamp' : expected absolute event time.
            'current_intensity' : intensity immediately after the
                last event.
        """
        history = np.sort(np.asarray(history, dtype=float))
        if len(history) == 0:
            # No history: pure Poisson with rate mu
            expected_wait = 1.0 / self.mu if self.mu > 0 else np.inf
            return {
                "expected_time": expected_wait,
                "expected_timestamp": expected_wait,
                "current_intensity": self.mu,
            }

        t_start = float(history[-1])
        current_lam = self.intensity(t_start + 1e-10, history)

        # Monte-Carlo via thinning
        n_sim = 10_000
        waits = np.empty(n_sim)
        for i in range(n_sim):
            waits[i] = self._simulate_next(history, t_start)

        expected_wait = float(np.mean(waits))
        return {
            "expected_time": expected_wait,
            "expected_timestamp": t_start + expected_wait,
            "current_intensity": current_lam,
        }

    def simulate(
        self,
        n_events: Optional[int] = None,
        duration: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a synthetic event sequence via Ogata thinning.

        Provide at least one of *n_events* or *duration*.

        Parameters
        ----------
        n_events : int, optional
            Stop after this many events.
        duration : float, optional
            Stop after this much time.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Sorted event times.
        """
        if n_events is None and duration is None:
            raise ValueError("Specify at least one of n_events or duration.")

        self._check_stability(warn=True)
        rng = np.random.default_rng(seed)
        events: List[float] = []
        t = 0.0

        # Upper-bound intensity
        lam_bar = self.mu / (1.0 - self.alpha / self.beta) if self.alpha / self.beta < 1 else self.mu + 10 * self.alpha

        while True:
            if n_events is not None and len(events) >= n_events:
                break
            if duration is not None and t >= duration:
                break

            # Inter-arrival from homogeneous Poisson(lam_bar)
            dt = rng.exponential(1.0 / lam_bar)
            t += dt

            if duration is not None and t > duration:
                break

            lam_t = self.intensity(t, np.array(events))
            # Accept with probability lam_t / lam_bar
            if rng.uniform() <= lam_t / lam_bar:
                events.append(t)
                # Update upper bound
                lam_bar = max(lam_bar, lam_t + self.alpha)

        return np.array(events)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def branching_ratio(self) -> float:
        """alpha / beta — must be < 1 for stationarity."""
        return self.alpha / self.beta if self.beta > 0 else np.inf

    @property
    def stationary_rate(self) -> float:
        """Expected event rate under stationarity: mu / (1 - alpha/beta)."""
        br = self.branching_ratio
        if br >= 1:
            return np.inf
        return self.mu / (1.0 - br)

    @property
    def params(self) -> Dict[str, float]:
        """Current parameter values."""
        return {"mu": self.mu, "alpha": self.alpha, "beta": self.beta}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_stability(self, warn: bool = False) -> None:
        """Verify alpha/beta < 1."""
        if self.beta <= 0 or self.alpha / self.beta >= 1:
            msg = (
                f"Stability violated: alpha/beta = "
                f"{self.alpha / self.beta:.4f} >= 1. "
                f"The process is explosive."
            )
            if warn:
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
            else:
                raise ValueError(msg)

    @staticmethod
    def _log_likelihood(
        times: np.ndarray,
        mu: float,
        alpha: float,
        beta: float,
        T: float,
    ) -> float:
        """Exact log-likelihood for a single realisation.

        Uses the recursive computation from Ozaki (1979) to avoid
        O(n^2) cost.

        Parameters
        ----------
        times : (n,) sorted event times
        mu, alpha, beta : parameters
        T : observation window end

        Returns
        -------
        float
            log L
        """
        n = len(times)
        if n == 0:
            return -mu * T

        # Term 1: sum of log lambda(t_i)
        # Recursive: A_i = sum_{j<i} exp(-beta*(t_i - t_j))
        #            A_i = exp(-beta*(t_i - t_{i-1})) * (1 + A_{i-1})
        A = 0.0
        log_lam_sum = np.log(mu)  # first event: lambda = mu (no history)
        for i in range(1, n):
            A = np.exp(-beta * (times[i] - times[i - 1])) * (1.0 + A)
            lam_i = mu + alpha * A
            if lam_i <= 0:
                lam_i = 1e-300
            log_lam_sum += np.log(lam_i)

        # Term 2: - integral_0^T lambda(s) ds
        #        = -mu*T - (alpha/beta) * sum_i [1 - exp(-beta*(T - t_i))]
        compensator = mu * T
        compensator += (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))

        return log_lam_sum - compensator

    def _simulate_next(self, history: np.ndarray, t_start: float) -> float:
        """Simulate a single next-event time via thinning (internal)."""
        rng = np.random.default_rng()
        t = t_start
        lam_bar = self.intensity(t + 1e-10, history) + self.alpha + 0.1

        for _ in range(100_000):
            dt = rng.exponential(1.0 / lam_bar)
            t += dt
            lam_t = self.intensity(t, history)
            if rng.uniform() <= lam_t / lam_bar:
                return t - t_start
            lam_bar = max(lam_bar, lam_t + 0.01)

        # Fallback: return a Poisson-equivalent wait
        return 1.0 / self.mu if self.mu > 0 else 100.0

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self.fitted_ else "not fitted"
        return (
            f"HawkesProcess(mu={self.mu:.4f}, alpha={self.alpha:.4f}, "
            f"beta={self.beta:.4f}, branching={self.branching_ratio:.4f}, {status})"
        )
