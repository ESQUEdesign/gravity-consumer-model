"""Hierarchical Bayesian Multinomial Logit model with Gibbs sampling.

Pools information across groups (markets, consumer segments) so that
data-sparse markets borrow strength from data-rich ones.  The sampler
alternates between:

    1.  Metropolis-Hastings updates of group-level coefficients beta_g,
    2.  Gibbs draw for the population mean mu,
    3.  Gibbs draw for the population covariance Sigma.

The multinomial-logit likelihood maps store utilities into choice
probabilities; utilities are linear in log-distance, log-square-footage
and standardised median income.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, invwishart

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

_FEATURE_NAMES = ["log_distance", "log_sqft", "std_income"]


def _build_feature_matrix(
    origins_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    distance_matrix: pd.DataFrame,
) -> np.ndarray:
    """Return a 3-D array  (n_origins, n_stores, n_features).

    Features
    --------
    0 – log(distance_km)  (clipped to min 0.1 km before log)
    1 – log(square_footage) (broadcast across origins)
    2 – standardised median_income (broadcast across stores)
    """
    n_origins = len(origins_df)
    n_stores = len(stores_df)
    n_feat = len(_FEATURE_NAMES)

    X = np.zeros((n_origins, n_stores, n_feat), dtype=np.float64)

    # 0 – log distance
    dist_vals = distance_matrix.values.astype(np.float64)
    dist_vals = np.clip(dist_vals, 0.1, None)
    X[:, :, 0] = np.log(dist_vals)

    # 1 – log square footage (store attribute, same for every origin)
    sqft = stores_df["square_footage"].values.astype(np.float64)
    sqft = np.clip(sqft, 1.0, None)
    X[:, :, 1] = np.log(sqft)[np.newaxis, :]

    # 2 – standardised median income (origin attribute, same for every store)
    income = origins_df["median_income"].values.astype(np.float64)
    mu_inc = np.mean(income)
    sd_inc = np.std(income) if np.std(income) > 0 else 1.0
    std_income = (income - mu_inc) / sd_inc
    X[:, :, 2] = std_income[:, np.newaxis]

    return X


# ---------------------------------------------------------------------------
# MNL log-likelihood helpers
# ---------------------------------------------------------------------------


def _mnl_log_probs(X_group: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute log-choice probabilities for a group.

    Parameters
    ----------
    X_group : (n_origins_g, n_stores, n_features)
    beta    : (n_features,)

    Returns
    -------
    log_probs : (n_origins_g, n_stores)
    """
    # V[i,j] = X[i,j,:] @ beta
    V = X_group @ beta  # (n_origins_g, n_stores)
    log_denom = logsumexp(V, axis=1, keepdims=True)  # (n_origins_g, 1)
    return V - log_denom


def _mnl_log_likelihood(
    X_group: np.ndarray,
    y_group: np.ndarray,
    beta: np.ndarray,
) -> float:
    """Log-likelihood of observed shares under MNL.

    Parameters
    ----------
    X_group : (n_origins_g, n_stores, n_features)
    y_group : (n_origins_g, n_stores)  – observed shares / counts
    beta    : (n_features,)

    Returns
    -------
    ll : float
    """
    log_p = _mnl_log_probs(X_group, beta)
    return float(np.sum(y_group * log_p))


def _log_prior(
    beta: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
    log_det_Sigma: float,
) -> float:
    """Log of MVN prior  N(beta | mu, Sigma)."""
    diff = beta - mu
    k = len(beta)
    return -0.5 * (k * np.log(2 * np.pi) + log_det_Sigma + diff @ Sigma_inv @ diff)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class HierarchicalBayesMNL:
    """Hierarchical Bayesian Multinomial Logit estimated via Gibbs sampling.

    The model assumes that group-level coefficient vectors beta_g are drawn
    from a common multivariate-normal population distribution:

        beta_g ~ N(mu, Sigma)

    Within each group the MNL likelihood links utilities to observed choice
    shares.  The Gibbs sampler iterates over beta_g (MH step), mu (conjugate
    normal), and Sigma (conjugate inverse-Wishart).

    Parameters
    ----------
    n_draws : int
        Total MCMC iterations (including burn-in).
    burn_in : int
        Number of initial draws to discard.
    thin : int
        Keep every ``thin``-th draw after burn-in.
    n_groups : int or ``"auto"``
        Number of groups to cluster origins into.  ``"auto"`` chooses
        between 5 and 10 based on the number of origins.
    """

    def __init__(
        self,
        n_draws: int = 2000,
        burn_in: int = 500,
        thin: int = 2,
        n_groups: int | str = "auto",
    ) -> None:
        self.n_draws = n_draws
        self.burn_in = burn_in
        self.thin = thin
        self.n_groups = n_groups

        # Populated after fit()
        self._chain_beta: dict[int, np.ndarray] | None = None
        self._chain_mu: np.ndarray | None = None
        self._chain_Sigma: np.ndarray | None = None
        self._group_ids: np.ndarray | None = None
        self._unique_groups: np.ndarray | None = None
        self._X: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._n_features: int | None = None
        self._acceptance_rates: dict[int, float] | None = None
        self._origins_index: pd.Index | None = None
        self._stores_index: pd.Index | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: pd.DataFrame,
        observed_shares: Optional[pd.DataFrame] = None,
        group_ids: Optional[pd.Series] = None,
    ) -> "HierarchicalBayesMNL":
        """Fit the HB-MNL model via Gibbs sampling.

        Parameters
        ----------
        origins_df : DataFrame
            Indexed by origin_id with columns lat, lon, population,
            households, median_income.
        stores_df : DataFrame
            Indexed by store_id with columns name, lat, lon,
            square_footage, category, brand, avg_rating, price_level.
        distance_matrix : DataFrame
            Origins as rows, stores as columns; values are km distances.
        observed_shares : DataFrame, optional
            Same shape as distance_matrix; values are visit probabilities /
            shares.  If *None*, a simple gravity-based proxy is generated.
        group_ids : Series, optional
            Maps origin_id to a group label.  If *None*, origins are
            clustered by geographic proximity using KMeans.

        Returns
        -------
        self
        """
        logger.info("Building feature matrix ...")
        X = _build_feature_matrix(origins_df, stores_df, distance_matrix)
        n_origins, n_stores, n_feat = X.shape
        self._n_features = n_feat
        self._origins_index = origins_df.index.copy()
        self._stores_index = stores_df.index.copy()
        self._X = X

        # Observed shares – use a gravity proxy if nothing supplied
        if observed_shares is not None:
            Y = observed_shares.values.astype(np.float64)
        else:
            logger.info("No observed_shares supplied – using gravity proxy.")
            Y = self._gravity_proxy(distance_matrix, stores_df)
        self._Y = Y

        # Group assignments
        if group_ids is not None:
            gids = group_ids.reindex(origins_df.index).values
        else:
            gids = self._auto_cluster(origins_df)
        self._group_ids = gids
        self._unique_groups = np.unique(gids)
        G = len(self._unique_groups)
        logger.info("Number of groups: %d", G)

        # Pre-compute group masks
        group_masks: dict[int, np.ndarray] = {}
        for idx, g in enumerate(self._unique_groups):
            group_masks[idx] = np.where(gids == g)[0]

        # ----------------------------------------------------------
        # Priors
        # ----------------------------------------------------------
        mu_0 = np.zeros(n_feat)
        Sigma_prior = np.eye(n_feat) * 10.0  # vague prior on mu
        Sigma_prior_inv = np.eye(n_feat) / 10.0

        nu_0 = n_feat + 2  # IW degrees of freedom
        S_0 = np.eye(n_feat)  # IW scale

        # ----------------------------------------------------------
        # Initialise
        # ----------------------------------------------------------
        beta = {idx: np.zeros(n_feat) for idx in range(G)}
        mu = np.zeros(n_feat)
        Sigma = np.eye(n_feat)
        Sigma_inv = np.eye(n_feat)
        log_det_Sigma = 0.0

        proposal_scale = np.full(G, 0.1)  # per-group adaptive proposal
        accept_count = np.zeros(G)
        total_count = np.zeros(G)

        # Storage for thinned post-burn-in draws
        n_keep = max(1, (self.n_draws - self.burn_in) // self.thin)
        chain_beta: dict[int, list[np.ndarray]] = {idx: [] for idx in range(G)}
        chain_mu: list[np.ndarray] = []
        chain_Sigma: list[np.ndarray] = []

        rng = np.random.default_rng(42)

        logger.info(
            "Starting Gibbs sampler: %d draws, %d burn-in, thin=%d",
            self.n_draws, self.burn_in, self.thin,
        )

        for it in range(self.n_draws):
            # (a) Update each beta_g via MH
            for idx in range(G):
                mask = group_masks[idx]
                X_g = X[mask]
                Y_g = Y[mask]

                beta_curr = beta[idx]
                beta_prop = beta_curr + rng.normal(0, proposal_scale[idx], size=n_feat)

                ll_curr = _mnl_log_likelihood(X_g, Y_g, beta_curr)
                ll_prop = _mnl_log_likelihood(X_g, Y_g, beta_prop)

                lp_curr = _log_prior(beta_curr, mu, Sigma_inv, log_det_Sigma)
                lp_prop = _log_prior(beta_prop, mu, Sigma_inv, log_det_Sigma)

                log_ratio = (ll_prop + lp_prop) - (ll_curr + lp_curr)

                if np.log(rng.uniform()) < log_ratio:
                    beta[idx] = beta_prop
                    accept_count[idx] += 1
                total_count[idx] += 1

            # Adapt proposal scale during burn-in (target 25-40 %)
            if it < self.burn_in and it > 0 and it % 50 == 0:
                for idx in range(G):
                    rate = accept_count[idx] / max(total_count[idx], 1)
                    if rate < 0.25:
                        proposal_scale[idx] *= 0.8
                    elif rate > 0.40:
                        proposal_scale[idx] *= 1.2

            # (b) Draw mu | betas, Sigma
            beta_mat = np.array([beta[idx] for idx in range(G)])  # (G, K)
            beta_bar = beta_mat.mean(axis=0)
            prec_post = Sigma_prior_inv + G * Sigma_inv
            cov_post = linalg.inv(prec_post)
            mu_post = cov_post @ (Sigma_prior_inv @ mu_0 + G * Sigma_inv @ beta_bar)
            mu = rng.multivariate_normal(mu_post, cov_post)

            # (c) Draw Sigma | betas, mu  from Inverse-Wishart
            diff_mat = beta_mat - mu[np.newaxis, :]  # (G, K)
            S = diff_mat.T @ diff_mat  # (K, K)
            nu_post = nu_0 + G
            scale_post = S_0 + S
            # scipy invwishart parameterisation: scale matrix, df
            Sigma = invwishart.rvs(df=nu_post, scale=scale_post, random_state=rng)
            if Sigma.ndim == 0:
                Sigma = np.array([[Sigma]])
            Sigma_inv = linalg.inv(Sigma)
            sign, log_det_Sigma = np.linalg.slogdet(Sigma)
            log_det_Sigma = float(log_det_Sigma)

            # Store thinned post-burn-in draws
            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                for idx in range(G):
                    chain_beta[idx].append(beta[idx].copy())
                chain_mu.append(mu.copy())
                chain_Sigma.append(Sigma.copy())

            if it % 500 == 0:
                logger.debug("Iteration %d / %d", it, self.n_draws)

        # Convert chains to arrays
        self._chain_beta = {
            idx: np.array(chain_beta[idx]) for idx in range(G)
        }
        self._chain_mu = np.array(chain_mu)  # (n_keep, K)
        self._chain_Sigma = np.array(chain_Sigma)  # (n_keep, K, K)

        self._acceptance_rates = {
            idx: float(accept_count[idx] / max(total_count[idx], 1))
            for idx in range(G)
        }

        self._fitted = True
        logger.info("Gibbs sampler complete. Kept %d draws.", len(chain_mu))
        return self

    def predict(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: Optional[pd.DataFrame] = None,
        group_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Posterior predictive probabilities.

        Parameters
        ----------
        origins_df, stores_df : DataFrame
            As in ``fit``.
        distance_matrix : DataFrame, optional
            If *None*, uses the distance matrix from fitting.
        group_id : int, optional
            If given, uses that group's posterior mean beta.  Otherwise
            uses the population-level mean mu.

        Returns
        -------
        prob_df : DataFrame
            Origin x store predicted probabilities.
        """
        self._check_fitted()

        if distance_matrix is None:
            X = self._X
        else:
            X = _build_feature_matrix(origins_df, stores_df, distance_matrix)

        if group_id is not None:
            idx = int(np.where(self._unique_groups == group_id)[0][0])
            beta = self._chain_beta[idx].mean(axis=0)
        else:
            beta = self._chain_mu.mean(axis=0)

        log_p = _mnl_log_probs(X, beta)
        probs = np.exp(log_p)

        return pd.DataFrame(
            probs,
            index=origins_df.index,
            columns=stores_df.index,
        )

    def predict_interval(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: Optional[pd.DataFrame] = None,
        confidence: float = 0.90,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Credible intervals from the posterior draws.

        Parameters
        ----------
        origins_df, stores_df : DataFrame
        distance_matrix : DataFrame, optional
        confidence : float
            Credible interval level (default 0.90).

        Returns
        -------
        (lower_df, upper_df) : tuple of DataFrames
        """
        self._check_fitted()

        if distance_matrix is None:
            X = self._X
        else:
            X = _build_feature_matrix(origins_df, stores_df, distance_matrix)

        alpha = 1.0 - confidence
        n_keep = self._chain_mu.shape[0]

        # Compute probabilities for each posterior draw of mu
        all_probs = np.zeros((n_keep, X.shape[0], X.shape[1]))
        for t in range(n_keep):
            beta_t = self._chain_mu[t]
            log_p = _mnl_log_probs(X, beta_t)
            all_probs[t] = np.exp(log_p)

        lower = np.quantile(all_probs, alpha / 2, axis=0)
        upper = np.quantile(all_probs, 1 - alpha / 2, axis=0)

        idx = origins_df.index
        cols = stores_df.index
        return (
            pd.DataFrame(lower, index=idx, columns=cols),
            pd.DataFrame(upper, index=idx, columns=cols),
        )

    @property
    def population_coefficients(self) -> dict:
        """Posterior mean and std of population-level mu.

        Returns
        -------
        dict with keys ``"mean"`` and ``"std"``, each mapping feature
        names to floats.
        """
        self._check_fitted()
        mu_mean = self._chain_mu.mean(axis=0)
        mu_std = self._chain_mu.std(axis=0)
        return {
            "mean": dict(zip(_FEATURE_NAMES, mu_mean)),
            "std": dict(zip(_FEATURE_NAMES, mu_std)),
        }

    @property
    def group_coefficients(self) -> pd.DataFrame:
        """Per-group posterior mean coefficients.

        Returns
        -------
        DataFrame indexed by group label with one column per feature.
        """
        self._check_fitted()
        rows = {}
        for idx, g in enumerate(self._unique_groups):
            rows[g] = self._chain_beta[idx].mean(axis=0)
        return pd.DataFrame(rows, index=_FEATURE_NAMES).T

    def convergence_diagnostics(self) -> dict:
        """Convergence diagnostics for the sampler.

        Returns
        -------
        dict with keys:
            ``r_hat``    – Gelman-Rubin statistic (split-chain) per feature
            ``ess``      – effective sample size per feature
            ``acceptance_rate`` – per-group MH acceptance rates
        """
        self._check_fitted()
        chain = self._chain_mu  # (n_keep, K)
        n = chain.shape[0]

        # Split chain in half for R-hat
        mid = n // 2
        chain_a = chain[:mid]
        chain_b = chain[mid: 2 * mid]

        r_hat = {}
        ess = {}
        for k, name in enumerate(_FEATURE_NAMES):
            # Means
            mean_a = chain_a[:, k].mean()
            mean_b = chain_b[:, k].mean()
            overall = np.concatenate([chain_a[:, k], chain_b[:, k]]).mean()

            m = 2  # number of sub-chains
            n_sub = mid

            B = n_sub * ((mean_a - overall) ** 2 + (mean_b - overall) ** 2) / (m - 1)
            W = (chain_a[:, k].var(ddof=1) + chain_b[:, k].var(ddof=1)) / m

            if W > 0:
                var_hat = (1 - 1 / n_sub) * W + B / n_sub
                r_hat[name] = float(np.sqrt(var_hat / W))
            else:
                r_hat[name] = float("nan")

            # Effective sample size (simple autocorrelation-based estimate)
            x = chain[:, k] - chain[:, k].mean()
            if np.var(x) > 0:
                acf = np.correlate(x, x, mode="full")[n - 1:]
                acf = acf / acf[0]
                # Sum paired autocorrelations until they go negative
                tau = 1.0
                for lag in range(1, n // 2):
                    rho = acf[lag] if lag < len(acf) else 0.0
                    if rho < 0.05:
                        break
                    tau += 2 * rho
                ess[name] = max(1, int(n / tau))
            else:
                ess[name] = n

        return {
            "r_hat": r_hat,
            "ess": ess,
            "acceptance_rate": self._acceptance_rates,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

    def _auto_cluster(self, origins_df: pd.DataFrame) -> np.ndarray:
        """Cluster origins by lat/lon using KMeans."""
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for automatic grouping. "
                "Install it or provide group_ids explicitly."
            ) from exc

        n = len(origins_df)
        if self.n_groups == "auto":
            k = max(2, min(10, n // 20))
            k = np.clip(k, 5, 10) if n >= 100 else max(2, min(k, n))
        else:
            k = int(self.n_groups)
        k = min(k, n)

        coords = origins_df[["lat", "lon"]].values.astype(np.float64)
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(coords)
        logger.info("Auto-clustered %d origins into %d groups.", n, k)
        return labels

    @staticmethod
    def _gravity_proxy(
        distance_matrix: pd.DataFrame,
        stores_df: pd.DataFrame,
    ) -> np.ndarray:
        """Simple gravity proxy when observed shares are unavailable.

        P(j|i) proportional to sqft_j / dist_ij^2 , then row-normalised.
        """
        dist = distance_matrix.values.astype(np.float64)
        dist = np.clip(dist, 0.1, None)
        sqft = stores_df["square_footage"].values.astype(np.float64)
        attraction = sqft[np.newaxis, :] / (dist ** 2)
        row_sums = attraction.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return attraction / row_sums
