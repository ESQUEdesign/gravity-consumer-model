"""
Mixed Logit (Random Coefficients Logit) Discrete Choice Model
==============================================================
An academic upgrade from the Huff gravity model that allows heterogeneous
consumer preferences by letting taste parameters vary across the population.

In the standard conditional logit, all consumers share identical preference
parameters (beta).  This is the well-known IIA (Independence of Irrelevant
Alternatives) limitation -- relative odds between two stores are the same
for every consumer regardless of context.

The Mixed Logit relaxes IIA by specifying:

    U_ij = beta_i' * x_ij + epsilon_ij

where:
    x_ij        = vector of observed attributes for origin i / store j
    beta_i      = consumer-specific coefficient vector drawn from f(theta)
    epsilon_ij  = i.i.d. Type I extreme value (Gumbel) error term

Some coefficients may be *fixed* (identical across consumers), while
others are *random* and drawn from Normal(mean, std) distributions.
The researcher specifies which coefficients are of each type.

Estimation is via Simulated Maximum Likelihood (SML).  For each choice
occasion, R draws from the mixing distribution are taken, conditional
choice probabilities are computed for each draw, and the simulated
probability is the average across draws.  Halton sequences replace
pseudo-random draws for variance reduction.

The default attribute vector includes:
    distance_km, square_footage, avg_rating, price_level,
    parking_spaces, product_count

but any set of numeric columns can be used.

References
----------
Train, K.E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed.
    Cambridge University Press.  Chapters 6 (Mixed Logit) and 9 (Simulation).

McFadden, D. & Train, K. (2000). "Mixed MNL Models for Discrete Response."
    *Journal of Applied Econometrics*, 15(5), 447-470.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ATTRIBUTES: list[str] = [
    "distance_km",
    "square_footage",
    "avg_rating",
    "price_level",
    "parking_spaces",
    "product_count",
]

_DEFAULT_RANDOM_COEFFICIENTS: list[str] = [
    "distance_km",
    "square_footage",
    "avg_rating",
]

_DEFAULT_FIXED_COEFFICIENTS: list[str] = [
    "price_level",
    "parking_spaces",
    "product_count",
]

_DEFAULT_N_DRAWS: int = 500
_MIN_STD: float = 1e-6  # floor for estimated standard deviations


# ---------------------------------------------------------------------------
# Halton sequence generator
# ---------------------------------------------------------------------------

def _halton_sequence(n: int, prime: int = 2) -> np.ndarray:
    """Generate the first *n* elements of a Halton sequence for a given prime base.

    Halton sequences are low-discrepancy (quasi-random) sequences that
    provide better coverage of the unit interval than pseudo-random draws,
    reducing simulation variance for a given number of draws.

    Parameters
    ----------
    n : int
        Number of elements to generate.
    prime : int
        Base prime for the sequence (2, 3, 5, 7, ...).

    Returns
    -------
    np.ndarray
        1-D array of length *n* with values in (0, 1).
    """
    sequence = np.zeros(n)
    for i in range(n):
        f = 1.0
        result = 0.0
        index = i + 1  # Halton sequences are 1-indexed
        while index > 0:
            f /= prime
            result += f * (index % prime)
            index //= prime
        sequence[i] = result
    return sequence


def _halton_draws(n_draws: int, n_dims: int) -> np.ndarray:
    """Generate a matrix of Halton draws mapped to standard-normal quantiles.

    Each column uses a different prime base, ensuring low correlation
    between dimensions.

    Parameters
    ----------
    n_draws : int
        Number of simulation draws (rows).
    n_dims : int
        Number of random coefficient dimensions (columns).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_draws, n_dims)`` with standard-normal quantiles.
    """
    from scipy.stats import norm

    # First n_dims primes as bases.
    primes = _first_n_primes(n_dims)
    draws = np.column_stack(
        [_halton_sequence(n_draws, p) for p in primes]
    )
    # Clip away from 0 and 1 so the inverse-normal transform stays finite.
    draws = np.clip(draws, 1e-6, 1.0 - 1e-6)
    return norm.ppf(draws)


def _first_n_primes(n: int) -> list[int]:
    """Return the first *n* prime numbers."""
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


# ---------------------------------------------------------------------------
# MixedLogitModel
# ---------------------------------------------------------------------------

class MixedLogitModel:
    """Mixed Logit (Random Coefficients Logit) discrete choice model.

    Parameters
    ----------
    attributes : list[str] or None
        Names of the attribute columns in the choice data.  If ``None``,
        defaults to ``["distance_km", "square_footage", "avg_rating",
        "price_level", "parking_spaces", "product_count"]``.
    random_coefficients : list[str] or None
        Attribute names whose coefficients vary across consumers, drawn
        from Normal(mean, std).  If ``None``, defaults to
        ``["distance_km", "square_footage", "avg_rating"]``.
    fixed_coefficients : list[str] or None
        Attribute names whose coefficients are fixed (identical for all
        consumers).  If ``None``, defaults to
        ``["price_level", "parking_spaces", "product_count"]``.

    Notes
    -----
    Every attribute must appear in exactly one of ``random_coefficients``
    or ``fixed_coefficients``.  The constructor validates this constraint.

    Examples
    --------
    >>> model = MixedLogitModel()
    >>> model.fit(choice_data, n_draws=500)
    >>> probs = model.predict(choice_data)
    >>> model.get_coefficients()
    """

    def __init__(
        self,
        attributes: Optional[list[str]] = None,
        random_coefficients: Optional[list[str]] = None,
        fixed_coefficients: Optional[list[str]] = None,
    ) -> None:
        self.attributes = list(attributes or _DEFAULT_ATTRIBUTES)
        self.random_coefficients = list(
            random_coefficients or _DEFAULT_RANDOM_COEFFICIENTS
        )
        self.fixed_coefficients = list(
            fixed_coefficients or _DEFAULT_FIXED_COEFFICIENTS
        )

        # Validate that every attribute is accounted for.
        all_specified = set(self.random_coefficients) | set(self.fixed_coefficients)
        all_attrs = set(self.attributes)
        if all_specified != all_attrs:
            missing = all_attrs - all_specified
            extra = all_specified - all_attrs
            parts = []
            if missing:
                parts.append(
                    f"attributes not assigned to fixed or random: {sorted(missing)}"
                )
            if extra:
                parts.append(
                    f"coefficients not in attributes list: {sorted(extra)}"
                )
            raise ValueError(
                "Coefficient / attribute mismatch. " + "; ".join(parts)
            )

        overlap = set(self.random_coefficients) & set(self.fixed_coefficients)
        if overlap:
            raise ValueError(
                f"Attributes cannot be both fixed and random: {sorted(overlap)}"
            )

        # Dimension counts.
        self.n_random = len(self.random_coefficients)
        self.n_fixed = len(self.fixed_coefficients)
        self.n_params = self.n_fixed + 2 * self.n_random  # mean + std per random

        # Build a canonical ordering: random coefficients first, then fixed.
        # This determines the column order of the attribute matrix X.
        self._coeff_order: list[str] = (
            list(self.random_coefficients) + list(self.fixed_coefficients)
        )

        # Populated after fit().
        self._fitted: bool = False
        self._params: Optional[np.ndarray] = None
        self._fit_result: Optional[object] = None
        self._n_draws_used: int = _DEFAULT_N_DRAWS

    # ------------------------------------------------------------------
    # Parameter packing / unpacking
    # ------------------------------------------------------------------

    def _unpack_params(
        self, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unpack the flat parameter vector into interpretable components.

        Parameters
        ----------
        theta : np.ndarray
            Flat parameter vector of length ``n_params``.

        Returns
        -------
        random_means : np.ndarray
            Means of the random coefficient distributions, shape ``(n_random,)``.
        random_stds : np.ndarray
            Standard deviations of the random coefficient distributions,
            shape ``(n_random,)``.  Exponentiated from the raw values to
            ensure positivity.
        fixed_betas : np.ndarray
            Fixed coefficient values, shape ``(n_fixed,)``.
        """
        random_means = theta[: self.n_random]
        # Store log(std) internally; exponentiate for positivity.
        random_log_stds = theta[self.n_random : 2 * self.n_random]
        random_stds = np.exp(random_log_stds)
        fixed_betas = theta[2 * self.n_random :]
        return random_means, random_stds, fixed_betas

    def _pack_params(
        self,
        random_means: np.ndarray,
        random_stds: np.ndarray,
        fixed_betas: np.ndarray,
    ) -> np.ndarray:
        """Pack components into a flat parameter vector.

        Parameters
        ----------
        random_means : np.ndarray
        random_stds : np.ndarray
            Actual standard deviations (will be log-transformed).
        fixed_betas : np.ndarray

        Returns
        -------
        np.ndarray
            Flat parameter vector of length ``n_params``.
        """
        random_log_stds = np.log(np.maximum(random_stds, _MIN_STD))
        return np.concatenate([random_means, random_log_stds, fixed_betas])

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self, choice_data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate and reshape choice data into arrays for estimation.

        Parameters
        ----------
        choice_data : pd.DataFrame
            Must contain columns: ``origin_id``, ``store_id``, ``chosen``
            (0/1 indicator), plus all attribute columns.

        Returns
        -------
        X : np.ndarray
            Attribute matrix, shape ``(n_obs, n_attributes)``, columns
            ordered as ``_coeff_order``.
        chosen : np.ndarray
            Binary choice indicator, shape ``(n_obs,)``.
        origin_ids : np.ndarray
            Origin identifier for each row, shape ``(n_obs,)``.
        choice_sets : list[np.ndarray]
            List of index arrays, one per origin, giving the row indices
            belonging to that origin's choice set.
        """
        required_cols = {"origin_id", "store_id", "chosen"} | set(self.attributes)
        missing = required_cols - set(choice_data.columns)
        if missing:
            raise ValueError(
                f"choice_data is missing required columns: {sorted(missing)}"
            )

        df = choice_data.copy()
        X = df[self._coeff_order].values.astype(np.float64)
        chosen = df["chosen"].values.astype(np.float64)
        origin_ids = df["origin_id"].values

        # Build choice-set indices grouped by origin.
        unique_origins = np.unique(origin_ids)
        choice_sets = []
        for oid in unique_origins:
            mask = origin_ids == oid
            indices = np.where(mask)[0]
            choice_sets.append(indices)

        return X, chosen, origin_ids, choice_sets

    # ------------------------------------------------------------------
    # Simulated log-likelihood
    # ------------------------------------------------------------------

    def _simulated_log_likelihood(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        chosen: np.ndarray,
        choice_sets: list[np.ndarray],
        halton_draws: np.ndarray,
    ) -> float:
        """Compute the negative simulated log-likelihood.

        For each origin's choice set and each Halton draw r:
            1. Construct the full beta vector:
               beta_r = [random_means + random_stds * z_r ; fixed_betas]
            2. Compute utilities V_ij = beta_r' * x_ij for all j in choice set.
            3. Compute conditional logit probabilities.
            4. Extract the probability of the observed choice.
        Average over draws to get the simulated probability, then sum
        log-probabilities across origins.

        Parameters
        ----------
        theta : np.ndarray
            Flat parameter vector.
        X : np.ndarray
            Attribute matrix ``(n_obs, n_attributes)``.
        chosen : np.ndarray
            Binary indicator ``(n_obs,)``.
        choice_sets : list[np.ndarray]
            Row indices for each origin's choice set.
        halton_draws : np.ndarray
            Standard-normal Halton draws ``(n_draws, n_random)``.

        Returns
        -------
        float
            Negative simulated log-likelihood (to be minimised).
        """
        random_means, random_stds, fixed_betas = self._unpack_params(theta)
        n_draws = halton_draws.shape[0]

        total_ll = 0.0

        for cs_indices in choice_sets:
            # Attribute sub-matrix for this choice set: (n_alts, n_attrs).
            X_cs = X[cs_indices]
            chosen_cs = chosen[cs_indices]

            # Which alternative was chosen?
            chosen_idx = np.where(chosen_cs == 1)[0]
            if len(chosen_idx) == 0:
                # No observed choice in this set -- skip.
                continue
            chosen_idx = chosen_idx[0]

            # Separate random and fixed attribute columns.
            X_random = X_cs[:, : self.n_random]   # (n_alts, n_random)
            X_fixed = X_cs[:, self.n_random :]     # (n_alts, n_fixed)

            # Fixed component of utility (same for all draws).
            V_fixed = X_fixed @ fixed_betas  # (n_alts,)

            # Simulate over draws.
            sim_prob = 0.0
            for r in range(n_draws):
                # Draw-specific betas for random coefficients.
                beta_random_r = random_means + random_stds * halton_draws[r]

                # Total utility for each alternative in this draw.
                V_r = X_random @ beta_random_r + V_fixed  # (n_alts,)

                # Softmax for numerical stability.
                V_r_shifted = V_r - np.max(V_r)
                exp_V = np.exp(V_r_shifted)
                prob_r = exp_V / max(np.sum(exp_V), 1e-30)

                sim_prob += prob_r[chosen_idx]

            sim_prob /= n_draws

            # Guard against log(0).
            sim_prob = max(sim_prob, 1e-30)
            total_ll += np.log(sim_prob)

        return -total_ll

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        choice_data: pd.DataFrame,
        n_draws: int = _DEFAULT_N_DRAWS,
        *,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
        verbose: bool = False,
        x0: Optional[np.ndarray] = None,
    ) -> "MixedLogitModel":
        """Estimate model parameters via Simulated Maximum Likelihood.

        Parameters
        ----------
        choice_data : pd.DataFrame
            Panel of choice occasions.  Required columns:
            ``origin_id``, ``store_id``, ``chosen`` (0/1 binary indicator),
            plus all attribute columns specified at construction.

            Each ``origin_id`` group is one choice set: the alternatives
            available to that consumer, with exactly one row where
            ``chosen == 1``.
        n_draws : int, default 500
            Number of Halton draws per choice set for simulating the
            mixed-logit probability integral.  More draws improve
            accuracy at the cost of computation time.
        method : str, default "L-BFGS-B"
            Optimiser method passed to ``scipy.optimize.minimize``.
        maxiter : int, default 1000
            Maximum iterations for the optimiser.
        verbose : bool, default False
            If True, display optimisation progress.
        x0 : np.ndarray or None
            Starting parameter vector.  If ``None``, all means and fixed
            coefficients start at 0; log-stds start at 0 (i.e. std = 1).

        Returns
        -------
        MixedLogitModel
            ``self``, with fitted parameters accessible via
            ``get_coefficients()``.

        Raises
        ------
        ValueError
            If ``choice_data`` is missing required columns.
        """
        X, chosen, origin_ids, choice_sets = self._prepare_data(choice_data)

        # Generate Halton draws (shared across all choice sets for stability).
        halton = _halton_draws(n_draws, self.n_random) if self.n_random > 0 else np.empty((n_draws, 0))
        self._n_draws_used = n_draws

        # Starting values.
        if x0 is None:
            theta0 = np.zeros(self.n_params)
            # log-stds start at 0 => std = 1.
        else:
            if len(x0) != self.n_params:
                raise ValueError(
                    f"x0 must have length {self.n_params}, got {len(x0)}"
                )
            theta0 = x0.copy()

        logger.info(
            "Starting Mixed Logit estimation: %d observations, %d choice sets, "
            "%d draws, %d parameters (%d random, %d fixed).",
            len(X),
            len(choice_sets),
            n_draws,
            self.n_params,
            self.n_random,
            self.n_fixed,
        )

        result = minimize(
            self._simulated_log_likelihood,
            theta0,
            args=(X, chosen, choice_sets, halton),
            method=method,
            options={"disp": verbose, "maxiter": maxiter},
        )

        if not result.success:
            logger.warning("Optimisation did not converge: %s", result.message)

        self._params = result.x
        self._fitted = True
        self._fit_result = result

        random_means, random_stds, fixed_betas = self._unpack_params(result.x)
        logger.info(
            "Estimation complete. Log-likelihood: %.4f, Converged: %s",
            -result.fun,
            result.success,
        )
        for i, name in enumerate(self.random_coefficients):
            logger.info(
                "  %s (random): mean=%.4f, std=%.4f",
                name,
                random_means[i],
                random_stds[i],
            )
        for i, name in enumerate(self.fixed_coefficients):
            logger.info("  %s (fixed): %.4f", name, fixed_betas[i])

        return self

    def predict(self, choice_data: pd.DataFrame, n_draws: Optional[int] = None) -> pd.Series:
        """Compute predicted choice probabilities for each row.

        Parameters
        ----------
        choice_data : pd.DataFrame
            Same format as the training data: ``origin_id``, ``store_id``,
            plus attribute columns.  The ``chosen`` column is optional
            (ignored if present).
        n_draws : int or None
            Number of Halton draws for simulation.  If ``None``, uses the
            same value from ``fit()``.

        Returns
        -------
        pd.Series
            Predicted probability for each row, indexed like the input.
            Within each ``origin_id`` group, probabilities sum to 1.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted or self._params is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict()."
            )

        if n_draws is None:
            n_draws = self._n_draws_used

        # Ensure 'chosen' column exists for _prepare_data (fill with 0 if absent).
        df = choice_data.copy()
        if "chosen" not in df.columns:
            df["chosen"] = 0

        X, _, _, choice_sets = self._prepare_data(df)
        random_means, random_stds, fixed_betas = self._unpack_params(self._params)

        halton = _halton_draws(n_draws, self.n_random) if self.n_random > 0 else np.empty((n_draws, 0))

        probabilities = np.zeros(len(X))

        for cs_indices in choice_sets:
            X_cs = X[cs_indices]
            X_random = X_cs[:, : self.n_random]
            X_fixed = X_cs[:, self.n_random :]

            V_fixed = X_fixed @ fixed_betas

            # Accumulate probabilities across draws.
            prob_accum = np.zeros(len(cs_indices))

            for r in range(n_draws):
                beta_random_r = random_means + random_stds * halton[r]
                V_r = X_random @ beta_random_r + V_fixed

                V_r_shifted = V_r - np.max(V_r)
                exp_V = np.exp(V_r_shifted)
                prob_r = exp_V / max(np.sum(exp_V), 1e-30)
                prob_accum += prob_r

            prob_accum /= n_draws
            probabilities[cs_indices] = prob_accum

        return pd.Series(probabilities, index=choice_data.index, name="probability")

    def willingness_to_travel(self) -> pd.DataFrame:
        """Compute the marginal rate of substitution between distance and other attributes.

        The willingness-to-travel (WTT) for attribute *k* is defined as:

            WTT_k = -beta_k / beta_distance

        This gives the additional distance (in km) a consumer would
        travel for a one-unit increase in attribute *k*, holding utility
        constant.  For random coefficients, the mean of the distribution
        is used.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``attribute``, ``wtt_km``, and
            ``interpretation``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If ``distance_km`` is not among the model attributes.
        """
        if not self._fitted or self._params is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before willingness_to_travel()."
            )

        if "distance_km" not in self.attributes:
            raise ValueError(
                "willingness_to_travel() requires 'distance_km' in the "
                "attribute list."
            )

        random_means, random_stds, fixed_betas = self._unpack_params(self._params)

        # Get the distance coefficient (mean if random, point value if fixed).
        coeff_values = self._get_mean_coefficients(random_means, fixed_betas)
        coeff_dict = dict(zip(self._coeff_order, coeff_values))
        beta_distance = coeff_dict["distance_km"]

        if abs(beta_distance) < 1e-12:
            logger.warning(
                "Distance coefficient is near zero (%.6f); WTT values "
                "will be unstable.",
                beta_distance,
            )

        rows = []
        for attr in self.attributes:
            if attr == "distance_km":
                continue
            beta_k = coeff_dict[attr]
            wtt = -beta_k / beta_distance if abs(beta_distance) > 1e-12 else np.inf
            interpretation = (
                f"Consumers would travel {abs(wtt):.2f} km "
                f"{'more' if wtt > 0 else 'less'} for a one-unit "
                f"increase in {attr}."
            )
            rows.append(
                {"attribute": attr, "wtt_km": wtt, "interpretation": interpretation}
            )

        return pd.DataFrame(rows)

    def attribute_importance(self) -> pd.DataFrame:
        """Rank attributes by the absolute magnitude of their coefficients.

        For random coefficients, the mean of the distribution is used.
        This provides a first-order summary of which attributes most
        strongly influence consumer choice.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``attribute``, ``coefficient_mean``,
            ``abs_coefficient``, ``type`` (fixed/random), sorted by
            ``abs_coefficient`` descending.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted or self._params is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before attribute_importance()."
            )

        random_means, random_stds, fixed_betas = self._unpack_params(self._params)
        coeff_values = self._get_mean_coefficients(random_means, fixed_betas)

        rows = []
        for i, attr in enumerate(self._coeff_order):
            coeff_type = (
                "random" if attr in self.random_coefficients else "fixed"
            )
            rows.append(
                {
                    "attribute": attr,
                    "coefficient_mean": coeff_values[i],
                    "abs_coefficient": abs(coeff_values[i]),
                    "type": coeff_type,
                }
            )

        df = pd.DataFrame(rows).sort_values(
            "abs_coefficient", ascending=False
        ).reset_index(drop=True)
        return df

    def get_coefficients(self) -> pd.DataFrame:
        """Return a summary DataFrame of all estimated coefficients.

        Returns
        -------
        pd.DataFrame
            Columns: ``name``, ``type`` (fixed/random), ``mean``,
            ``std`` (NaN for fixed coefficients).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted or self._params is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before get_coefficients()."
            )

        random_means, random_stds, fixed_betas = self._unpack_params(self._params)

        rows = []
        for i, name in enumerate(self.random_coefficients):
            rows.append(
                {
                    "name": name,
                    "type": "random",
                    "mean": random_means[i],
                    "std": random_stds[i],
                }
            )
        for i, name in enumerate(self.fixed_coefficients):
            rows.append(
                {
                    "name": name,
                    "type": "fixed",
                    "mean": fixed_betas[i],
                    "std": np.nan,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Diagnostics / introspection
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit()`` has been called successfully."""
        return self._fitted

    @property
    def log_likelihood(self) -> float:
        """Log-likelihood at the estimated parameters.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted or self._fit_result is None:
            raise RuntimeError("Model has not been fitted.")
        return -self._fit_result.fun

    def summary(self) -> str:
        """Human-readable summary of the model state."""
        status = "fitted" if self._fitted else "not fitted"
        lines = [
            "MixedLogitModel",
            f"  status            : {status}",
            f"  attributes        : {self.attributes}",
            f"  random coefficients: {self.random_coefficients}",
            f"  fixed coefficients : {self.fixed_coefficients}",
            f"  n_parameters      : {self.n_params}",
        ]
        if self._fitted and self._fit_result is not None:
            lines.append(f"  log-likelihood    : {-self._fit_result.fun:.4f}")
            lines.append(f"  converged         : {self._fit_result.success}")
            lines.append(f"  n_draws           : {self._n_draws_used}")
            lines.append("")
            lines.append("  Coefficients:")
            random_means, random_stds, fixed_betas = self._unpack_params(
                self._params
            )
            for i, name in enumerate(self.random_coefficients):
                lines.append(
                    f"    {name:20s}  (random)  mean={random_means[i]:+.4f}  "
                    f"std={random_stds[i]:.4f}"
                )
            for i, name in enumerate(self.fixed_coefficients):
                lines.append(
                    f"    {name:20s}  (fixed)   beta={fixed_betas[i]:+.4f}"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MixedLogitModel(attributes={self.attributes}, "
            f"n_random={self.n_random}, n_fixed={self.n_fixed}, "
            f"fitted={self._fitted})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mean_coefficients(
        self,
        random_means: np.ndarray,
        fixed_betas: np.ndarray,
    ) -> np.ndarray:
        """Concatenate random means and fixed betas in canonical order.

        Returns
        -------
        np.ndarray
            Coefficient means in ``_coeff_order``, shape ``(n_attributes,)``.
        """
        return np.concatenate([random_means, fixed_betas])
