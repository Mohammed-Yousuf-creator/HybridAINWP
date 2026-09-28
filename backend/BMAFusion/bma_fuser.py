"""
bma_fuser.py

Bayesian Model Averaging (BMA) fuser that combines ECMWF IFS (NWP) and
ECMWF AIFS (AI) point forecasts into a single probabilistic forecast --
a fitted 2-component Gaussian-mixture per variable, sampled down to a
central value + quantiles (+ optional exceedance probability).

Two separate concerns live here, matching the architecture diagram:

  - BMAFuser.fit_variable / fit_from_training   (offline / "TRAINING ONLY")
        Fits per-variable weight/bias/sigma from historical IFS+AIFS
        forecasts verified against IMD gridded observations.
        Driven by train_bma.py.

  - BMAFuser.predict                            (operational)
        Uses previously-fitted parameters plus today's IFS/AIFS point
        values to produce the probabilistic forecast for the DB/dashboard.
        Driven by run_forecast_pipeline.py.

Parameters are plain JSON (see save_params/load_params) so they can be
versioned, checked into git, or produced by any training environment --
this class doesn't care how they were made, only their shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

DEFAULT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass
class VariableBMAParams:
    """Fitted BMA parameters for one variable (one 2-component Gaussian mixture)."""
    variable: str
    weight_ifs: float
    weight_aifs: float
    bias_ifs: float
    bias_aifs: float
    sigma_ifs: float
    sigma_aifs: float
    n_train: int = 0


class BMAFuser:
    def __init__(self, quantiles: Iterable[float] = DEFAULT_QUANTILES,
                 n_samples: int = 20_000, random_state: int = 0):
        self.quantiles = tuple(quantiles)
        self.n_samples = n_samples
        self._rng = np.random.default_rng(random_state)
        self.params: dict[str, VariableBMAParams] = {}

    # ---------------------------------------------------------- fitting
    def fit_variable(self, variable: str, obs: np.ndarray,
                      ifs_fc: np.ndarray, aifs_fc: np.ndarray) -> VariableBMAParams:
        """
        Fits a 2-component Gaussian-mixture BMA model for one variable by
        maximum likelihood:

            obs ~ w_ifs  * N(ifs_fc  + bias_ifs,  sigma_ifs)
                + w_aifs * N(aifs_fc + bias_aifs, sigma_aifs)
            w_ifs + w_aifs = 1

        obs/ifs_fc/aifs_fc must be paired, same-length arrays (e.g. every
        historical location x valid_time triple you have ground truth for).
        """
        obs = np.asarray(obs, dtype=float)
        ifs_fc = np.asarray(ifs_fc, dtype=float)
        aifs_fc = np.asarray(aifs_fc, dtype=float)
        mask = np.isfinite(obs) & np.isfinite(ifs_fc) & np.isfinite(aifs_fc)
        obs, ifs_fc, aifs_fc = obs[mask], ifs_fc[mask], aifs_fc[mask]
        if len(obs) < 20:
            raise ValueError(f"[{variable}] need >=20 paired training samples, got {len(obs)}")

        def neg_log_lik(theta):
            w_logit, b_ifs, b_aifs, log_s_ifs, log_s_aifs = theta
            w = 1.0 / (1.0 + np.exp(-w_logit))  # sigmoid -> (0, 1)
            s_ifs = np.exp(log_s_ifs)
            s_aifs = np.exp(log_s_aifs)
            density = (w * norm.pdf(obs, ifs_fc + b_ifs, s_ifs) +
                       (1 - w) * norm.pdf(obs, aifs_fc + b_aifs, s_aifs))
            return -np.sum(np.log(np.clip(density, 1e-300, None)))

        resid_ifs = obs - ifs_fc
        resid_aifs = obs - aifs_fc
        x0 = [0.0, np.mean(resid_ifs), np.mean(resid_aifs),
              np.log(np.std(resid_ifs) + 1e-6), np.log(np.std(resid_aifs) + 1e-6)]

        result = minimize(neg_log_lik, x0, method="L-BFGS-B")
        w_logit, b_ifs, b_aifs, log_s_ifs, log_s_aifs = result.x
        w = 1.0 / (1.0 + np.exp(-w_logit))

        params = VariableBMAParams(
            variable=variable,
            weight_ifs=float(w),
            weight_aifs=float(1 - w),
            bias_ifs=float(b_ifs),
            bias_aifs=float(b_aifs),
            sigma_ifs=float(np.exp(log_s_ifs)),
            sigma_aifs=float(np.exp(log_s_aifs)),
            n_train=int(len(obs)),
        )
        self.params[variable] = params
        return params

    def fit_from_training(self, training_data: dict[str, dict]) -> dict[str, VariableBMAParams]:
        """
        training_data: {variable: {"obs": array, "ifs": array, "aifs": array}}
        Convenience wrapper around fit_variable for multiple variables at once.
        """
        for variable, d in training_data.items():
            self.fit_variable(variable, d["obs"], d["ifs"], d["aifs"])
        return self.params

    # ------------------------------------------------------- persistence
    def save_params(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({v: asdict(p) for v, p in self.params.items()}, f, indent=2)

    def load_params(self, path: str) -> dict[str, VariableBMAParams]:
        with open(path) as f:
            raw = json.load(f)
        self.params = {v: VariableBMAParams(**p) for v, p in raw.items()}
        return self.params

    # ------------------------------------------------------------ predict
    def predict(self, variable: str, ifs_value: float, aifs_value: float,
                event_threshold: Optional[float] = None) -> dict:
        """
        Produces the probabilistic forecast for one variable at one
        location/valid_time, given today's IFS and AIFS point values
        (both in the SAME raw units the model was trained on).

        Returns
        -------
        dict with keys: central_value, p10, p25, p50, p75, p90, probability
        (probability is None unless event_threshold is given).
        """
        if variable not in self.params:
            raise KeyError(
                f"No fitted BMA parameters for variable {variable!r}. "
                "Call fit_variable/fit_from_training or load_params first."
            )
        p = self.params[variable]

        empty = {f"p{int(round(q * 100))}": None for q in self.quantiles}
        if not (np.isfinite(ifs_value) and np.isfinite(aifs_value)):
            return {"central_value": None, **empty, "probability": None}

        n_ifs = int(round(self.n_samples * p.weight_ifs))
        n_aifs = self.n_samples - n_ifs
        parts = []
        if n_ifs > 0:
            parts.append(self._rng.normal(ifs_value + p.bias_ifs, p.sigma_ifs, n_ifs))
        if n_aifs > 0:
            parts.append(self._rng.normal(aifs_value + p.bias_aifs, p.sigma_aifs, n_aifs))
        samples = np.concatenate(parts)

        quantile_values = np.quantile(samples, self.quantiles)
        result = {"central_value": float(np.mean(samples))}
        for q, v in zip(self.quantiles, quantile_values):
            result[f"p{int(round(q * 100))}"] = float(v)

        result["probability"] = (
            float(np.mean(samples >= event_threshold)) if event_threshold is not None else None
        )
        return result