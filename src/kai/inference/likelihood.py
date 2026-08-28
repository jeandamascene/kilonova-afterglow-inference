"""
kai/inference/likelihood.py
Bilby v3 compatible kilonova likelihood.
"""

import numpy as np
import pandas as pd
import bilby
from bilby.core.likelihood import Likelihood

from kai.models.kilonova import KilonovaComponent, MultiComponentKilonova

COMPONENT_NAMES = {1: ["blue"], 2: ["blue", "red"], 3: ["blue", "purple", "red"]}


def build_model_from_params(params: dict, component_names: list) -> MultiComponentKilonova:
    return MultiComponentKilonova([
        KilonovaComponent(
            mej   = params[f"{name}_mej"],
            vej   = params[f"{name}_vej"],
            kappa = params[f"{name}_kappa"],
            name  = name,
        )
        for name in component_names
    ])


class KilonovaLikelihood(Likelihood):
    """
    Gaussian likelihood for multiwavelength kilonova photometry.
    Fully bilby v3 compatible.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        n_components: int = 2,
        distance_mpc: float = 40.0,
        sigma_floor: float = 0.1,
    ):
        if n_components not in COMPONENT_NAMES:
            raise ValueError(f"n_components must be 1, 2, or 3.")

        self.data          = data.copy().reset_index(drop=True)
        self.n_components  = n_components
        self.distance_mpc  = distance_mpc
        self.sigma_floor   = sigma_floor
        self.comp_names    = COMPONENT_NAMES[n_components]

        self.t_obs     = self.data["t_days"].values
        self.mag_obs   = self.data["mag"].values
        self.mag_err   = self.data["mag_err"].values
        self.bands     = self.data["band"].values
        self.sigma_eff = np.sqrt(self.mag_err ** 2 + self.sigma_floor ** 2)

        param_names = []
        for comp in self.comp_names:
            param_names += [f"{comp}_mej", f"{comp}_vej", f"{comp}_kappa"]

        # bilby v3: parameters dict with None values
        super().__init__(parameters={p: None for p in param_names})

    def _evaluate(self, params: dict) -> float:
        """Core likelihood evaluation given a parameter dict."""
        try:
            model = build_model_from_params(params, self.comp_names)
        except Exception:
            return -np.inf

        mag_model = np.empty_like(self.mag_obs)
        for i, (t_i, band_i) in enumerate(zip(self.t_obs, self.bands)):
            try:
                mag_model[i] = model.magnitude(
                    np.array([t_i]), band=band_i,
                    distance_mpc=self.distance_mpc
                )[0]
            except Exception:
                return -np.inf

        if not np.all(np.isfinite(mag_model)):
            return -np.inf

        residuals = self.mag_obs - mag_model
        return float(-0.5 * np.sum(
            (residuals / self.sigma_eff) ** 2
            + np.log(2.0 * np.pi * self.sigma_eff ** 2)
        ))

    def log_likelihood(self) -> float:
        """Called by bilby — reads from self.parameters."""
        return self._evaluate(self.parameters)

    def log_likelihood_ratio(self) -> float:
        return self.log_likelihood() - self.noise_log_likelihood()

    def noise_log_likelihood(self) -> float:
        residuals = self.mag_obs - 99.0
        return float(-0.5 * np.sum(
            (residuals / self.sigma_eff) ** 2
            + np.log(2.0 * np.pi * self.sigma_eff ** 2)
        ))

    def summary(self) -> None:
        print(f"KilonovaLikelihood")
        print(f"  Components : {self.comp_names}")
        print(f"  Free params: {list(self.parameters.keys())}")
        print(f"  Data points: {len(self.data)}")
        print(f"  Bands      : {sorted(set(self.bands))}")
        print(f"  Distance   : {self.distance_mpc} Mpc")
        print(f"  Sigma floor: {self.sigma_floor} mag")


class KilonovaLikelihoodWithSystematics(KilonovaLikelihood):
    """Extends KilonovaLikelihood with per-band calibration offsets."""

    def __init__(self, *args, offset_prior_sigma: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset_prior_sigma = offset_prior_sigma
        self.unique_bands = sorted(set(self.bands))
        for band in self.unique_bands:
            self.parameters[f"delta_{band}"] = None

    def log_likelihood(self) -> float:
        try:
            model = build_model_from_params(self.parameters, self.comp_names)
        except Exception:
            return -np.inf

        mag_model = np.empty_like(self.mag_obs)
        for i, (t_i, band_i) in enumerate(zip(self.t_obs, self.bands)):
            try:
                mag_model[i] = (
                    model.magnitude(
                        np.array([t_i]), band=band_i,
                        distance_mpc=self.distance_mpc
                    )[0]
                    + self.parameters.get(f"delta_{band_i}", 0.0)
                )
            except Exception:
                return -np.inf

        if not np.all(np.isfinite(mag_model)):
            return -np.inf

        residuals = self.mag_obs - mag_model
        ln_l = -0.5 * np.sum(
            (residuals / self.sigma_eff) ** 2
            + np.log(2.0 * np.pi * self.sigma_eff ** 2)
        )
        for band in self.unique_bands:
            delta = self.parameters.get(f"delta_{band}", 0.0)
            ln_l -= 0.5 * (delta / self.offset_prior_sigma) ** 2
        return float(ln_l)