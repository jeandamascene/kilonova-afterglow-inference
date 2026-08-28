"""
Unit tests for kai.inference.likelihood and kai.inference.priors.
"""

import numpy as np
import pytest

from kai.inference.likelihood import (
    KilonovaLikelihood,
    KilonovaLikelihoodWithSystematics,
    build_model_from_params,
    COMPONENT_NAMES,
)
from kai.inference.priors import gw170817_priors, broad_priors


# ── build_model_from_params ───────────────────────────────────────────────────

class TestBuildModelFromParams:

    def test_two_component(self):
        params = {
            "blue_mej": 0.025, "blue_vej": 0.27, "blue_kappa": 0.5,
            "red_mej":  0.040, "red_vej":  0.15, "red_kappa":  10.0,
        }
        model = build_model_from_params(params, ["blue", "red"])
        assert len(model.components) == 2

    def test_single_component(self):
        params = {"blue_mej": 0.03, "blue_vej": 0.25, "blue_kappa": 0.5}
        model = build_model_from_params(params, ["blue"])
        assert len(model.components) == 1


# ── KilonovaLikelihood ────────────────────────────────────────────────────────

class TestKilonovaLikelihood:

    def test_construction(self, mock_photometry):
        lk = KilonovaLikelihood(data=mock_photometry, n_components=2,
                                distance_mpc=40.0)
        assert len(lk.t_obs) == len(mock_photometry)

    def test_parameter_names(self, mock_photometry):
        lk = KilonovaLikelihood(data=mock_photometry, n_components=2)
        expected = {
            "blue_mej", "blue_vej", "blue_kappa",
            "red_mej",  "red_vej",  "red_kappa",
        }
        assert set(lk.parameters.keys()) == expected

    def test_invalid_n_components(self, mock_photometry):
        with pytest.raises(ValueError):
            KilonovaLikelihood(data=mock_photometry, n_components=5)

    def test_log_likelihood_finite_for_good_params(self, mock_photometry):
        lk = KilonovaLikelihood(data=mock_photometry, n_components=2,
                                distance_mpc=40.0)
        lk.parameters = {
            "blue_mej": 0.025, "blue_vej": 0.27, "blue_kappa": 0.5,
            "red_mej":  0.040, "red_vej":  0.15, "red_kappa":  10.0,
        }
        ln_l = lk.log_likelihood()
        assert np.isfinite(ln_l)

    def test_log_likelihood_returns_float(self, mock_photometry):
        lk = KilonovaLikelihood(data=mock_photometry, n_components=2,
                                distance_mpc=40.0)
        lk.parameters = {
            "blue_mej": 0.025, "blue_vej": 0.27, "blue_kappa": 0.5,
            "red_mej":  0.040, "red_vej":  0.15, "red_kappa":  10.0,
        }
        assert isinstance(lk.log_likelihood(), float)

    def test_log_likelihood_negative_inf_for_bad_params(self, mock_photometry):
        """Invalid velocity should return -inf."""
        lk = KilonovaLikelihood(data=mock_photometry, n_components=2,
                                distance_mpc=40.0)
        lk.parameters = {
            "blue_mej": 0.025, "blue_vej": 99.0,  # unphysical
            "blue_kappa": 0.5,
            "red_mej": 0.04, "red_vej": 0.15, "red_kappa": 10.0,
        }
        assert lk.log_likelihood() == -np.inf

    def test_better_params_higher_likelihood(self, mock_photometry):
        """
        Villar+2017 best-fit params should give a higher likelihood
        than a clearly wrong set of parameters.
        """
        lk = KilonovaLikelihood(data=mock_photometry, n_components=2,
                                distance_mpc=40.0)
        # Good params
        lk.parameters = {
            "blue_mej": 0.025, "blue_vej": 0.27, "blue_kappa": 0.5,
            "red_mej":  0.040, "red_vej":  0.15, "red_kappa":  10.0,
        }
        ln_l_good = lk.log_likelihood()

        # Bad params (very wrong ejecta mass)
        lk.parameters = {
            "blue_mej": 0.001, "blue_vej": 0.05, "blue_kappa": 15.0,
            "red_mej":  0.001, "red_vej":  0.05, "red_kappa":  0.1,
        }
        ln_l_bad = lk.log_likelihood()

        assert ln_l_good > ln_l_bad

    def test_sigma_floor_applied(self, mock_photometry):
        """sigma_eff must be >= mag_err for all data points."""
        lk = KilonovaLikelihood(data=mock_photometry, n_components=2,
                                sigma_floor=0.1)
        assert np.all(lk.sigma_eff >= lk.mag_err)

    def test_noise_log_likelihood_finite(self, mock_photometry):
        lk = KilonovaLikelihood(data=mock_photometry)
        assert np.isfinite(lk.noise_log_likelihood())

    def test_one_component(self, mock_photometry):
        lk = KilonovaLikelihood(data=mock_photometry, n_components=1)
        lk.parameters = {
            "blue_mej": 0.05, "blue_vej": 0.20, "blue_kappa": 1.0,
        }
        assert np.isfinite(lk.log_likelihood())


# ── KilonovaLikelihoodWithSystematics ─────────────────────────────────────────

class TestKilonovaLikelihoodWithSystematics:

    def test_extra_parameters(self, mock_photometry):
        lk = KilonovaLikelihoodWithSystematics(
            data=mock_photometry, n_components=2
        )
        # Should have per-band offset parameters
        assert "delta_r" in lk.parameters
        assert "delta_J" in lk.parameters

    def test_zero_offsets_matches_base(self, mock_photometry):
        """With all offsets=0, result should match base likelihood."""
        base = KilonovaLikelihood(
            data=mock_photometry, n_components=2, distance_mpc=40.0,
            sigma_floor=0.0
        )
        syst = KilonovaLikelihoodWithSystematics(
            data=mock_photometry, n_components=2, distance_mpc=40.0,
            sigma_floor=0.0, offset_prior_sigma=1e6  # effectively flat prior
        )
        good_params = {
            "blue_mej": 0.025, "blue_vej": 0.27, "blue_kappa": 0.5,
            "red_mej":  0.040, "red_vej":  0.15, "red_kappa":  10.0,
        }
        base.parameters = good_params.copy()
        syst.parameters = {**good_params, "delta_r": 0.0, "delta_J": 0.0}

        assert abs(base.log_likelihood() - syst.log_likelihood()) < 1e-6


# ── Priors ────────────────────────────────────────────────────────────────────

class TestPriors:

    @pytest.mark.parametrize("n_comp", [1, 2, 3])
    def test_gw170817_priors_n_components(self, n_comp):
        priors = gw170817_priors(n_components=n_comp)
        expected_n_params = n_comp * 3
        assert len(priors) == expected_n_params

    def test_prior_keys_match_likelihood(self, mock_photometry):
        """Prior keys must exactly match likelihood parameter names."""
        lk = KilonovaLikelihood(data=mock_photometry, n_components=2)
        priors = gw170817_priors(n_components=2)
        assert set(priors.keys()) == set(lk.parameters.keys())

    def test_broad_priors(self):
        priors = broad_priors(n_components=2)
        assert len(priors) == 6

    def test_prior_samples_within_bounds(self):
        """Sampled values should satisfy physical constraints."""
        priors = gw170817_priors(n_components=2)
        for _ in range(100):
            sample = priors.sample()
            for name in ["blue", "red"]:
                assert 0 < sample[f"{name}_mej"] < 1.0
                assert 0 < sample[f"{name}_vej"] < 1.0
                assert sample[f"{name}_kappa"] > 0