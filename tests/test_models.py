"""
Unit tests for kai.models.kilonova

Tests cover:
- Physical validation of component parameters
- Monotonicity and magnitude ranges
- Multi-component flux superposition
- Edge cases and error handling
"""

import numpy as np
import pytest

from kai.models.kilonova import (
    KilonovaComponent,
    MultiComponentKilonova,
    heating_rate,
    thermalization_efficiency,
    FILTER_WAVELENGTHS,
)


# ── heating_rate ──────────────────────────────────────────────────────────────

class TestHeatingRate:

    def test_positive_values(self):
        t = np.linspace(0.1, 20, 100)
        assert np.all(heating_rate(t) > 0)

    def test_decreasing_with_time(self):
        """Heating rate should generally decrease after ~1 day."""
        t = np.linspace(1.0, 20.0, 50)
        eps = heating_rate(t)
        # Not strictly monotonic but should trend downward
        assert eps[-1] < eps[0]

    def test_scalar_input(self):
        """Should handle scalar input gracefully."""
        result = heating_rate(1.0)
        assert np.isfinite(result)
        assert result > 0

    def test_units_order_of_magnitude(self):
        """At t=1 day, heating rate should be ~1e10 erg/s/g."""
        eps_1d = heating_rate(1.0)
        assert 1e8 < eps_1d < 1e12


# ── thermalization_efficiency ─────────────────────────────────────────────────

class TestThermalizationEfficiency:

    def test_range(self):
        t = np.linspace(0.1, 30, 100)
        f = thermalization_efficiency(t)
        assert np.all(f >= 0)
        assert np.all(f <= 1)

    def test_decreasing(self):
        """Efficiency should decrease with time."""
        t = np.linspace(1, 30, 50)
        f = thermalization_efficiency(t)
        assert f[-1] < f[0]


# ── KilonovaComponent ─────────────────────────────────────────────────────────

class TestKilonovaComponent:

    def test_valid_construction(self, blue_component):
        assert blue_component.mej == 0.025
        assert blue_component.vej == 0.27
        assert blue_component.kappa == 0.5

    def test_invalid_vej_raises(self):
        with pytest.raises(ValueError, match="vej must be in"):
            KilonovaComponent(mej=0.025, vej=1.5, kappa=0.5)

    def test_invalid_kappa_raises(self):
        with pytest.raises(ValueError):
            KilonovaComponent(mej=0.025, vej=0.27, kappa=-1.0)

    def test_mej_cgs(self, blue_component):
        """Ejecta mass conversion to CGS."""
        expected = 0.025 * 1.989e33
        assert abs(blue_component.mej_cgs - expected) / expected < 1e-6

    def test_t_diff_positive(self, blue_component):
        assert blue_component.t_diff() > 0

    def test_t_diff_increases_with_kappa(self):
        """Higher opacity → longer diffusion time."""
        c1 = KilonovaComponent(mej=0.03, vej=0.2, kappa=1.0)
        c2 = KilonovaComponent(mej=0.03, vej=0.2, kappa=10.0)
        assert c2.t_diff() > c1.t_diff()

    def test_t_diff_increases_with_mej(self):
        """More massive ejecta → longer diffusion time."""
        c1 = KilonovaComponent(mej=0.01, vej=0.2, kappa=1.0)
        c2 = KilonovaComponent(mej=0.10, vej=0.2, kappa=1.0)
        assert c2.t_diff() > c1.t_diff()

    def test_bolometric_luminosity_positive(self, blue_component, time_array):
        L = blue_component.bolometric_luminosity(time_array)
        assert np.all(L >= 0)

    def test_bolometric_luminosity_finite(self, blue_component, time_array):
        L = blue_component.bolometric_luminosity(time_array)
        assert np.all(np.isfinite(L))

    def test_bolometric_luminosity_peaks_early(self, blue_component):
        """Blue component should peak within ~2 days."""
        t = np.linspace(0.3, 15.0, 100)
        L = blue_component.bolometric_luminosity(t)
        t_peak = t[np.argmax(L)]
        assert t_peak < 5.0, f"Peak at {t_peak:.1f} d, expected < 5 d"

    def test_temperature_positive(self, blue_component, time_array):
        T = blue_component.temperature(time_array)
        assert np.all(T >= 1000.0)   # floor enforced

    def test_temperature_decreasing(self, blue_component):
        """Temperature should generally decline with time."""
        t = np.array([0.5, 2.0, 5.0, 10.0])
        T = blue_component.temperature(t)
        assert T[-1] < T[0]

    def test_magnitude_finite(self, blue_component, time_array):
        mag = blue_component.magnitude(time_array, band="r", distance_mpc=40.0)
        assert np.all(np.isfinite(mag))

    def test_magnitude_range_gw170817(self, blue_component):
        """
        At GW170817 distance (40 Mpc), peak r-band magnitude
        should be roughly 16--20.
        """
        t = np.linspace(0.5, 5.0, 20)
        mag = blue_component.magnitude(t, band="r", distance_mpc=40.0)
        assert mag.min() < 20.0
        assert mag.min() > 14.0

    def test_magnitude_fainter_at_larger_distance(self, blue_component):
        """Source should appear fainter at larger distance."""
        t = np.array([1.0, 2.0, 3.0])
        mag_near = blue_component.magnitude(t, band="r", distance_mpc=40.0)
        mag_far  = blue_component.magnitude(t, band="r", distance_mpc=200.0)
        assert np.all(mag_far > mag_near)

    def test_unknown_band_raises(self, blue_component):
        with pytest.raises(ValueError, match="not recognised"):
            blue_component.magnitude(np.array([1.0]), band="X", distance_mpc=40.0)

    @pytest.mark.parametrize("band", ["g", "r", "i", "z", "J", "H"])
    def test_all_standard_bands(self, blue_component, band):
        """All standard bands should return finite magnitudes."""
        mag = blue_component.magnitude(np.array([1.0, 3.0]), band=band,
                                       distance_mpc=40.0)
        assert np.all(np.isfinite(mag))

    def test_redder_bands_fainter_for_red_component(self, red_component):
        """
        Red component (high kappa) should be brighter in NIR than optical
        at late times (>3 days) — a key kilonova signature.
        """
        t = np.array([4.0, 5.0])
        mag_r = red_component.magnitude(t, band="r", distance_mpc=40.0)
        mag_H = red_component.magnitude(t, band="H", distance_mpc=40.0)
        # H should be brighter (smaller magnitude) than r at late times
        assert np.mean(mag_H) < np.mean(mag_r)


# ── MultiComponentKilonova ────────────────────────────────────────────────────

class TestMultiComponentKilonova:

    def test_construction(self, two_component_model):
        assert len(two_component_model.components) == 2

    def test_empty_components_raises(self):
        with pytest.raises(ValueError):
            MultiComponentKilonova([])

    def test_from_dict(self):
        params = {
            "blue_mej": 0.025, "blue_vej": 0.27, "blue_kappa": 0.5,
            "red_mej":  0.040, "red_vej":  0.15, "red_kappa":  10.0,
        }
        model = MultiComponentKilonova.from_dict(params)
        assert len(model.components) == 2

    def test_total_flux_brighter_than_components(
        self, two_component_model, time_array
    ):
        """Combined flux must exceed any individual component flux."""
        lam = 6230.0   # r-band
        d   = 40.0
        F_total = two_component_model.total_flux(time_array, band="r",
                                                  distance_mpc=d)
        for comp in two_component_model.components:
            F_comp = comp.blackbody_flux(time_array, lam, d)
            assert np.all(F_total >= F_comp - 1e-50)

    def test_combined_magnitude_brighter_than_components(
        self, two_component_model, time_array
    ):
        """Combined magnitude must be <= (brighter) than any single component."""
        mag_total = two_component_model.magnitude(
            time_array, band="r", distance_mpc=40.0
        )
        for comp in two_component_model.components:
            mag_comp = comp.magnitude(time_array, band="r", distance_mpc=40.0)
            assert np.all(mag_total <= mag_comp + 1e-6)

    def test_component_magnitudes_dict(self, two_component_model, time_array):
        comp_mags = two_component_model.component_magnitudes(
            time_array, band="r", distance_mpc=40.0
        )
        assert "blue" in comp_mags
        assert "red"  in comp_mags
        assert comp_mags["blue"].shape == time_array.shape

    def test_magnitude_finite(self, two_component_model, time_array):
        mag = two_component_model.magnitude(
            time_array, band="i", distance_mpc=40.0
        )
        assert np.all(np.isfinite(mag))