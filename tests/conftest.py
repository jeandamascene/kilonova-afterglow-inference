"""
Shared pytest fixtures for the kai test suite.
"""

import numpy as np
import pandas as pd
import pytest

from kai.models.kilonova import KilonovaComponent, MultiComponentKilonova


@pytest.fixture
def blue_component():
    """Standard blue kilonova component (Villar+2017 best-fit)."""
    return KilonovaComponent(mej=0.025, vej=0.27, kappa=0.5, name="blue")


@pytest.fixture
def red_component():
    """Standard red kilonova component (Villar+2017 best-fit)."""
    return KilonovaComponent(mej=0.040, vej=0.15, kappa=10.0, name="red")


@pytest.fixture
def two_component_model(blue_component, red_component):
    """Two-component kilonova model."""
    return MultiComponentKilonova([blue_component, red_component])


@pytest.fixture
def time_array():
    """Standard time array for tests [days]."""
    return np.linspace(0.5, 15.0, 30)


@pytest.fixture
def mock_photometry():
    """
    Minimal mock GW170817-like photometry DataFrame.
    Mimics the structure of real data without requiring network access.
    """
    return pd.DataFrame({
        "t_days":  [0.5,  1.0,  2.0,  3.0,  5.0,
                    0.5,  1.0,  2.0,  3.0,  5.0],
        "band":    ["r",  "r",  "r",  "r",  "r",
                    "J",  "J",  "J",  "J",  "J"],
        "mag":     [17.2, 17.5, 18.4, 19.2, 20.5,
                    17.1, 17.0, 17.1, 17.2, 17.6],
        "mag_err": [0.05, 0.05, 0.05, 0.08, 0.10,
                    0.05, 0.05, 0.05, 0.06, 0.08],
    })