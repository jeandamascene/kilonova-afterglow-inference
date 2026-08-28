"""
Unit tests for kai.data.loaders.
"""

import pandas as pd
import pytest

from kai.data.loaders import (
    load_gw170817_photometry,
    load_upper_limits,
    summarise_dataset,
)

REQUIRED_COLUMNS = {"t_days", "band", "mag", "mag_err"}


class TestLoadGW170817Photometry:

    def test_returns_dataframe(self):
        df = load_gw170817_photometry()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = load_gw170817_photometry()
        assert REQUIRED_COLUMNS.issubset(df.columns)

    def test_no_nan_values(self):
        df = load_gw170817_photometry()
        assert not df[list(REQUIRED_COLUMNS)].isnull().any().any()

    def test_positive_uncertainties(self):
        df = load_gw170817_photometry()
        assert (df["mag_err"] > 0).all()

    def test_positive_times(self):
        df = load_gw170817_photometry()
        assert (df["t_days"] > 0).all()

    def test_band_filter(self):
        df = load_gw170817_photometry(bands=["r", "J"])
        assert set(df["band"].unique()) == {"r", "J"}

    def test_time_filter(self):
        df = load_gw170817_photometry(t_max=3.0)
        assert df["t_days"].max() <= 3.0

    def test_sorted_by_time(self):
        df = load_gw170817_photometry()
        assert (df["t_days"].diff().dropna() >= 0).all()

    def test_empty_result_warns(self):
        with pytest.warns(UserWarning):
            load_gw170817_photometry(bands=["X"])

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            load_gw170817_photometry(source="unknown")


class TestLoadUpperLimits:

    def test_returns_dataframe(self):
        df = load_upper_limits()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = load_upper_limits()
        assert {"t_days", "band", "mag_limit"}.issubset(df.columns)

    def test_band_filter(self):
        df = load_upper_limits(band="J")
        assert (df["band"] == "J").all()


class TestSummariseDataset:

    def test_runs_without_error(self, capsys):
        df = load_gw170817_photometry()
        summarise_dataset(df)
        captured = capsys.readouterr()
        assert "Total data points" in captured.out