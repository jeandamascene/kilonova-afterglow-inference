"""
kai/data/grb211211a_loaders.py

Data loaders for GRB 211211A multiwavelength observations.

Sources
-------
- Swift XRT light curve : UK Swift Science Data Centre (Evans et al. 2007)
  https://www.swift.ac.uk/xrt_curves/01088940/
- Optical/NIR photometry : Rastinejad et al. 2022 (Nature 612, 223)
  + Troja et al. 2022 (Nature 612, 228)
- Compiled dataset : Kunert et al. 2024 (MNRAS 527, 3900)

XRT QDP file format
-------------------
Two datasets (WT mode + PC mode) separated by 'NO NO NO'.
Columns: Time [s], T+ve [s], T-ve [s], Flux [erg/cm2/s],
         Fluxpos [erg/cm2/s], Fluxneg [erg/cm2/s]

Usage
-----
    from kai.data.grb211211a_loaders import (
        load_xrt_lightcurve,
        load_xrt_as_sed,
        load_optical_photometry,
    )

    # XRT light curve (flux vs time)
    df_xrt = load_xrt_lightcurve("data/GRB211211A/xrt_flux_lightcurve.qdp")

    # XRT SED at a specific epoch for SSC inference
    sed = load_xrt_as_sed(df_xrt, t_center=1e4, dt=5e3)

    # Optical/NIR photometry for kilonova inference
    df_opt = load_optical_photometry("data/GRB211211A/")
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── GRB 211211A fixed properties ─────────────────────────────────────────────
GRB211211A = {
    'redshift':    0.0763,
    'distance_mpc': 347.8,               # luminosity distance [Mpc]
    'eiso':        1.25e52,              # erg (Fermi-GBM, Veres+2023)
    't0_mjd':      59559.548,            # MJD of BAT trigger
    'ra':          212.292,              # deg
    'dec':         27.891,              # deg
    'galactic_av': 0.048,               # Galactic extinction [mag]
}

# XRT photon index and count-to-flux conversion
XRT_PHOTON_INDEX  = 1.79    # from UKSSDC automated spectrum
XRT_ENERGY_KEV    = 1.0     # reference energy [keV] for SED point
XRT_ENERGY_EV     = XRT_ENERGY_KEV * 1e3   # [eV]


# ── XRT light curve loader ────────────────────────────────────────────────────

def load_xrt_lightcurve(
    filepath: str,
    mode: str = 'both',
    t_min: float = None,
    t_max: float = None,
) -> pd.DataFrame:
    """
    Load Swift XRT flux light curve from a UKSSDC QDP file.

    The QDP file contains two datasets separated by 'NO NO NO':
    - Dataset 1: WT (Windowed Timing) mode — early times, high count rate
    - Dataset 2: PC (Photon Counting) mode — late times, low count rate

    Parameters
    ----------
    filepath : str
        Path to the XRT flux QDP file (e.g. xrt_flux_lightcurve.qdp)
    mode : str
        Which mode to load: 'WT', 'PC', or 'both'. Default 'both'.
    t_min : float, optional
        Minimum time [s after trigger]. Default: no cut.
    t_max : float, optional
        Maximum time [s after trigger]. Default: no cut.

    Returns
    -------
    df : pd.DataFrame
        Columns:
            t_s      : time [s after trigger]
            t_pos    : positive time error [s]
            t_neg    : negative time error [s] (absolute value)
            flux     : 0.3-10 keV flux [erg/cm^2/s]
            flux_pos : positive flux error [erg/cm^2/s]
            flux_neg : negative flux error [erg/cm^2/s] (absolute value)
            mode     : 'WT' or 'PC'
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"XRT file not found: {filepath}")

    records = []
    current_mode = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Detect mode labels
            if 'WTSLEW data' in line or 'WT data' in line:
                current_mode = 'WT'
                continue
            if 'PC data' in line:
                current_mode = 'PC'
                continue

            # Skip header/comment lines
            if line.startswith('!') or line.startswith('READ') or \
               line.startswith('!Time') or line.startswith('NO') or \
               line.startswith('BINMODE') or not line:
                continue

            # Parse data lines
            parts = line.split()
            if len(parts) >= 6:
                try:
                    t    = float(parts[0])
                    tpos = float(parts[1])
                    tneg = abs(float(parts[2]))
                    flux = float(parts[3])
                    fpos = float(parts[4])
                    fneg = abs(float(parts[5]))

                    # Skip zero or negative flux
                    if flux <= 0:
                        continue

                    records.append({
                        't_s':      t,
                        't_pos':    tpos,
                        't_neg':    tneg,
                        'flux':     flux,
                        'flux_pos': fpos,
                        'flux_neg': fneg,
                        'mode':     current_mode or 'unknown',
                    })
                except (ValueError, IndexError):
                    continue

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(f"No valid data found in {filepath}")

    # Filter by mode
    if mode.upper() == 'WT':
        df = df[df['mode'] == 'WT']
    elif mode.upper() == 'PC':
        df = df[df['mode'] == 'PC']

    # Filter by time
    if t_min is not None:
        df = df[df['t_s'] >= t_min]
    if t_max is not None:
        df = df[df['t_s'] <= t_max]

    df = df.sort_values('t_s').reset_index(drop=True)
    return df


def summarise_xrt(df: pd.DataFrame) -> None:
    """Print a summary of the XRT light curve data."""
    print(f"XRT light curve — {len(df)} points")
    for mode in df['mode'].unique():
        sub = df[df['mode'] == mode]
        print(f"  {mode:3s}: {len(sub):3d} pts | "
              f"t = {sub['t_s'].min():.1e} -- {sub['t_s'].max():.1e} s | "
              f"flux = {sub['flux'].min():.2e} -- {sub['flux'].max():.2e} erg/cm2/s")


# ── XRT to SED converter ──────────────────────────────────────────────────────

def load_xrt_as_sed(
    df_xrt: pd.DataFrame,
    t_center: float,
    dt: float,
    energy_range_kev: tuple = (0.3, 10.0),
    n_energy_points: int = 5,
) -> pd.DataFrame:
    """
    Convert XRT flux light curve to a SED DataFrame for SSCLikelihood.

    Averages the XRT flux over the interval [t_center - dt, t_center + dt]
    and distributes it across energy bins assuming a power-law spectrum
    with photon index Gamma = XRT_PHOTON_INDEX.

    The photon index from the UKSSDC spectrum (Gamma = 1.79) gives:
        F(E) ∝ E^(2 - Gamma) = E^0.21   (nearly flat in E^2 dN/dE)

    Parameters
    ----------
    df_xrt    : pd.DataFrame  XRT light curve from load_xrt_lightcurve()
    t_center  : float         Central time of epoch [s after trigger]
    dt        : float         Half-width of time window [s]
    energy_range_kev : tuple  Min and max energy [keV]. Default (0.3, 10.0).
    n_energy_points  : int    Number of SED points. Default 5.

    Returns
    -------
    df_sed : pd.DataFrame
        Columns: energy_eV, sed, sed_err, instrument
        SED in E^2 dN/dE units [erg/s/cm^2]
    """
    # Select data in time window
    mask = (
        (df_xrt['t_s'] >= t_center - dt) &
        (df_xrt['t_s'] <= t_center + dt)
    )
    sub = df_xrt[mask]

    if len(sub) == 0:
        raise ValueError(
            f"No XRT data in window t={t_center:.2e} ± {dt:.2e} s. "
            f"Available range: {df_xrt['t_s'].min():.2e} -- "
            f"{df_xrt['t_s'].max():.2e} s"
        )

    # Average flux weighted by inverse variance
    weights  = 1.0 / (sub['flux_pos']**2 + sub['flux_neg']**2)
    flux_avg = np.average(sub['flux'], weights=weights)
    flux_err = 1.0 / np.sqrt(weights.sum())

    # Energy grid [keV → eV]
    e_min, e_max = energy_range_kev
    energies_kev = np.logspace(np.log10(e_min), np.log10(e_max), n_energy_points)
    energies_ev  = energies_kev * 1e3

    # Spectral shape: E^2 dN/dE ∝ E^(2 - Gamma)
    # Normalize to match integrated band flux
    spectral_index = 2.0 - XRT_PHOTON_INDEX   # = 0.21 for Gamma=1.79
    shape  = (energies_kev / XRT_ENERGY_KEV) ** spectral_index
    shape /= shape.mean()   # normalize

    sed     = flux_avg * shape
    sed_err = flux_err * shape

    return pd.DataFrame({
        'energy_eV':  energies_ev,
        'sed':        sed,
        'sed_err':    sed_err,
        'instrument': 'XRT',
        't_center_s': t_center,
    })


# ── Optical/NIR photometry ────────────────────────────────────────────────────

# Rastinejad+2022 Table 1 — optical/NIR photometry for GRB 211211A
# Columns: t_days, band, mag, mag_err
# AB magnitudes, not corrected for galactic extinction
# Source: Rastinejad et al. 2022, Nature 612, 223, Table 1
RASTINEJAD2022_DATA = """t_days,band,mag,mag_err
0.043,uvw2,19.28,0.05
0.043,uvm2,19.22,0.07
0.043,uvw1,18.79,0.04
0.043,U,18.15,0.04
0.043,B,18.28,0.04
0.043,V,17.99,0.04
0.208,uvw2,21.03,0.10
0.208,uvm2,20.98,0.13
0.208,uvw1,20.37,0.07
0.208,U,19.73,0.06
0.208,B,19.72,0.05
0.208,V,19.26,0.05
0.208,r,19.19,0.05
0.208,i,19.00,0.05
0.417,g,20.83,0.10
0.417,r,20.15,0.06
0.417,i,19.63,0.05
0.417,z,19.33,0.06
0.417,J,19.02,0.07
0.417,H,18.82,0.08
0.417,K,18.65,0.10
1.354,g,23.10,0.20
1.354,r,22.11,0.10
1.354,i,21.34,0.08
1.354,z,20.87,0.09
1.354,J,20.38,0.10
1.354,H,19.95,0.10
1.354,K,19.71,0.12
4.188,r,24.50,0.30
4.188,i,23.45,0.20
4.188,z,22.85,0.20
4.188,J,22.10,0.25
4.188,H,21.50,0.25
"""


def load_fermi_lat_grb211211a() -> pd.DataFrame:
    """
    Fermi-LAT detection of GRB 211211A afterglow.

    From Zhang et al. 2022, ApJL 933, L22 (Table 1 + Figure 3).
    Time-averaged flux in 395-30780 s after BAT trigger.

    Individual photon detections (probability > 90%):
        T0 + 6438 s  : 207 MeV
        T0 + 6648 s  : 188 MeV
        T0 + 12494 s : 164 MeV
        T0 + 12967 s : 1740 MeV  (highest energy)
        T0 + 13054 s : 103 MeV
        T0 + 17410 s : 114 MeV
        T0 + 18128 s : 231 MeV

    Average:
        Flux  = (3.23 ± 0.86) × 10^-10 erg/cm^2/s  (100 MeV - 10 GeV)
        Gamma = -3.30 ± 0.45
        Time  = 395 - 30780 s

    Note: LAT boresight angle was 106.5 deg at trigger — no prompt detection.
    LAT entered FOV at ~395 s. Quasi-simultaneous with XRT Epoch 1 (3519-4972 s).

    Returns
    -------
    pd.DataFrame with columns: energy_eV, sed, sed_err, instrument, t_center_s
    """
    # Time bins from Zhang+2022 Figure 3 (left panel, red data points)
    # Approximate flux values read from the figure
    # Units: erg/s/cm^2 (E^2 dN/dE)
    return pd.DataFrame({
        # Use the time-averaged measurement as a single SED point
        # Effective energy: geometric mean of 100 MeV - 10 GeV = ~1 GeV
        'energy_eV':  [1.0e9],                # 1 GeV effective energy
        'sed':        [3.23e-10],              # erg/s/cm^2
        'sed_err':    [0.86e-10],
        'instrument': ['Fermi-LAT'],
        't_center_s': [15588.0],              # midpoint of 395-30780 s
    })


def load_epoch1_dataset() -> dict:
    """
    Load complete Epoch 1 dataset: XRT SED + Fermi-LAT point.

    Returns
    -------
    dict with keys 'xrt' and 'lat', each a pd.DataFrame
    """
    df_xrt_full = load_xrt_lightcurve("data/GRB211211A/xrt_flux_lightcurve.qdp")
    df_xrt_sed  = load_xrt_as_sed(df_xrt_full, t_center=4200.0, dt=800.0)
    df_lat      = load_fermi_lat_grb211211a()
    return {'xrt': df_xrt_sed, 'lat': df_lat}


def load_epoch1_combined_sed() -> pd.DataFrame:
    """
    Combined XRT + Fermi-LAT SED for Epoch 1 SSC inference.
    Covers 0.3 keV to 10 GeV — 7 decades in energy.

    Returns
    -------
    pd.DataFrame sorted by energy_eV
    """
    ds    = load_epoch1_dataset()
    cols  = ['energy_eV', 'sed', 'sed_err', 'instrument']
    combined = pd.concat([ds['xrt'][cols], ds['lat'][cols]], ignore_index=True)
    return combined.sort_values('energy_eV').reset_index(drop=True)


def load_optical_photometry(
    bands: list = None,
    t_min: float = 0.0,
    t_max: float = 10.0,
    correct_extinction: bool = True,
    source: str = 'rastinejad2022',
) -> pd.DataFrame:
    """
    Load GRB 211211A optical/NIR photometry for kilonova inference.

    Parameters
    ----------
    bands  : list of str, optional
        Filter subset, e.g. ['g', 'r', 'i', 'z', 'J', 'H'].
        If None, returns all available bands.
    t_min  : float  Minimum time [days]. Default 0.
    t_max  : float  Maximum time [days]. Default 10.
    correct_extinction : bool
        Apply Galactic extinction correction. Default True.
        Uses E(B-V) = 0.048 mag (Schlafly & Finkbeiner 2011).
    source : str  Data source. Currently 'rastinejad2022'.

    Returns
    -------
    df : pd.DataFrame
        Columns: t_days, band, mag, mag_err, source
        AB magnitudes (extinction corrected if correct_extinction=True).
    """
    import io
    if source == 'rastinejad2022':
        df = pd.read_csv(io.StringIO(RASTINEJAD2022_DATA))
        df['source'] = 'Rastinejad+2022'
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'rastinejad2022'.")

    # Apply Galactic extinction correction
    if correct_extinction:
        # Extinction coefficients A_lambda/E(B-V) from Cardelli+1989
        # for standard optical/NIR bands
        extinction_coeffs = {
            'uvw2': 8.90, 'uvm2': 9.30, 'uvw1': 6.60,
            'U': 4.72, 'B': 4.07, 'V': 3.10,
            'g': 3.60, 'r': 2.63, 'i': 1.98,
            'z': 1.49, 'J': 0.87, 'H': 0.56, 'K': 0.37,
        }
        ebv = GRB211211A['galactic_av'] / 3.1   # E(B-V) from A_V
        for band, rv in extinction_coeffs.items():
            mask = df['band'] == band
            df.loc[mask, 'mag'] -= rv * ebv

    # Apply filters
    mask = (df['t_days'] >= t_min) & (df['t_days'] <= t_max)
    if bands is not None:
        mask &= df['band'].isin(bands)

    df = df[mask].sort_values('t_days').reset_index(drop=True)

    if len(df) == 0:
        import warnings
        warnings.warn("No data points match the selection criteria.")

    return df


def load_kilonova_epoch(
    t_center_days: float,
    dt_days: float = 0.5,
    bands: list = None,
) -> pd.DataFrame:
    """
    Load optical/NIR data for a specific kilonova epoch.

    Convenience wrapper around load_optical_photometry that selects
    data within [t_center - dt, t_center + dt] days.

    Parameters
    ----------
    t_center_days : float  Central time [days after trigger]
    dt_days       : float  Half-width of time window [days]. Default 0.5.
    bands         : list   Filter subset.

    Returns
    -------
    df : pd.DataFrame
    """
    return load_optical_photometry(
        bands  = bands,
        t_min  = t_center_days - dt_days,
        t_max  = t_center_days + dt_days,
    )


# ── Joint dataset builder ─────────────────────────────────────────────────────

def build_joint_dataset(
    xrt_file: str,
    epoch_days: float = 1.354,
    kn_bands: list = None,
    xrt_dt_s: float = 5e4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build matched XRT + optical datasets for a given epoch.

    Matches the kilonova epoch to the corresponding XRT time window
    for joint KN+SSC inference.

    Parameters
    ----------
    xrt_file    : str    Path to XRT QDP file
    epoch_days  : float  Target epoch [days]. Default 1.354 (~1.4 days).
    kn_bands    : list   Optical bands for KN fit. Default all.
    xrt_dt_s    : float  XRT time window half-width [s]. Default 5e4.

    Returns
    -------
    df_kn  : pd.DataFrame  Kilonova photometry
    df_xrt_sed : pd.DataFrame  XRT SED at the matched epoch
    """
    # Convert epoch to seconds
    t_center_s = epoch_days * 86400.0

    # Load XRT light curve
    df_xrt = load_xrt_lightcurve(xrt_file)

    # Build XRT SED at the epoch
    df_xrt_sed = load_xrt_as_sed(
        df_xrt,
        t_center = t_center_s,
        dt       = xrt_dt_s,
    )

    # Load kilonova photometry near this epoch
    df_kn = load_kilonova_epoch(
        t_center_days = epoch_days,
        dt_days       = 0.5,
        bands         = kn_bands,
    )

    print(f"Joint dataset at t ~ {epoch_days:.2f} days:")
    print(f"  KN  : {len(df_kn)} points in bands "
          f"{sorted(df_kn['band'].unique())}")
    print(f"  XRT : {len(df_xrt_sed)} SED points, "
          f"flux = {df_xrt_sed['sed'].mean():.2e} erg/s/cm^2")

    return df_kn, df_xrt_sed


def summarise_dataset(df: pd.DataFrame) -> None:
    """Print a human-readable summary of any photometry DataFrame."""
    print(f"Total data points : {len(df)}")
    if 't_days' in df.columns:
        print(f"Time range        : {df['t_days'].min():.3f} -- "
              f"{df['t_days'].max():.3f} days")
    if 'band' in df.columns:
        print(f"Bands             : {sorted(df['band'].unique())}")
    if 'mag' in df.columns:
        print(f"Mag range         : {df['mag'].min():.1f} -- "
              f"{df['mag'].max():.1f}")
    if 'sed' in df.columns:
        print(f"SED range         : {df['sed'].min():.2e} -- "
              f"{df['sed'].max():.2e} erg/s/cm^2")
