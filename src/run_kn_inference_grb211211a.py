"""
run_kn_inference_grb211211a.py

Stage 2: Kilonova Bayesian inference for GRB 211211A.

Uses SSC posterior samples from Stage 1 to subtract the afterglow
contribution from optical/NIR photometry, then fits the residual
with the KAI kilonova model using bilby/dynesty.

Three epochs are analysed:
    Epoch 2 (t ~ 0.208 days): optical/NIR + afterglow subtraction
    Epoch 3 (t ~ 0.417 days): optical/NIR + afterglow subtraction
    Epoch 4 (t ~ 1.354 days): optical/NIR (afterglow negligible)

Usage
-----
    python src/run_kn_inference_grb211211a.py --epoch 2 --nlive 300
    python src/run_kn_inference_grb211211a.py --epoch 3 --nlive 300
    python src/run_kn_inference_grb211211a.py --epoch 4 --nlive 300

Requires
--------
    Stage 1 output: results/grb211211a/epoch1_ssc/grb211211a_epoch1_posteriors.npy
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as con
from astropy.cosmology import WMAP9 as cosmo

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from kai.models.kilonova import MultiComponentKilonova
from kai.models.ssc import GRBShock, SSCModel
from kai.inference.likelihood import KilonovaLikelihood
from kai.inference.priors import gw170817_priors
from kai.inference.sampler import run_inference
from kai.data.grb211211a_loaders import (
    load_optical_photometry, GRB211211A,
)
import bilby

# ── Configuration ─────────────────────────────────────────────────────────────
DIST_MPC  = GRB211211A['distance_mpc']
REDSHIFT  = GRB211211A['redshift']
EISO      = GRB211211A['eiso']
DENSITY   = 1e-3

SSC_POSTERIORS = Path(
    "results/grb211211a/epoch1_ssc/grb211211a_epoch1_posteriors.npy"
)
OUTDIR_BASE = "results/grb211211a"

# ── Epoch definitions ─────────────────────────────────────────────────────────
EPOCHS = {
    2: {
        'label':      'epoch2_kn',
        't_days':      0.208,
        'dt_days':     0.05,
        'bands':      ['B', 'V', 'r', 'i', 'uvw1', 'uvm2', 'uvw2', 'U'],
        'tstart_s':    15975.0,
        'tstop_s':     22089.0,
        'subtract_ag': True,
    },
    3: {
        'label':      'epoch3_kn',
        't_days':      0.417,
        'dt_days':     0.10,
        'bands':      ['g', 'r', 'i', 'z', 'J', 'H', 'K'],
        'subtract_ag': True,
        'tstart_s':    30000.0,
        'tstop_s':     40000.0,
    },
    4: {
        'label':      'epoch4_kn',
        't_days':      1.354,
        'dt_days':     0.20,
        'bands':      ['g', 'r', 'i', 'z', 'J', 'H', 'K'],
        'subtract_ag': False,   # afterglow negligible at ~1.4 days
    },
}

# ── Filter effective wavelengths [Angstrom] ───────────────────────────────────
FILTER_WAVELENGTHS_AA = {
    'uvw2': 1928.0, 'uvm2': 2246.0, 'uvw1': 2600.0,
    'U': 3465.0, 'B': 4392.0, 'V': 5468.0,
    'g': 4770.0, 'r': 6230.0, 'i': 7630.0,
    'z': 9130.0, 'J': 12200.0, 'H': 16300.0, 'K': 21900.0,
}


# ── Afterglow subtraction ─────────────────────────────────────────────────────

def load_ssc_posteriors() -> dict:
    """Load Stage 1 SSC posterior samples."""
    if not SSC_POSTERIORS.exists():
        raise FileNotFoundError(
            f"SSC posteriors not found: {SSC_POSTERIORS}\n"
            "Run Stage 1 first: python src/run_ssc_naima_grb211211a.py"
        )
    posteriors = np.load(SSC_POSTERIORS, allow_pickle=True).item()
    n_samples  = len(posteriors['log10_eta_e'])
    print(f"Loaded {n_samples} SSC posterior samples from Stage 1.")
    return posteriors


def predict_afterglow_mag(
    ssc_posteriors: dict,
    t_days: float,
    bands: list,
    shock_tstart: float,
    shock_tstop: float,
    n_samples: int = 500,
) -> dict:
    """
    Predict afterglow magnitude at a given epoch using SSC posterior samples.

    Returns the median and 1-sigma uncertainty of the afterglow contribution
    in each band, for subtraction from the observed photometry.

    Parameters
    ----------
    ssc_posteriors : dict   Stage 1 posterior samples
    t_days         : float  Observation epoch [days]
    bands          : list   Filter bands
    shock_tstart   : float  Shock start time [s]
    shock_tstop    : float  Shock stop time [s]
    n_samples      : int    Number of posterior samples to use

    Returns
    -------
    dict mapping band -> {'med': float, 'std': float}  [AB magnitudes]
    """
    # Build shock for this epoch
    shock = GRBShock(
        eiso     = EISO,
        density  = DENSITY,
        tstart   = shock_tstart,
        tstop    = shock_tstop,
        redshift = REDSHIFT,
        scenario = 'ISM',
    )

    n_post = len(ssc_posteriors['log10_eta_e'])
    idx    = np.random.choice(n_post, size=min(n_samples, n_post), replace=False)

    ag_mags = {band: [] for band in bands}

    for i in idx:
        params = {
            'log10_eta_e':  ssc_posteriors['log10_eta_e'][i],
            'log10_ebreak': ssc_posteriors['log10_ebreak'][i],
            'alpha2':       ssc_posteriors['alpha2'][i],
            'log10_ecut':   ssc_posteriors['log10_ecut'][i],
            'log10_B':      ssc_posteriors['log10_B'][i],
        }
        try:
            model = SSCModel(shock, absorption_method=2)
            for band in bands:
                if band not in FILTER_WAVELENGTHS_AA:
                    continue
                lam_aa = FILTER_WAVELENGTHS_AA[band]
                e_eV   = (4.136e-15 * 3e18 / lam_aa)   # h*nu in eV

                energy = np.array([e_eV]) * u.eV
                sed    = model.sed(energy, params)
                f_nu   = sed.to('erg/(s cm^2)').value[0] / (3e18 / lam_aa)
                if f_nu > 0:
                    mag = -2.5 * np.log10(f_nu) - 48.6
                    ag_mags[band].append(mag)
        except Exception:
            continue

    result = {}
    for band in bands:
        mags = ag_mags[band]
        if len(mags) > 10:
            result[band] = {
                'med': float(np.median(mags)),
                'std': float(np.std(mags)),
                'n':   len(mags),
            }
        else:
            result[band] = None   # insufficient samples

    return result


def subtract_afterglow(
    df_obs: pd.DataFrame,
    ag_prediction: dict,
) -> pd.DataFrame:
    """
    Subtract afterglow contribution from observed photometry.

    Converts magnitudes to flux, subtracts afterglow flux, converts back.
    Propagates uncertainties in quadrature.

    Parameters
    ----------
    df_obs        : pd.DataFrame  Observed photometry
    ag_prediction : dict          Afterglow mag predictions per band

    Returns
    -------
    df_kn : pd.DataFrame  Kilonova photometry (afterglow subtracted)
    """
    rows = []
    for _, row in df_obs.iterrows():
        band = row['band']
        ag   = ag_prediction.get(band)

        if ag is None:
            # No afterglow prediction — use observed flux directly
            rows.append(row.to_dict())
            continue

        # Convert observed mag to flux density [arbitrary units]
        f_obs = 10.0**(-row['mag'] / 2.5)
        f_ag  = 10.0**(-ag['med'] / 2.5)
        f_kn  = f_obs - f_ag

        if f_kn <= 0:
            # Afterglow dominates — skip this data point
            print(f"  Warning: afterglow dominates in {band} at "
                  f"t={row['t_days']:.3f} days — skipping")
            continue

        # Propagate uncertainty
        df_obs_flux = f_obs * row['mag_err'] * np.log(10) / 2.5
        df_ag_flux  = f_ag * ag['std'] * np.log(10) / 2.5
        df_kn_flux  = np.sqrt(df_obs_flux**2 + df_ag_flux**2)

        mag_kn    = -2.5 * np.log10(f_kn)
        mag_err_kn = df_kn_flux / (f_kn * np.log(10) / 2.5)

        new_row = row.to_dict()
        new_row['mag']     = mag_kn
        new_row['mag_err'] = mag_err_kn
        new_row['source']  = row['source'] + ' (AG subtracted)'
        rows.append(new_row)

    df_kn = pd.DataFrame(rows).reset_index(drop=True)
    return df_kn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KN inference for GRB 211211A with afterglow subtraction"
    )
    parser.add_argument(
        '--epoch', type=int, default=4, choices=[2, 3, 4],
        help="Epoch to analyse (2=0.208d, 3=0.417d, 4=1.354d). Default: 4"
    )
    parser.add_argument(
        '--nlive', type=int, default=300,
        help="Dynesty live points. Default: 300"
    )
    parser.add_argument(
        '--clean', action='store_true',
        help="Re-run even if cached results exist"
    )
    parser.add_argument(
        '--no-subtract', action='store_true',
        help="Skip afterglow subtraction (fit observed mags directly)"
    )
    args = parser.parse_args()

    epoch  = EPOCHS[args.epoch]
    outdir = Path(OUTDIR_BASE) / epoch['label']
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GRB 211211A — KN inference")
    print(f"Epoch {args.epoch}: t ~ {epoch['t_days']:.3f} days")
    print(f"Bands : {epoch['bands']}")
    print(f"nlive : {args.nlive}")
    print(f"Output: {outdir}")
    print(f"{'='*60}\n")

    # ── Load optical photometry ───────────────────────────────────────────────
    df_obs = load_optical_photometry(
        bands  = epoch['bands'],
        t_min  = epoch['t_days'] - epoch['dt_days'],
        t_max  = epoch['t_days'] + epoch['dt_days'],
    )
    print(f"Observed photometry: {len(df_obs)} points in "
          f"{sorted(df_obs['band'].unique())}")

    # ── Afterglow subtraction ─────────────────────────────────────────────────
    if epoch['subtract_ag'] and not args.no_subtract:
        print("\nLoading SSC posteriors for afterglow subtraction...")
        ssc_post = load_ssc_posteriors()

        print(f"Predicting afterglow at t={epoch['t_days']:.3f} days...")
        ag_pred = predict_afterglow_mag(
            ssc_posteriors = ssc_post,
            t_days         = epoch['t_days'],
            bands          = epoch['bands'],
            shock_tstart   = epoch.get('tstart_s', epoch['t_days'] * 86400 * 0.9),
            shock_tstop    = epoch.get('tstop_s',  epoch['t_days'] * 86400 * 1.1),
        )

        print("\nAfterglowcontribution per band:")
        for band, pred in ag_pred.items():
            if pred:
                print(f"  {band:6s}: {pred['med']:.2f} ± {pred['std']:.2f} mag "
                      f"({pred['n']} samples)")
            else:
                print(f"  {band:6s}: insufficient samples")

        df_kn = subtract_afterglow(df_obs, ag_pred)
        print(f"\nAfter subtraction: {len(df_kn)} valid KN data points")
    else:
        df_kn = df_obs
        print("Skipping afterglow subtraction.")

    if len(df_kn) == 0:
        print("ERROR: No data points after afterglow subtraction!")
        return

    print(df_kn[['t_days', 'band', 'mag', 'mag_err']].to_string())

    # ── Build likelihood and priors ───────────────────────────────────────────
    likelihood = KilonovaLikelihood(
        data          = df_kn,
        n_components  = 2,
        distance_mpc  = DIST_MPC,
        sigma_floor   = 0.05,
    )
    priors = gw170817_priors(n_components=2)

    # ── Pre-flight check ──────────────────────────────────────────────────────
    print("\n--- Pre-flight check ---")
    finite = 0
    for _ in range(20):
        s = priors.sample()
        for k, v in s.items():
            likelihood.parameters[k] = v
        ln_l = likelihood.log_likelihood()
        if np.isfinite(ln_l):
            finite += 1
    print(f"Finite likelihoods: {finite}/20")

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"\nRunning KN inference (nlive={args.nlive})...")
    result = run_inference(
        likelihood = likelihood,
        priors     = priors,
        outdir     = str(outdir),
        label      = epoch['label'],
        sampler    = 'dynesty',
        nlive      = args.nlive,
        clean      = args.clean,
        sample     = 'rwalk',
        walks      = 50,
        dlogz      = 0.1,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Posterior credible intervals (median ± 1σ):")
    for param in priors.keys():
        samples      = result.posterior[param]
        lo, med, hi = np.percentile(samples, [16, 50, 84])
        print(f"  {param:22s} = {med:.4f} + {hi-med:.4f} - {med-lo:.4f}")

    print(f"\nlog Evidence (logZ) = "
          f"{result.log_evidence:.2f} ± {result.log_evidence_err:.2f}")

    # ── Corner plot ───────────────────────────────────────────────────────────
    result.plot_corner(
        parameters = list(priors.keys()),
        quantiles  = [0.16, 0.84],
        save       = True,
        filename   = str(outdir / f"{epoch['label']}_corner.pdf"),
    )

    # ── Posterior predictive lightcurves ──────────────────────────────────────
    from kai.inference.likelihood import build_model_from_params, COMPONENT_NAMES
    t_plot = np.linspace(0.05, 5.0, 200)
    bands  = sorted(df_kn['band'].unique())
    n_cols = min(4, len(bands))
    n_rows = (len(bands) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             sharex=True)
    axes = np.array(axes).flatten()

    post_samples = result.posterior.sample(100)
    for ax, band in zip(axes, bands):
        for _, row in post_samples.iterrows():
            model = build_model_from_params(row.to_dict(), COMPONENT_NAMES[2])
            mag   = model.magnitude(t_plot, band=band, distance_mpc=DIST_MPC)
            ax.plot(t_plot, mag, color='steelblue', alpha=0.08, lw=0.8)

        bd = df_kn[df_kn['band'] == band]
        ax.errorbar(bd['t_days'], bd['mag'], yerr=bd['mag_err'],
                    fmt='ko', ms=5, capsize=2, zorder=5)
        ax.invert_yaxis()
        ax.set_title(band)
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("AB mag")

    for ax in axes[len(bands):]:
        ax.set_visible(False)

    plt.suptitle(
        f"GRB 211211A — KN posterior predictive (t~{epoch['t_days']:.3f}d)",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(outdir / f"{epoch['label']}_lightcurves.pdf",
                bbox_inches='tight')
    plt.show()

    print(f"\nDone. Results saved to {outdir}/")
    return result


if __name__ == "__main__":
    main()
