"""
run_joint_inference_grb211211a.py

Joint kilonova + SSC afterglow Bayesian inference for GRB 211211A.

Three analysis epochs:
    Epoch 1 (t ~ 0.043 days): SSC-only  — XRT + UVOT
    Epoch 2 (t ~ 0.208 days): Joint     — XRT + optical/NIR  [MAIN RESULT]
    Epoch 3 (t ~ 0.417 days): KN-only   — optical/NIR + afterglow subtraction

GRB 211211A properties
-----------------------
    Redshift    : 0.0763 (Rastinejad+2022)
    Distance    : 347.8 Mpc
    Eiso        : 1.25e52 erg (Fermi-GBM, Veres+2023)
    T90         : 34.3 s
    Host galaxy : z=0.0763, low-density halo environment

Run
---
    python src/run_joint_inference_grb211211a.py --epoch 2 --nlive 200
    python src/run_joint_inference_grb211211a.py --epoch 1 --nlive 200
    python src/run_joint_inference_grb211211a.py --epoch 3 --nlive 200
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from kai.models.ssc import GRBShock, SSCModel
from kai.models.kilonova import KilonovaComponent, MultiComponentKilonova
from kai.inference.likelihood import KilonovaLikelihood
from kai.inference.ssc_likelihood import (
    SSCLikelihood, JointLikelihood,
    ssc_priors_grb211211a, joint_priors_grb211211a,
)
from kai.inference.priors import gw170817_priors
from kai.inference.sampler import run_inference
from kai.data.grb211211a_loaders import (
    load_xrt_lightcurve, load_xrt_as_sed,
    load_optical_photometry, GRB211211A,
)
import bilby

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
XRT_FILE   = "data/GRB211211A/xrt_flux_lightcurve.qdp"
OUTDIR_BASE = "results/grb211211a"
REDSHIFT   = GRB211211A['redshift']
DIST_MPC   = GRB211211A['distance_mpc']
EISO       = GRB211211A['eiso']

# Density: short GRBs typically in low-density halo environments
# Literature values for GRB 211211A: n ~ 1e-3 to 1e-2 cm^-3
DENSITY = 1e-3   # cm^-3

# ── Epoch definitions ─────────────────────────────────────────────────────────
EPOCHS = {
    1: {
        'label':       'epoch1_ssc',
        'description': 'SSC-only inference (XRT + UVOT, t~0.043d)',
        't_days':       0.043,
        'tstart_s':     3519.0,
        'tstop_s':      4972.0,
        'xrt_t_center': 4200.0,
        'xrt_dt':       800.0,
        'kn_bands':     ['B', 'U', 'V', 'uvw1', 'uvm2', 'uvw2'],
        'kn_t_center':  0.043,
        'kn_dt':        0.02,
        'mode':         'ssc',
    },
    2: {
        'label':       'epoch2_joint',
        'description': 'Joint KN+SSC inference (XRT + optical, t~0.208d)',
        't_days':       0.208,
        'tstart_s':     15975.0,
        'tstop_s':      22089.0,
        'xrt_t_center': 18000.0,
        'xrt_dt':       6000.0,
        'kn_bands':     ['B', 'V', 'r', 'i', 'uvw1', 'uvm2', 'uvw2', 'U'],
        'kn_t_center':  0.208,
        'kn_dt':        0.05,
        'mode':         'joint',
    },
    3: {
        'label':       'epoch3_kn',
        'description': 'KN-only inference (optical/NIR, t~0.417d)',
        't_days':       0.417,
        'kn_bands':     ['g', 'r', 'i', 'z', 'J', 'H', 'K'],
        'kn_t_center':  0.417,
        'kn_dt':        0.1,
        'mode':         'kn',
    },
}


# ── Shock builder ─────────────────────────────────────────────────────────────

def build_shock(epoch: dict) -> GRBShock:
    """Build GRBShock for a given epoch."""
    return GRBShock(
        eiso      = EISO,
        density   = DENSITY,
        tstart    = epoch['tstart_s'],
        tstop     = epoch['tstop_s'],
        redshift  = REDSHIFT,
        scenario  = 'ISM',
    )


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_epoch_data(epoch: dict) -> tuple:
    """
    Load XRT SED and/or optical photometry for a given epoch.

    Returns
    -------
    df_xrt_sed : pd.DataFrame or None
    df_kn      : pd.DataFrame or None
    """
    df_xrt_sed = None
    df_kn      = None

    # Load XRT SED if needed
    if epoch['mode'] in ('ssc', 'joint'):
        df_xrt = load_xrt_lightcurve(XRT_FILE)
        df_xrt_sed = load_xrt_as_sed(
            df_xrt,
            t_center = epoch['xrt_t_center'],
            dt       = epoch['xrt_dt'],
        )
        print(f"XRT SED ({len(df_xrt_sed)} pts): "
              f"flux = {df_xrt_sed['sed'].mean():.3e} erg/s/cm2")

    # Load optical photometry if needed
    if epoch['mode'] in ('kn', 'joint'):
        df_kn = load_optical_photometry(
            bands  = epoch['kn_bands'],
            t_min  = epoch['kn_t_center'] - epoch['kn_dt'],
            t_max  = epoch['kn_t_center'] + epoch['kn_dt'],
        )
        print(f"Optical ({len(df_kn)} pts): "
              f"bands = {sorted(df_kn['band'].unique())}")

    return df_xrt_sed, df_kn


# ── Likelihood builders ───────────────────────────────────────────────────────

def build_ssc_likelihood(df_xrt_sed: pd.DataFrame,
                         shock: GRBShock) -> SSCLikelihood:
    """Build SSCLikelihood from XRT SED data."""
    return SSCLikelihood(
        data              = df_xrt_sed,
        shock             = shock,
        sigma_floor       = 0.1,
        absorption_method = 2,
    )


def build_kn_likelihood(df_kn: pd.DataFrame) -> KilonovaLikelihood:
    """Build KilonovaLikelihood from optical/NIR photometry."""
    return KilonovaLikelihood(
        data          = df_kn,
        n_components  = 2,
        distance_mpc  = DIST_MPC,
        sigma_floor   = 0.05,
    )


def build_joint_likelihood(
    df_xrt_sed: pd.DataFrame,
    df_kn: pd.DataFrame,
    shock: GRBShock,
) -> JointLikelihood:
    """Build JointLikelihood combining KN + SSC."""
    kn_lk  = build_kn_likelihood(df_kn)
    ssc_lk = build_ssc_likelihood(df_xrt_sed, shock)
    return JointLikelihood(kn_lk, ssc_lk)


# ── Prior builders ────────────────────────────────────────────────────────────

def build_priors(mode: str) -> bilby.core.prior.PriorDict:
    """Build priors for the given inference mode."""
    if mode == 'ssc':
        return ssc_priors_grb211211a()
    elif mode == 'kn':
        return gw170817_priors(n_components=2)
    elif mode == 'joint':
        return joint_priors_grb211211a(n_kn_components=2)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Pre-flight check ──────────────────────────────────────────────────────────

def preflight_check(likelihood, priors, n_samples: int = 10):
    """Test likelihood at prior samples before running sampler."""
    print("\n--- Pre-flight likelihood check ---")

    # Test at a fixed reference point
    sample = priors.sample()
    for k, v in sample.items():
        likelihood.parameters[k] = v

    ln_l = likelihood.log_likelihood()
    print(f"  Reference sample log_likelihood : {ln_l:.4f}")
    print(f"  Finite                          : {np.isfinite(ln_l)}")

    # Test n_samples random prior samples
    finite_count = 0
    ln_l_values  = []
    for _ in range(n_samples):
        sample = priors.sample()
        for k, v in sample.items():
            likelihood.parameters[k] = v
        ln_l = likelihood.log_likelihood()
        if np.isfinite(ln_l):
            finite_count += 1
            ln_l_values.append(ln_l)

    print(f"  Finite likelihoods: {finite_count}/{n_samples}")
    if ln_l_values:
        print(f"  ln_L range: {min(ln_l_values):.1f} -- {max(ln_l_values):.1f}")

    if finite_count == 0:
        raise RuntimeError(
            "All likelihood evaluations returned -inf! "
            "Check model parameters and data."
        )
    return finite_count > 0


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_posterior_predictive(
    result,
    epoch: dict,
    df_xrt_sed: pd.DataFrame,
    df_kn: pd.DataFrame,
    shock: GRBShock,
    outdir: Path,
    n_samples: int = 100,
):
    """Plot posterior predictive SED and lightcurves."""
    post = result.posterior.sample(n_samples)
    mode = epoch['mode']

    if mode == 'ssc':
        fig, ax = plt.subplots(figsize=(8, 5))
        energy  = np.logspace(2, 13, 200) * u.eV

        for _, row in post.iterrows():
            params = row.to_dict()
            model  = SSCModel(shock, absorption_method=2)
            sed    = model.sed(energy, params)
            ax.loglog(energy.value, sed.value,
                      color='steelblue', alpha=0.05, lw=0.8)

        # Data
        ax.errorbar(df_xrt_sed['energy_eV'], df_xrt_sed['sed'],
                    yerr=df_xrt_sed['sed_err'],
                    fmt='ro', ms=6, capsize=3, label='Swift XRT', zorder=5)

        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel(r"$E^2\,dN/dE$ [erg/s/cm$^2$]")
        ax.set_title(f"GRB 211211A — {epoch['description']}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(outdir / "posterior_ssc_sed.pdf")
        plt.show()

    elif mode == 'kn':
        bands  = sorted(df_kn['band'].unique())
        t_plot = np.linspace(0.1, 5.0, 200)

        fig, axes = plt.subplots(
            2, 4, figsize=(16, 8), sharex=True
        )
        axes = axes.flatten()

        for ax, band in zip(axes, bands):
            for _, row in post.iterrows():
                params = row.to_dict()
                from kai.inference.likelihood import (
                    build_model_from_params, COMPONENT_NAMES
                )
                model = build_model_from_params(
                    params, COMPONENT_NAMES[2]
                )
                mag = model.magnitude(t_plot, band=band,
                                      distance_mpc=DIST_MPC)
                ax.plot(t_plot, mag, color='steelblue',
                        alpha=0.05, lw=0.8)

            bd = df_kn[df_kn['band'] == band]
            ax.errorbar(bd['t_days'], bd['mag'], yerr=bd['mag_err'],
                        fmt='ko', ms=5, capsize=2, zorder=5)
            ax.invert_yaxis()
            ax.set_title(band)
            ax.set_xlabel("Time [days]")
            ax.set_ylabel("AB mag")

        plt.suptitle(f"GRB 211211A — {epoch['description']}")
        plt.tight_layout()
        plt.savefig(outdir / "posterior_kn_lightcurves.pdf")
        plt.show()

    elif mode == 'joint':
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: SSC SED
        energy = np.logspace(2, 13, 200) * u.eV
        for _, row in post.iterrows():
            params = {k: row[k] for k in
                      ['log10_eta_e', 'log10_ebreak', 'alpha2',
                       'log10_ecut', 'log10_B']}
            model = SSCModel(shock, absorption_method=2)
            sed   = model.sed(energy, params)
            axes[0].loglog(energy.value, sed.value,
                           color='steelblue', alpha=0.05, lw=0.8)

        axes[0].errorbar(df_xrt_sed['energy_eV'], df_xrt_sed['sed'],
                         yerr=df_xrt_sed['sed_err'],
                         fmt='ro', ms=6, capsize=3,
                         label='Swift XRT', zorder=5)
        axes[0].set_xlabel("Energy [eV]")
        axes[0].set_ylabel(r"$E^2\,dN/dE$ [erg/s/cm$^2$]")
        axes[0].set_title("SSC afterglow")
        axes[0].legend()

        # Right: KN lightcurve in r-band
        t_plot = np.linspace(0.05, 2.0, 100)
        for _, row in post.iterrows():
            params = row.to_dict()
            from kai.inference.likelihood import (
                build_model_from_params, COMPONENT_NAMES
            )
            model = build_model_from_params(params, COMPONENT_NAMES[2])
            mag   = model.magnitude(t_plot, band='r',
                                    distance_mpc=DIST_MPC)
            axes[1].plot(t_plot, mag, color='steelblue',
                         alpha=0.05, lw=0.8)

        bd = df_kn[df_kn['band'] == 'r'] if df_kn is not None else None
        if bd is not None and len(bd):
            axes[1].errorbar(bd['t_days'], bd['mag'], yerr=bd['mag_err'],
                             fmt='ko', ms=5, capsize=2, zorder=5)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Time [days]")
        axes[1].set_ylabel("r-band AB mag")
        axes[1].set_title("Kilonova")

        plt.suptitle(f"GRB 211211A — {epoch['description']}")
        plt.tight_layout()
        plt.savefig(outdir / "posterior_joint.pdf")
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Joint KN+SSC inference for GRB 211211A"
    )
    parser.add_argument(
        '--epoch', type=int, default=2, choices=[1, 2, 3],
        help="Which epoch to analyse (1=SSC, 2=Joint, 3=KN). Default: 2"
    )
    parser.add_argument(
        '--nlive', type=int, default=200,
        help="Number of dynesty live points. Default: 200"
    )
    parser.add_argument(
        '--clean', action='store_true',
        help="Ignore cached results and re-run sampler"
    )
    args = parser.parse_args()

    epoch  = EPOCHS[args.epoch]
    outdir = Path(OUTDIR_BASE) / epoch['label']
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GRB 211211A — {epoch['description']}")
    print(f"Mode    : {epoch['mode']}")
    print(f"nlive   : {args.nlive}")
    print(f"Output  : {outdir}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────
    df_xrt_sed, df_kn = load_epoch_data(epoch)

    # ── Build shock (only needed for SSC/joint) ───────────────────────────
    shock = None
    if epoch['mode'] in ('ssc', 'joint'):
        shock = build_shock(epoch)
        shock.summary()

    # ── Build likelihood ──────────────────────────────────────────────────
    print("\nBuilding likelihood...")
    if epoch['mode'] == 'ssc':
        likelihood = build_ssc_likelihood(df_xrt_sed, shock)
    elif epoch['mode'] == 'kn':
        likelihood = build_kn_likelihood(df_kn)
    elif epoch['mode'] == 'joint':
        likelihood = build_joint_likelihood(df_xrt_sed, df_kn, shock)

    # ── Build priors ──────────────────────────────────────────────────────
    priors = build_priors(epoch['mode'])
    print(f"\nPriors ({len(priors)} parameters):")
    for k, v in priors.items():
        print(f"  {k:20s}: {v}")

    # ── Pre-flight check ──────────────────────────────────────────────────
    preflight_check(likelihood, priors, n_samples=20)

    # ── Run inference ─────────────────────────────────────────────────────
    print(f"\nRunning posterior sampling (nlive={args.nlive})...")

    # SSC landscape is wide — use more walks and relaxed convergence
    extra_kwargs = {}
    if epoch['mode'] == 'ssc':
        extra_kwargs = {'sample': 'rwalk', 'walks': 50,  'dlogz': 0.5}
    elif epoch['mode'] == 'joint':
        extra_kwargs = {'sample': 'rwalk', 'walks': 100, 'dlogz': 0.1}

    result = run_inference(
        likelihood = likelihood,
        priors     = priors,
        outdir     = str(outdir),
        label      = epoch['label'],
        sampler    = 'dynesty',
        nlive      = args.nlive,
        clean      = args.clean,
        **extra_kwargs,
    )

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Posterior credible intervals (median ± 1σ):")
    for param in priors.keys():
        samples      = result.posterior[param]
        lo, med, hi = np.percentile(samples, [16, 50, 84])
        print(f"  {param:22s} = {med:.4f} + {hi-med:.4f} - {med-lo:.4f}")

    print(f"\nlog Evidence (logZ) = "
          f"{result.log_evidence:.2f} ± {result.log_evidence_err:.2f}")

    # ── Corner plot ───────────────────────────────────────────────────────
    result.plot_corner(
        parameters = list(priors.keys()),
        quantiles  = [0.16, 0.84],
        save       = True,
        filename   = str(outdir / f"{epoch['label']}_corner.pdf"),
    )

    # ── Posterior predictive plot ─────────────────────────────────────────
    plot_posterior_predictive(
        result, epoch, df_xrt_sed, df_kn, shock, outdir
    )

    print(f"\nDone. Results saved to {outdir}/")
    return result


if __name__ == "__main__":
    main()
