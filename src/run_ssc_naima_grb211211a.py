"""
run_ssc_naima_grb211211a.py

SSC afterglow inference for GRB 211211A using naima/emcee.
Applies to Epoch 1 (t ~ 0.043 days, XRT + UVOT simultaneous data).

This script uses the naima framework (which wraps emcee) for the SSC
inference — the same approach validated on GRB 190829A in Jean's PhD work.
The resulting posterior samples are saved and used in Stage 2 for
afterglow subtraction before kilonova inference.

Usage
-----
    python src/run_ssc_naima_grb211211a.py           # full run
    python src/run_ssc_naima_grb211211a.py --quick   # quick test (fewer steps)
    python src/run_ssc_naima_grb211211a.py --reload  # reload saved chain

Output
------
    results/grb211211a/epoch1_ssc/
        grb211211a_epoch1_chain.h5      — emcee chain (naima format)
        grb211211a_epoch1_results.fits  — posterior summary table
        grb211211a_epoch1_corner.pdf    — corner plot
        grb211211a_epoch1_sed.pdf       — SED fit plot
        grb211211a_epoch1_posteriors.npy — posterior samples (for Stage 2)

GRB 211211A Epoch 1
-------------------
    t_obs  : 0.043 days = 3519 -- 4972 s after trigger
    XRT    : 0.3-10 keV (28 PC-mode points, averaged to 5 SED points)
    UVOT   : B, U, V, uvw1, uvm2, uvw2 (6 bands, t ~ 0.043 days)
    z      : 0.0763
    Eiso   : 1.25e52 erg
    n      : 1e-3 cm^-3 (short GRB halo environment)
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as con
from astropy.table import Table, vstack
from astropy.cosmology import WMAP9 as cosmo

import naima
from naima.models import (
    Synchrotron, InverseCompton, ExponentialCutoffBrokenPowerLaw
)
from naima import uniform_prior, normal_prior
import naima.utils

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from kai.data.grb211211a_loaders import (
    load_xrt_lightcurve, load_xrt_as_sed,
    load_optical_photometry, GRB211211A,
)

# ── Output directory ──────────────────────────────────────────────────────────
OUTDIR = Path("results/grb211211a/epoch1_ssc")
OUTDIR.mkdir(parents=True, exist_ok=True)
BASENAME = str(OUTDIR / "grb211211a_epoch1")

# ── Physical constants ────────────────────────────────────────────────────────
M_E       = con.m_e.cgs.value
C_LIGHT   = con.c.cgs.value
MEC2_EV   = (con.m_e * con.c**2).to('eV').value
H_PLANCK  = con.h.cgs.value
EL        = con.e.gauss.value
ERG_TO_EV = 624150912588.3258
SIGMA_T   = con.sigma_T.cgs.value
MPC2      = (con.m_p * con.c**2).to('eV')
MPC2_ERG  = MPC2.to('erg').value

# ── GRB 211211A Epoch 1 parameters ───────────────────────────────────────────
REDSHIFT  = GRB211211A['redshift']       # 0.0763
EISO      = GRB211211A['eiso']           # 1.25e52 erg
DENSITY   = 1e-3                         # cm^-3 (halo environment)
TSTART    = 3519.0                       # s after trigger
TSTOP     = 4972.0                       # s after trigger
AVTIME    = (TSTART + TSTOP) / 2.0      # s
DL        = cosmo.luminosity_distance(REDSHIFT)

# ── Blandford-McKee shock dynamics ───────────────────────────────────────────
GAMMA = (
    (1.0/8.0)**(3.0/8.0)
    * (3.0 * EISO / (4.0 * np.pi * DENSITY * MPC2_ERG
                      * (C_LIGHT * AVTIME)**3))**0.125
)
SIZER    = 8.0 * C_LIGHT * AVTIME * GAMMA**2
DEPTHPAR = 9.0
SHOCK_E  = 2.0 * GAMMA**2 * DENSITY * MPC2_ERG   # erg/cm^3

print(f"GRB 211211A Epoch 1 — Shock dynamics:")
print(f"  t_obs   = {AVTIME:.1f} s")
print(f"  Gamma   = {GAMMA:.2f}")
print(f"  R       = {SIZER:.3e} cm")
print(f"  E_shock = {SHOCK_E:.3e} erg/cm^3")
print(f"  D_L     = {DL:.3f}")


# ── Helper functions (from grbloader.py, validated on GRB 190829A) ────────────

def sigma_gammagamma(Eph1, Eph2):
    CMene = Eph1 * Eph2 / (MEC2_EV**2)
    mask  = CMene > 1.0
    res   = np.full(CMene.shape, 0.0)
    res[mask] = (
        3.0 / (2.0 * CMene[mask]**2) * SIGMA_T
        * (
            (CMene[mask] + 0.5 * np.log(CMene[mask])
             - 1.0/6.0 + 1.0 / (2.0 * CMene[mask]))
            * np.log(np.sqrt(CMene[mask]) + np.sqrt(CMene[mask] - 1.0))
            - (CMene[mask] + 4.0/9.0 - 1.0 / (9.0 * CMene[mask]))
            * np.sqrt(1.0 - 1.0 / CMene[mask])
        )
    )
    return res * u.cm**2


def absorption_coeff(egamma, targetene, target):
    product  = sigma_gammagamma(
        np.vstack(egamma.to('eV').value),
        targetene.to('eV').value
    ) * target
    return naima.utils.trapz_loglog(product, targetene, axis=1)


def tau_val(Egamma, targetene, target, size):
    coeff = absorption_coeff(Egamma, targetene, target)
    return (size.to('cm') * coeff).decompose().value


def cutoff_limit(bfield):
    eff    = 1.0
    cutoff = (
        (3.0/2.0)**(3.0/4.0)
        * np.sqrt(1.0 / (EL**3 * bfield))
        * (M_E**2 * C_LIGHT**4) * eff**(-0.5)
    ) * u.erg
    return np.log10(cutoff.value * ERG_TO_EV * 1e-12)


def synch_cooltime(bfield, partene):
    bf    = bfield.to('G').value
    epar  = partene.to('erg').value
    tcool = (6.0 * np.pi * M_E**4 * C_LIGHT**3) / (
        SIGMA_T * M_E**2 * epar * bf**2
    )
    return tcool * u.s


def synch_charene(bfield, partene):
    bf    = bfield.to('G').value
    epar  = partene.to('erg').value
    charene = (
        np.sqrt(3.0/2.0) * (H_PLANCK * EL * bf)
        / (2.0 * np.pi * M_E**3 * C_LIGHT**5) * epar**2
    )
    return charene * ERG_TO_EV * u.eV


# ── Naima model function ──────────────────────────────────────────────────────

# Module-level storage for model components (naima pattern)
_synch_comp   = None
_ic_comp      = None
_synch_compGG = None
_ic_compGG    = None
_Emin         = None


def grb211211a_ssc_model(pars, data):
    """
    Naima-compatible SSC model for GRB 211211A Epoch 1.

    Parameters (pars)
    -----------------
    pars[0] : log10(eta_e)        electron energy fraction
    pars[1] : log10(E_break/TeV)  break energy
    pars[2] : alpha2              high-energy spectral index
    pars[3] : log10(E_cut/TeV)   cutoff energy
    pars[4] : log10(B/G)         magnetic field

    Returns
    -------
    [total_sed, synch, ic]  each as astropy Quantity [erg/s/cm^2]
    """
    global _synch_comp, _ic_comp, _synch_compGG, _ic_compGG, _Emin

    # Unpack parameters
    eta_e  = 10.0 ** pars[0]
    ebreak = 10.0 ** pars[1] * u.TeV
    alpha2 = pars[2]
    alpha1 = alpha2 - 1.0          # cooling break
    ecut   = 10.0 ** pars[3] * u.TeV
    bfield = 10.0 ** pars[4] * u.G

    # Shock-derived quantities
    redf     = 1.0 + REDSHIFT
    doppler  = GAMMA
    size_reg = SIZER * u.cm
    vol      = 4.0 * np.pi * SIZER**2 * (SIZER / (DEPTHPAR * GAMMA)) * u.cm**3
    shock_ed = SHOCK_E * u.erg / u.cm**3
    eemax    = ecut.to('eV').value * 1e13   # max electron energy [eV]

    # Step 1: temporary ECBPL to get Emin
    ampl_tmp = 1.0 / u.eV
    ECBPL = ExponentialCutoffBrokenPowerLaw(
        ampl_tmp, 1.0 * u.TeV, ebreak, alpha1, alpha2, ecut
    )

    ener_grid = np.logspace(9, np.log10(eemax), 100) * u.eV
    eldis     = ECBPL(ener_grid)
    mean_e    = (
        naima.utils.trapz_loglog(ener_grid * eldis, ener_grid)
        / naima.utils.trapz_loglog(eldis, ener_grid)
    )
    Emin = (eta_e * GAMMA * MPC2) / mean_e * 1e9 * u.eV
    _Emin = Emin

    # Step 2: physical amplitude from energy conservation
    SYN_tmp = Synchrotron(
        ECBPL, B=bfield, Eemin=Emin, Eemax=eemax * u.eV, nEed=20
    )
    we_tmp = SYN_tmp.compute_We(Eemin=Emin, Eemax=eemax * u.eV)
    ampl   = ((eta_e * shock_ed * vol) / we_tmp) / u.eV

    ECBPL = ExponentialCutoffBrokenPowerLaw(
        ampl, 1.0 * u.TeV, ebreak, alpha1, alpha2, ecut
    )
    SYN = Synchrotron(
        ECBPL, B=bfield, Eemin=Emin, Eemax=eemax * u.eV, nEed=20
    )

    # Step 3: synchrotron seed photon field
    cutoff_char  = np.log10(synch_charene(bfield, ecut).value)
    bins_per_dec = 20
    bins = max(10, int((cutoff_char - (-4.0)) * bins_per_dec))
    Esy  = np.logspace(-4.0, cutoff_char + 1, bins) * u.eV
    Lsy  = SYN.flux(Esy, distance=0 * u.cm)
    phn_sy = Lsy / (4.0 * np.pi * size_reg**2 * C_LIGHT * u.cm / u.s)

    # Step 4: IC/SSC component
    IC = InverseCompton(
        ECBPL,
        seed_photon_fields=[['SSC', Esy, phn_sy]],
        Eemin=Emin, Eemax=eemax * u.eV, nEed=20,
    )

    # Step 5: observed SED with Doppler + redshift
    obs_energy = data['energy'] / doppler * redf
    synch      = doppler**2 * SYN.sed(obs_energy, distance=DL)
    ic         = doppler**2 * IC.sed(obs_energy,  distance=DL)

    _synch_comp = synch
    _ic_comp    = ic

    # Step 6: gamma-gamma absorption (Rybicki & Lightman 1979, Method 2)
    tau = tau_val(
        obs_energy, Esy, phn_sy,
        size_reg / (DEPTHPAR * GAMMA),
    )

    synch_gg = synch.copy()
    ic_gg    = ic.copy()
    mask = tau > 1e-4
    synch_gg[mask] = synch[mask] / tau[mask] * (1.0 - np.exp(-tau[mask]))
    ic_gg[mask]    = ic[mask]    / tau[mask] * (1.0 - np.exp(-tau[mask]))

    _synch_compGG = synch_gg
    _ic_compGG    = ic_gg

    total = synch_gg + ic_gg
    return total, synch_gg, ic_gg


# ── Prior function ────────────────────────────────────────────────────────────

def lnprior(pars):
    """
    Log-prior for GRB 211211A SSC inference.

    Tighter ranges than GRB 190829A (short GRB: lower energy, lower density).
    Includes cooling constraint: synchrotron cooling time at break ~ comoving age.

    pars[0]: log10(eta_e)   [-2, 0]
    pars[1]: log10(E_break) [-3, 0] TeV
    pars[2]: alpha2         [ 2, 4]
    pars[3]: log10(E_cut)   [ 0, 3] TeV  (must exceed break)
    pars[4]: log10(B)       [-2, 1] G
    """
    # Hard bounds
    p0 = uniform_prior(pars[0], -2.0, 0.0)
    p1 = uniform_prior(pars[1], -3.0, 0.0)
    p2 = uniform_prior(pars[2],  2.0, 4.0)
    p3 = uniform_prior(pars[3],  0.0, 3.0)
    p4 = uniform_prior(pars[4], -2.0, 1.0)

    # ecut must exceed ebreak
    if pars[3] <= pars[1]:
        return -np.inf

    # Synchrotron cutoff burn-off limit
    ecut_max = cutoff_limit(10.0**pars[4])
    if pars[3] > ecut_max:
        return -np.inf

    # Cooling constraint: t_cool(E_break) ~ t_obs * Gamma (comoving)
    t_cool = synch_cooltime(
        10.0**pars[4] * u.G,
        10.0**pars[1] * u.TeV
    ).value
    t_comov = AVTIME * GAMMA
    p_cooling = normal_prior(t_cool, t_comov, t_comov * 0.5)

    return p0 + p1 + p2 + p3 + p4 + p_cooling


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_dataset() -> list:
    """
    Build the naima dataset for Epoch 1 (XRT SED + UVOT + Fermi-LAT).

    Returns a list of naima-compatible astropy Tables.
    """
    from kai.data.grb211211a_loaders import load_fermi_lat_grb211211a

    # ── XRT SED ───────────────────────────────────────────────────────────────
    df_xrt = load_xrt_lightcurve("data/GRB211211A/xrt_flux_lightcurve.qdp")
    df_sed  = load_xrt_as_sed(df_xrt, t_center=AVTIME, dt=800.0)

    xrt_table = Table({
        'energy':       df_sed['energy_eV'].values * u.eV,
        'flux':         df_sed['sed'].values * u.erg / u.cm**2 / u.s,
        'flux_error':   df_sed['sed_err'].values * u.erg / u.cm**2 / u.s,
    })
    xrt_table.meta['name'] = 'Swift XRT (0.3-10 keV)'

    # ── Fermi-LAT (Zhang+2022, quasi-simultaneous 395-30780 s) ───────────────
    df_lat = load_fermi_lat_grb211211a()
    lat_table = Table({
        'energy':     df_lat['energy_eV'].values * u.eV,
        'flux':       df_lat['sed'].values * u.erg / u.cm**2 / u.s,
        'flux_error': df_lat['sed_err'].values * u.erg / u.cm**2 / u.s,
    })
    lat_table.meta['name'] = 'Fermi-LAT (100 MeV-10 GeV, Zhang+2022)'

    # ── UVOT optical (convert mag to flux) ────────────────────────────────────
    # UVOT filter effective wavelengths [Angstrom] and zero points [Jy]
    uvot_filters = {
        'uvw2': {'lam_aa': 1928.0, 'zp_jy': 3631.0},
        'uvm2': {'lam_aa': 2246.0, 'zp_jy': 3631.0},
        'uvw1': {'lam_aa': 2600.0, 'zp_jy': 3631.0},
        'U':    {'lam_aa': 3465.0, 'zp_jy': 3631.0},
        'B':    {'lam_aa': 4392.0, 'zp_jy': 3631.0},
        'V':    {'lam_aa': 5468.0, 'zp_jy': 3631.0},
    }

    df_opt = load_optical_photometry(
        bands  = list(uvot_filters.keys()),
        t_min  = 0.030,
        t_max  = 0.060,
    )

    energies_eV = []
    fluxes      = []
    flux_errs   = []

    for _, row in df_opt.iterrows():
        band = row['band']
        if band not in uvot_filters:
            continue
        lam_aa = uvot_filters[band]['lam_aa']
        zp_jy  = uvot_filters[band]['zp_jy']

        # Convert AB mag to F_nu [Jy] then to E^2 dN/dE [erg/s/cm^2]
        f_nu_jy    = zp_jy * 10.0**(-row['mag'] / 2.5)   # Jy
        f_nu_cgs   = f_nu_jy * 1e-23                      # erg/s/cm^2/Hz
        nu         = 3e18 / lam_aa                        # Hz (c/lambda in Angstrom)
        e2dnde     = f_nu_cgs * nu                        # erg/s/cm^2
        e2dnde_err = e2dnde * row['mag_err'] * np.log(10) / 2.5

        e_eV = 4.136e-15 * nu   # h*nu in eV (h = 4.136e-15 eV*s)

        energies_eV.append(e_eV)
        fluxes.append(e2dnde)
        flux_errs.append(e2dnde_err)

    if energies_eV:
        uvot_table = Table({
            'energy':     np.array(energies_eV) * u.eV,
            'flux':       np.array(fluxes) * u.erg / u.cm**2 / u.s,
            'flux_error': np.array(flux_errs) * u.erg / u.cm**2 / u.s,
        })
        uvot_table.meta['name'] = 'Swift UVOT'
        dataset = [xrt_table, uvot_table, lat_table]
        print(f"Dataset: {len(xrt_table)} XRT + {len(uvot_table)} UVOT + "
              f"{len(lat_table)} LAT points")
    else:
        dataset = [xrt_table, lat_table]
        print(f"Dataset: {len(xrt_table)} XRT + {len(lat_table)} LAT points")

    print(f"Energy range: {xrt_table['energy'].min():.2e} -- "
          f"{lat_table['energy'].max():.2e}")
    return dataset


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SSC inference for GRB 211211A Epoch 1 using naima/emcee"
    )
    parser.add_argument('--quick',  action='store_true',
                        help="Quick test: fewer walkers and steps")
    parser.add_argument('--reload', action='store_true',
                        help="Reload existing chain instead of re-running")
    args = parser.parse_args()

    # ── Sampler settings ──────────────────────────────────────────────────────
    if args.quick:
        NWALKERS = 16
        NBURN    = 50
        NRUN     = 100
        THREADS  = 1
        print("Quick mode: 16 walkers, 50 burn-in, 100 steps")
    else:
        NWALKERS = 32
        NBURN    = 200
        NRUN     = 1000
        THREADS  = 4
        print(f"Full run: {NWALKERS} walkers, {NBURN} burn-in, {NRUN} steps")

    # ── Starting parameters (GRB 190829A Night 1 as reference) ───────────────
    # These are reasonable starting points for the MCMC
    P0 = [
        -0.04,    # log10(eta_e)
        -1.5,     # log10(E_break/TeV)
        3.1,      # alpha2
        1.7,      # log10(E_cut/TeV)
        -0.45,    # log10(B/G)
    ]
    LABELS = [
        r'$\log_{10}(\eta_e)$',
        r'$\log_{10}(E_{\rm br}/{\rm TeV})$',
        r'$\alpha_2$',
        r'$\log_{10}(E_{\rm cut}/{\rm TeV})$',
        r'$\log_{10}(B/{\rm G})$',
    ]

    # ── Prepare dataset ───────────────────────────────────────────────────────
    print("\nPreparing dataset...")
    dataset = prepare_dataset()

    # ── Quick model check ─────────────────────────────────────────────────────
    print("\nModel check at starting parameters...")
    test_energy = Table([np.logspace(2, 12, 50) * u.eV], names=['energy'])
    try:
        result = grb211211a_ssc_model(P0, test_energy)
        print(f"  SED range: {result[0].min():.3e} -- {result[0].max():.3e}")
        print(f"  Prior at P0: {lnprior(P0):.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # ── Run or reload ─────────────────────────────────────────────────────────
    chain_file = BASENAME + "_chain.h5"

    if args.reload and Path(f"{chain_file}").exists():
        print(f"\nReloading chain from {chain_file}...")
        sampler = naima.read_run(chain_file + "_chain.h5", grb211211a_ssc_model)
    else:
        print(f"\nRunning naima/emcee sampler...")
        print(f"  Output: {BASENAME}_chain.h5")
        sampler, pos = naima.run_sampler(
            data_table = dataset,
            p0         = P0,
            labels     = LABELS,
            model      = grb211211a_ssc_model,
            prior      = lnprior,
            prefit     = True,
            guess      = False,
            nwalkers   = NWALKERS,
            nburn      = NBURN,
            nrun       = NRUN,
            threads    = THREADS,
        )
        naima.save_run(filename=BASENAME + "_chain.h5", sampler=sampler, clobber=True)
        naima.save_results_table(outname=BASENAME + "_results", sampler=sampler, overwrite=True)
        print("Sampling complete.")

    # ── Results ───────────────────────────────────────────────────────────────
    print("\nPosterior credible intervals (median ± 1σ):")
    param_names = ['log10_eta_e', 'log10_ebreak', 'alpha2',
                   'log10_ecut', 'log10_B']
    for i, (name, label) in enumerate(zip(param_names, LABELS)):
        chain = sampler.flatchain[:, i]
        lo, med, hi = np.percentile(chain, [16, 50, 84])
        print(f"  {name:18s} = {med:.4f} + {hi-med:.4f} - {med-lo:.4f}")

    # ── Save posterior samples for Stage 2 ───────────────────────────────────
    posteriors = {name: sampler.flatchain[:, i]
                  for i, name in enumerate(param_names)}
    np.save(f"{BASENAME}_posteriors.npy", posteriors)
    print(f"\nPosterior samples saved to {BASENAME}_posteriors.npy")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # Corner plot
    try:
        import corner
        corner.corner(sampler.flatchain, labels=LABELS,
                      quantiles=[0.16, 0.5, 0.84], show_titles=True)
        plt.savefig(BASENAME + "_corner.pdf", bbox_inches='tight')
        plt.close()
        print("Corner plot saved.")
    except Exception as e:
        print(f"Corner plot skipped: {e}")
    if False:  # disabled naima corner
        pass
    plt.savefig(f"{BASENAME}_corner.pdf", bbox_inches='tight')
    plt.close()

    # SED plot
    fig, ax = plt.subplots(figsize=(9, 6))
    # Plot data manually (naima.plot_data has matplotlib compat issue)
    colors = ['steelblue', 'orange', 'green']
    for ds, col in zip(dataset, colors):
        ax.errorbar(
            ds['energy'].to('eV').value,
            ds['flux'].to('erg/(s cm^2)').value,
            yerr=np.abs(ds['flux_error'].to('erg/(s cm^2)').value),
            fmt='o', color=col, capsize=3, ms=6, zorder=5,
            label=ds.meta.get('name', 'data'),
        )

    # Posterior predictive SED
    e_plot = Table([np.logspace(2, 13, 200) * u.eV], names=['energy'])
    a, b   = naima.plot._calc_CI(
        sampler, confs=[1, 2], modelidx=0,
        e_range=[1e2 * u.eV, 1e13 * u.eV]
    )
    xval   = a.value
    ax.fill_between(xval, b[1][0].value, b[1][1].value,
                    alpha=0.15, color='C0', label=r'$2\sigma$')
    ax.fill_between(xval, b[0][0].value, b[0][1].value,
                    alpha=0.35, color='C0', label=r'$1\sigma$')

    # Best-fit components
    pars_med = [np.median(sampler.flatchain[:, i]) for i in range(5)]
    grb211211a_ssc_model(pars_med, e_plot)
    ax.loglog(e_plot['energy'].value, _synch_compGG.value,
              'b--', lw=1.5, label='Synchrotron')
    ax.loglog(e_plot['energy'].value, _ic_compGG.value,
              'r-.', lw=1.5, label='IC/SSC')

    ax.set_xlabel("Energy [eV]", fontsize=13)
    ax.set_ylabel(r"$E^2\,dN/dE$ [erg s$^{-1}$ cm$^{-2}$]", fontsize=13)
    ax.set_title("GRB 211211A — Epoch 1 SSC fit (t ~ 0.043 days)", fontsize=12)
    ax.set_xlim(1e0, 1e13)   # include UVOT (eV) through LAT (GeV)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{BASENAME}_sed.pdf", bbox_inches='tight')
    plt.close()

    print(f"\nAll results saved to {OUTDIR}/")
    print("Next step: run Stage 2 KN inference using these posteriors.")

    return sampler


if __name__ == "__main__":
    main()
