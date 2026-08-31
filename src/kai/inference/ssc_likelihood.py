"""
kai/inference/ssc_likelihood.py

Bilby likelihood classes for SSC afterglow and joint KN+SSC inference.

Three classes are provided:

1. SSCLikelihood
   Gaussian likelihood for broadband SED data (X-ray + gamma-ray).
   Data in E^2 dN/dE [erg/s/cm^2] vs energy [eV].

2. JointLikelihood
   Combines KilonovaLikelihood (optical/NIR photometry) and
   SSCLikelihood (X-ray/gamma-ray SED) into a single bilby Likelihood.
   The joint log-likelihood is the sum of both components.

3. SSCDataLoader
   Utility class for loading public Swift XRT and Fermi SED data.

Usage
-----
    from kai.inference.ssc_likelihood import SSCLikelihood, JointLikelihood
    from kai.inference.likelihood import KilonovaLikelihood
    from kai.models.ssc import GRBShock, SSCModel
    from kai.models.kilonova import KilonovaComponent, MultiComponentKilonova
    import pandas as pd
    import numpy as np
    import astropy.units as u

    # Build shock
    shock = GRBShock(eiso=1e52, density=0.01, tstart=1e4, tstop=2e4,
                     redshift=0.076, scenario='ISM')

    # SSC likelihood
    ssc_data = pd.DataFrame({
        'energy_eV': [1e3, 1e4, 1e9, 1e10],
        'sed':       [1e-12, 1e-12, 1e-12, 1e-13],
        'sed_err':   [1e-13, 1e-13, 1e-13, 2e-14],
        'instrument': ['XRT', 'XRT', 'HESS', 'HESS'],
    })
    ssc_lk = SSCLikelihood(data=ssc_data, shock=shock)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
import bilby
from bilby.core.likelihood import Likelihood

from kai.models.ssc import GRBShock, SSCModel
from kai.models.kilonova import KilonovaComponent, MultiComponentKilonova
from kai.inference.likelihood import (
    KilonovaLikelihood,
    COMPONENT_NAMES,
    build_model_from_params,
)


# ── SSC parameter names ───────────────────────────────────────────────────────
SSC_PARAM_NAMES = [
    'log10_eta_e',
    'log10_ebreak',
    'alpha2',
    'log10_ecut',
    'log10_B',
]


# ── SSCLikelihood ─────────────────────────────────────────────────────────────

class SSCLikelihood(Likelihood):
    """
    Gaussian likelihood for broadband GRB afterglow SED data.

    Fits the SSC model to observed E^2 dN/dE [erg/s/cm^2] values
    at given photon energies [eV], covering X-ray and gamma-ray bands.

    The log-likelihood is:

        ln L = -0.5 * sum [ (F_obs - F_model)^2 / sigma_eff^2
                            + ln(2*pi*sigma_eff^2) ]

    where sigma_eff^2 = sigma_obs^2 + sigma_floor^2.

    Parameters
    ----------
    data        : pd.DataFrame
        Columns: energy_eV [float], sed [erg/s/cm^2], sed_err [erg/s/cm^2]
        Optional: instrument [str] for plotting/filtering
    shock       : GRBShock
        Shock dynamics object (fixed — not sampled)
    sigma_floor : float
        Fractional systematic error floor. Default 0.1 (10%).
        Applied as sigma_floor * sed_obs added in quadrature.
    absorption_method : int
        Gamma-gamma absorption method for SSCModel (1 or 2). Default 2.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        shock: GRBShock,
        sigma_floor: float = 0.1,
        absorption_method: int = 2,
    ):
        self.data             = data.copy().reset_index(drop=True)
        self.shock            = shock
        self.sigma_floor      = sigma_floor
        self.absorption_method = absorption_method

        # Pre-extract arrays
        self.energy_eV  = self.data['energy_eV'].values   # [eV]
        self.sed_obs    = self.data['sed'].values          # [erg/s/cm^2]
        self.sed_err    = self.data['sed_err'].values      # [erg/s/cm^2]

        # Effective sigma: quadrature sum of statistical + fractional floor
        self.sigma_eff = np.sqrt(
            self.sed_err**2 + (self.sigma_floor * self.sed_obs)**2
        )

        # Energy array with astropy units for SSCModel
        self._energy_q = self.energy_eV * u.eV

        super().__init__(parameters={p: None for p in SSC_PARAM_NAMES})

    def _evaluate_model(self, params: dict) -> np.ndarray:
        """
        Evaluate SSC model SED at observed energies.

        Returns
        -------
        sed_model : np.ndarray  [erg/s/cm^2]
        """
        model = SSCModel(self.shock, absorption_method=self.absorption_method)
        sed   = model.sed(self._energy_q, params)
        # Extract numerical values from astropy Quantity
        return sed.to('erg / (s cm^2)').value

    def log_likelihood(self) -> float:
        """Gaussian log-likelihood on SED values."""
        # Guard against None parameters — bilby v3 initialises with None
        if any(v is None for v in self.parameters.values()):
            return -np.inf

        log10_ebreak = self.parameters.get('log10_ebreak', -99)
        log10_ecut   = self.parameters.get('log10_ecut',   -99)
        log10_B      = self.parameters.get('log10_B',      -99)
        log10_eta_e  = self.parameters.get('log10_eta_e',  -99)
        alpha2       = self.parameters.get('alpha2',          0)

        # Physical constraints — reject unphysical combinations immediately
        if log10_ecut <= log10_ebreak:    return -np.inf  # ecut > ebreak
        if log10_ecut - log10_ebreak > 6: return -np.inf  # max 6 decades
        if alpha2 <= 1.0:                 return -np.inf  # cooling break
        if log10_eta_e > 0:               return -np.inf  # eta_e <= 1

        try:
            sed_model = self._evaluate_model(self.parameters)
        except Exception:
            return -np.inf

        if not np.all(np.isfinite(sed_model)):
            return -np.inf
        if np.any(sed_model <= 0):
            return -np.inf

        residuals = self.sed_obs - sed_model
        return float(-0.5 * np.sum(
            (residuals / self.sigma_eff)**2
            + np.log(2.0 * np.pi * self.sigma_eff**2)
        ))

    def noise_log_likelihood(self) -> float:
        """Log-likelihood of noise-only model (zero signal)."""
        residuals = self.sed_obs - 0.0
        return float(-0.5 * np.sum(
            (residuals / self.sigma_eff)**2
            + np.log(2.0 * np.pi * self.sigma_eff**2)
        ))

    def summary(self) -> None:
        """Print likelihood configuration."""
        print("SSCLikelihood")
        print(f"  Free params      : {SSC_PARAM_NAMES}")
        print(f"  Data points      : {len(self.data)}")
        print(f"  Energy range     : {self.energy_eV.min():.2e} "
              f"-- {self.energy_eV.max():.2e} eV")
        print(f"  Sigma floor      : {self.sigma_floor * 100:.0f}%")
        print(f"  Absorption method: {self.absorption_method}")
        self.shock.summary()


# ── JointLikelihood ───────────────────────────────────────────────────────────

class JointLikelihood(Likelihood):
    """
    Joint Bayesian likelihood combining kilonova photometry and
    SSC afterglow SED in a single bilby-compatible framework.

    The joint log-likelihood is the sum of both components:

        ln L_joint = ln L_KN(theta_KN) + ln L_SSC(theta_SSC)

    The kilonova and SSC parameters are independent, so the joint
    posterior factorises:

        p(theta_KN, theta_SSC | data) ∝
            L_KN(theta_KN) * L_SSC(theta_SSC) * pi(theta_KN) * pi(theta_SSC)

    Parameters
    ----------
    kn_likelihood  : KilonovaLikelihood
        Likelihood for optical/NIR photometry.
    ssc_likelihood : SSCLikelihood
        Likelihood for X-ray/gamma-ray SED data.

    The combined parameter space is the union of both parameter sets.
    KN parameters are prefixed as usual (e.g. blue_mej, red_vej).
    SSC parameters: log10_eta_e, log10_ebreak, alpha2, log10_ecut, log10_B.
    """

    def __init__(
        self,
        kn_likelihood:  KilonovaLikelihood,
        ssc_likelihood: SSCLikelihood,
    ):
        self.kn_lk  = kn_likelihood
        self.ssc_lk = ssc_likelihood

        # Combined parameter dict — union of both sets
        combined_params = {}
        combined_params.update({p: None for p in kn_likelihood.parameters})
        combined_params.update({p: None for p in ssc_likelihood.parameters})

        super().__init__(parameters=combined_params)

    def _split_params(self) -> tuple[dict, dict]:
        """
        Split joint parameter dict into KN and SSC components.

        Returns
        -------
        kn_params  : dict  Kilonova parameters
        ssc_params : dict  SSC parameters
        """
        kn_params  = {k: v for k, v in self.parameters.items()
                      if k in self.kn_lk.parameters}
        ssc_params = {k: v for k, v in self.parameters.items()
                      if k in self.ssc_lk.parameters}
        return kn_params, ssc_params

    def log_likelihood(self) -> float:
        """
        Joint log-likelihood = ln L_KN + ln L_SSC.

        Returns -inf if either component returns -inf.
        """
        kn_params, ssc_params = self._split_params()

        # Evaluate KN likelihood
        for k, v in kn_params.items():
            self.kn_lk.parameters[k] = v
        ln_l_kn = self.kn_lk.log_likelihood()

        if not np.isfinite(ln_l_kn):
            return -np.inf

        # Evaluate SSC likelihood
        for k, v in ssc_params.items():
            self.ssc_lk.parameters[k] = v
        ln_l_ssc = self.ssc_lk.log_likelihood()

        if not np.isfinite(ln_l_ssc):
            return -np.inf

        return float(ln_l_kn + ln_l_ssc)

    def noise_log_likelihood(self) -> float:
        """Sum of noise log-likelihoods from both components."""
        return (self.kn_lk.noise_log_likelihood()
                + self.ssc_lk.noise_log_likelihood())

    def summary(self) -> None:
        """Print summary of both likelihood components."""
        print("=" * 60)
        print("JointLikelihood")
        print(f"  Total parameters : {len(self.parameters)}")
        print(f"  KN parameters    : {list(self.kn_lk.parameters.keys())}")
        print(f"  SSC parameters   : {SSC_PARAM_NAMES}")
        print("=" * 60)
        self.kn_lk.summary()
        print("=" * 60)
        self.ssc_lk.summary()
        print("=" * 60)


# ── SSC priors ────────────────────────────────────────────────────────────────

def ssc_priors_grb211211a() -> bilby.core.prior.PriorDict:
    """
    Physically motivated priors for GRB 211211A SSC inference.

    Tighter ranges based on GRB 190829A Night 1 best-fit as reference.
    All parameters in log10 space — priors are Uniform on log10 values.

    Parameter ranges:
        log10_eta_e  : [-2,  0]   eta_e in [0.01, 1]
        log10_ebreak : [-3,  0]   E_break in [0.001, 1] TeV
        alpha2       : [ 2,  4]   electron spectral index
        log10_ecut   : [ 0,  3]   E_cut in [1, 1000] TeV
        log10_B      : [-2,  1]   B in [0.01, 10] Gauss

    Returns
    -------
    PriorDict
    """
    from bilby.core.prior import PriorDict, Uniform

    return PriorDict({
        'log10_eta_e': Uniform(
            minimum=-2.0, maximum=0.0,
            name='log10_eta_e',
            latex_label=r'$\log_{10}(\eta_e)$',
        ),
        'log10_ebreak': Uniform(
            minimum=-3.0, maximum=0.0,
            name='log10_ebreak',
            latex_label=r'$\log_{10}(E_{\rm br}/{\rm TeV})$',
        ),
        'alpha2': Uniform(
            minimum=2.0, maximum=4.0,
            name='alpha2',
            latex_label=r'$\alpha_2$',
        ),
        'log10_ecut': Uniform(
            minimum=0.0, maximum=3.0,
            name='log10_ecut',
            latex_label=r'$\log_{10}(E_{\rm cut}/{\rm TeV})$',
        ),
        'log10_B': Uniform(
            minimum=-2.0, maximum=1.0,
            name='log10_B',
            latex_label=r'$\log_{10}(B/{\rm G})$',
        ),
    })


def joint_priors_grb211211a(n_kn_components: int = 2) -> bilby.core.prior.PriorDict:
    """
    Combined kilonova + SSC priors for GRB 211211A joint inference.

    Parameters
    ----------
    n_kn_components : int  Number of kilonova ejecta components (1, 2, or 3)

    Returns
    -------
    PriorDict  Union of KN and SSC priors
    """
    from kai.inference.priors import gw170817_priors

    kn_priors  = gw170817_priors(n_components=n_kn_components)
    ssc_priors = ssc_priors_grb211211a()

    combined = bilby.core.prior.PriorDict()
    combined.update(kn_priors)
    combined.update(ssc_priors)
    return combined


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_xrt_sed(
    filepath: str,
    t_min: float = None,
    t_max: float = None,
) -> pd.DataFrame:
    """
    Load Swift XRT spectral data for SSC inference.

    Expected file format (from UK Swift Science Data Centre):
        # energy_keV  flux  flux_err_hi  flux_err_lo
        0.3  1.2e-12  2e-13  1.5e-13
        ...

    Parameters
    ----------
    filepath : str   Path to XRT spectral data file
    t_min    : float Optional minimum time filter [s]
    t_max    : float Optional maximum time filter [s]

    Returns
    -------
    df : pd.DataFrame
        Columns: energy_eV, sed, sed_err, instrument
    """
    df = pd.read_csv(
        filepath,
        comment='#',
        sep=r'\s+',
        names=['energy_keV', 'sed', 'sed_err_hi', 'sed_err_lo'],
    )
    # Convert keV → eV
    df['energy_eV'] = df['energy_keV'] * 1e3
    # Use symmetric error (average of hi/lo)
    df['sed_err'] = (df['sed_err_hi'] + df['sed_err_lo']) / 2.0
    df['instrument'] = 'XRT'

    return df[['energy_eV', 'sed', 'sed_err', 'instrument']].copy()


def load_fermi_sed(filepath: str) -> pd.DataFrame:
    """
    Load Fermi-LAT or GBM spectral data for SSC inference.

    Expected file format:
        # energy_MeV  e2dnde  e2dnde_err
        100  1e-12  1e-13
        ...

    Parameters
    ----------
    filepath : str  Path to Fermi spectral data file

    Returns
    -------
    df : pd.DataFrame
        Columns: energy_eV, sed, sed_err, instrument
    """
    df = pd.read_csv(
        filepath,
        comment='#',
        sep=r'\s+',
        names=['energy_MeV', 'sed', 'sed_err'],
    )
    df['energy_eV']  = df['energy_MeV'] * 1e6   # MeV → eV
    df['instrument'] = 'Fermi'
    return df[['energy_eV', 'sed', 'sed_err', 'instrument']].copy()


def mock_grb211211a_xrt_sed() -> pd.DataFrame:
    """
    Mock Swift XRT SED for GRB 211211A (Night 1, ~0.3 days).

    Based on flux levels reported in Rastinejad et al. 2022.
    Use this for testing before real data is downloaded.

    Returns
    -------
    pd.DataFrame with columns: energy_eV, sed, sed_err, instrument
    """
    # XRT energy range: 0.3 -- 10 keV
    # Typical afterglow flux at ~0.3 days: ~5e-12 erg/s/cm^2
    # Photon index ~ 1.8 → E^2 dN/dE ~ E^(2-gamma) ~ E^0.2 (nearly flat)
    energies_keV = np.array([0.5, 1.0, 2.0, 5.0, 8.0])
    flux_ref     = 5e-12   # erg/s/cm^2 at 1 keV

    sed     = flux_ref * (energies_keV / 1.0) ** 0.2
    sed_err = sed * 0.15   # 15% uncertainty

    return pd.DataFrame({
        'energy_eV':  energies_keV * 1e3,
        'sed':        sed,
        'sed_err':    sed_err,
        'instrument': 'XRT',
    })


def combine_sed_datasets(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple SED DataFrames (XRT, Fermi, etc.) into one,
    sorted by energy.

    Parameters
    ----------
    *dfs : pd.DataFrame  Any number of SED DataFrames

    Returns
    -------
    combined : pd.DataFrame  Sorted by energy_eV
    """
    combined = pd.concat(list(dfs), ignore_index=True)
    combined = combined.sort_values('energy_eV').reset_index(drop=True)
    return combined
