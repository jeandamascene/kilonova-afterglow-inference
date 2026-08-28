"""
kai/models/kilonova.py

Multi-component kilonova lightcurve model based on Villar et al. 2017
(https://doi.org/10.3847/2041-8213/aa9c84).

Physics
-------
Each ejecta component is characterised by:
    mej   : ejecta mass [solar masses]
    vej   : ejecta velocity [fraction of c]
    kappa : grey opacity [cm^2/g]

The bolometric luminosity uses the Arnett-like integral (Villar+2017 Eq.1),
with heating from Korobkin+2012 and thermalization from Barnes+2016.
Per-band AB magnitudes assume blackbody emission at the photospheric temperature.
"""

import warnings
from dataclasses import dataclass

import numpy as np

# ── Physical constants (CGS) ──────────────────────────────────────────────────
C_LIGHT  = 2.998e10    # cm/s
M_SUN    = 1.989e33    # g
PC_TO_CM = 3.086e18    # cm
DAY_TO_S = 86400.0     # s
SIGMA_SB = 5.6704e-5   # erg/cm2/s/K4
H_PLANCK = 6.626e-27   # erg s
K_BOLTZ  = 1.381e-16   # erg/K

# ── Heating rate (Korobkin+2012 power law) ────────────────────────────────────
EPSILON_0 = 1.58e10    # erg/s/g at t=1 day EPSILON_0 = 1.58e10    # erg/s/g at t=1 day

ALPHA_T   = 1.3

# ── Filter effective wavelengths [Angstrom] ───────────────────────────────────
FILTER_WAVELENGTHS = {
    "u": 3560.0, "g": 4770.0, "r": 6230.0,
    "i": 7630.0, "z": 9130.0, "y": 10200.0,
    "J": 12200.0, "H": 16300.0, "K": 21900.0,
    "uvw2": 1928.0, "uvm2": 2246.0, "uvw1": 2600.0,
    "U": 3465.0, "B": 4392.0, "V": 5468.0,
}


def thermalization_efficiency(t_days: np.ndarray) -> np.ndarray:
    """Barnes et al. 2016 thermalization efficiency f_th(t) in [0,1]."""
    t = np.maximum(np.atleast_1d(t_days).astype(float), 1e-10)
    a, b, d = 0.56, 0.17, 0.74
    f_th = 0.36 * (
        np.exp(-a * t)
        + np.log1p(2.0 * b * t ** d) / (2.0 * b * t ** d)
    )
    return np.clip(f_th, 0.0, 1.0)


def heating_rate(t_days: np.ndarray) -> np.ndarray:
    """
    Specific r-process heating rate [erg/s/g] — Korobkin et al. 2012.
    t_days : time in days.
    """
    t = np.maximum(np.atleast_1d(t_days).astype(float), 1e-10)
    t_eff = np.maximum(t, 0.3)   # floor at 0.3 days — suppresses t^-1.3 divergence
    return EPSILON_0 * t_eff ** (-ALPHA_T) * thermalization_efficiency(t)


@dataclass
class KilonovaComponent:
    """
    Single kilonova ejecta component.

    Parameters
    ----------
    mej   : float  Ejecta mass [Msun]
    vej   : float  Ejecta velocity [fraction of c]
    kappa : float  Grey opacity [cm^2/g]
    name  : str    Component label
    """

    mej:   float
    vej:   float
    kappa: float
    name:  str = "component"

    def __post_init__(self):
        if not 0 < self.vej < 1:
            raise ValueError(f"vej must be in (0,1), got {self.vej}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if not 0 < self.mej < 10:
            warnings.warn(f"mej={self.mej} Msun seems unphysical.")

    @property
    def mej_cgs(self) -> float:
        return self.mej * M_SUN

    @property
    def vej_cgs(self) -> float:
        return self.vej * C_LIGHT

    def t_diff(self) -> float:
        """
        Diffusion timescale [days].
        tau_d = sqrt(2 * kappa * mej / (beta * c * vej)), beta=13.8
        """
        beta = 13.8
        return np.sqrt(
            2.0 * self.kappa * self.mej_cgs
            / (beta * C_LIGHT * self.vej_cgs)
        ) / DAY_TO_S

    def t_transition(self) -> float:
        """
        Thick-to-thin transition time [days].
        t_tr = sqrt(kappa * mej / (4*pi * vej^2 * c))
        """
        return np.sqrt(
            self.kappa * self.mej_cgs
            / (4.0 * np.pi * self.vej_cgs ** 2 * C_LIGHT)
        ) / DAY_TO_S

    def bolometric_luminosity(self, t_days: np.ndarray) -> np.ndarray:
        """
        Bolometric luminosity [erg/s] — Villar+2017 Eq. 1, in days throughout.

        L(t) = (2/tau_d) * int_0^t Q(t') exp((t'^2-t^2)/tau_d^2) dt'

        where tau_d [days], Q = eps(t)*mej [erg/s], t' and t in days.
        The integral over days gives [erg/s * days]; multiplying by
        (2/tau_d [days]) gives [erg/s].
        """
        t     = np.atleast_1d(t_days).astype(float)
        tau_d = self.t_diff()   # days
        L_bol = np.zeros(len(t))

        for i, ti in enumerate(t):
            if ti <= 0:
                continue

            # Everything in days
            #t_int = np.linspace(1e-4, ti, 1000)   # days
            t_int = np.linspace(0.1, ti, 1000)    # days — start at first obs epoch

            # Total heating power [erg/s]
            Q = heating_rate(t_int) * self.mej_cgs

            # Arnett kernel — argument always <= 0
            exp_arg = (t_int ** 2 - ti ** 2) / (tau_d ** 2)
            exp_arg = np.clip(exp_arg, -500.0, 0.0)
            kernel  = np.exp(exp_arg)

            # Integral in days: [erg/s * days]
            integral = np.trapezoid(Q * kernel, t_int)

            # (2/tau_d [days]) * [erg/s * days] = [erg/s]
            L_bol[i] = (2.0 / tau_d) * integral

        return np.maximum(L_bol, 0.0)

    def photospheric_radius(self, t_days: np.ndarray) -> np.ndarray:
        """Photospheric radius [cm] — homologous expansion R = vej*c*t."""
        t_s = np.atleast_1d(t_days).astype(float) * DAY_TO_S
        return self.vej_cgs * t_s

    def temperature(self, t_days: np.ndarray) -> np.ndarray:
        """
        Effective blackbody temperature [K], floored at 1000 K.
        T = (L / (4*pi*sigma_SB*R^2))^(1/4)
        """
        t = np.atleast_1d(t_days).astype(float)
        L = self.bolometric_luminosity(t)
        R = np.maximum(self.photospheric_radius(t), 1.0)
        T = (L / (4.0 * np.pi * SIGMA_SB * R ** 2)) ** 0.25
        return np.maximum(T, 1000.0)

    def blackbody_flux(
        self,
        t_days: np.ndarray,
        wavelength_aa: float,
        distance_mpc: float,
    ) -> np.ndarray:
        """
        Monochromatic flux density [erg/s/cm2/Hz] — blackbody at T(t).

        Parameters
        ----------
        t_days        : array-like  Time [days]
        wavelength_aa : float       Filter wavelength [Angstrom]
        distance_mpc  : float       Luminosity distance [Mpc]
        """
        t    = np.atleast_1d(t_days).astype(float)
        T    = self.temperature(t)
        R    = self.photospheric_radius(t)
        d_cm = distance_mpc * 1e6 * PC_TO_CM
        lam  = wavelength_aa * 1e-8          # Angstrom -> cm
        nu   = C_LIGHT / lam                 # Hz

        x    = np.clip(H_PLANCK * nu / (K_BOLTZ * np.maximum(T, 1.0)),
                       1e-10, 500.0)
        B_nu = (2.0 * H_PLANCK * nu ** 3 / C_LIGHT ** 2) / np.expm1(x)
        F_nu = np.pi * B_nu * (R / d_cm) ** 2
        return np.maximum(F_nu, 1e-300)

    def magnitude(
        self,
        t_days: np.ndarray,
        band: str,
        distance_mpc: float,
    ) -> np.ndarray:
        """
        AB magnitude in a given filter band.

        Parameters
        ----------
        t_days       : array-like  Time [days]
        band         : str         Filter name
        distance_mpc : float       Luminosity distance [Mpc]
        """
        if band not in FILTER_WAVELENGTHS:
            raise ValueError(
                f"Band '{band}' not recognised. "
                f"Available: {list(FILTER_WAVELENGTHS.keys())}"
            )
        F_nu = self.blackbody_flux(t_days, FILTER_WAVELENGTHS[band],
                                   distance_mpc)
        return -2.5 * np.log10(F_nu) - 48.6


class MultiComponentKilonova:
    """
    Superposition of multiple KilonovaComponent instances.

    Parameters
    ----------
    components : list of KilonovaComponent

    Example
    -------
    >>> blue  = KilonovaComponent(mej=0.025, vej=0.27, kappa=0.5,  name="blue")
    >>> red   = KilonovaComponent(mej=0.040, vej=0.15, kappa=10.0, name="red")
    >>> model = MultiComponentKilonova([blue, red])
    >>> mags  = model.magnitude(np.linspace(0.5,10,50), "r", 40.0)
    """

    GW170817_PARAMS = {
        "blue":   dict(mej=0.025, vej=0.27, kappa=0.5),
        "purple": dict(mej=0.047, vej=0.15, kappa=3.0),
        "red":    dict(mej=0.011, vej=0.14, kappa=10.0),
    }

    def __init__(self, components: list):
        if not components:
            raise ValueError("Provide at least one KilonovaComponent.")
        self.components = components

    @classmethod
    def from_dict(cls, params: dict) -> "MultiComponentKilonova":
        """
        Construct from flat parameter dict.
        Keys: {name}_mej, {name}_vej, {name}_kappa.
        """
        names = sorted(set(k.rsplit("_", 1)[0] for k in params))
        return cls([
            KilonovaComponent(
                mej=params[f"{n}_mej"], vej=params[f"{n}_vej"],
                kappa=params[f"{n}_kappa"], name=n,
            )
            for n in names
        ])

    def total_flux(self, t_days, band, distance_mpc):
        """Total F_nu [erg/s/cm2/Hz] summed over all components."""
        lam  = FILTER_WAVELENGTHS[band]
        flux = np.zeros(len(np.atleast_1d(t_days)))
        for comp in self.components:
            flux += comp.blackbody_flux(t_days, lam, distance_mpc)
        return flux

    def magnitude(self, t_days, band, distance_mpc):
        """Combined AB magnitude from all components."""
        F = self.total_flux(t_days, band, distance_mpc)
        return -2.5 * np.log10(np.maximum(F, 1e-300)) - 48.6

    def component_magnitudes(self, t_days, band, distance_mpc):
        """Per-component magnitudes — useful for plotting decomposition."""
        return {
            c.name: c.magnitude(t_days, band, distance_mpc)
            for c in self.components
        }