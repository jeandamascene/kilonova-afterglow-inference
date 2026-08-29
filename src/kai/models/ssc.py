"""
kai/models/ssc.py

One-zone Synchrotron Self-Compton (SSC) afterglow model for GRBs.

Physics
-------
Based on the Blandford & McKee (1976) forward shock model. Given the
isotropic energy of the explosion (Eiso), the circumburst density (n),
and the observation time, the code computes:

    - Lorentz factor Gamma and shell radius R (Blandford-McKee dynamics)
    - Electron distribution: ExponentialCutoffBrokenPowerLaw (naima)
    - Synchrotron emission (naima)
    - Inverse Compton / SSC emission (naima)
    - gamma-gamma absorption (pair production) — two methods

Three circumburst medium scenarios are supported:
    - ISM   : homogeneous interstellar medium of density n [cm^-3]
    - Wind  : r^-2 density profile from stellar wind
    - average : geometric mean of ISM and Wind

Free parameters (sampled by bilby):
    - log10(eta_e)   : electron energy fraction          [-5,  0]
    - log10(E_break) : electron break energy [TeV]       [-6,  1]
    - alpha2         : high-energy electron index        [-1,  5]
    - log10(E_cut)   : electron cutoff energy [TeV]      [ 1, cutoff_limit(B)]
    - log10(B)       : magnetic field [G]                [-3,  1]

This module is a refactored version of grbloader.py (Jean, H.E.S.S. 2023/2024),
adapted for integration with the KAI bilby inference framework.

References
----------
- Blandford & McKee 1976, Physics of Fluids, 19, 1130
- Aharonian 2004, Very High Energy Cosmic-ray Sources
- Aharonian 2000, New Astronomy, 4, 377
- Eungwanichayapant & Aharonian 2009

Usage
-----
    from kai.models.ssc import GRBShock, SSCModel
    import numpy as np
    import astropy.units as u

    shock = GRBShock(eiso=8e53, density=0.5, tstart=68, tstop=110,
                     redshift=0.4245, scenario='ISM')

    model = SSCModel(shock)

    energy = np.logspace(3, 13, 100) * u.eV
    params = {
        'log10_eta_e':  -1.6,
        'log10_ebreak': -1.7,
        'alpha2':        3.1,
        'log10_ecut':    1.2,
        'log10_B':       0.37,
    }
    sed = model.sed(energy, params)
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import astropy.units as u
import astropy.constants as con
from astropy.cosmology import WMAP9 as cosmo
from naima.models import Synchrotron, InverseCompton, ExponentialCutoffBrokenPowerLaw
import naima.utils

# ── Physical constants (CGS) ──────────────────────────────────────────────────
M_E       = con.m_e.cgs.value
C_LIGHT   = con.c.cgs.value
MEC2_EV   = (con.m_e * con.c**2).to('eV').value
H_PLANCK  = con.h.cgs.value
EL        = con.e.gauss.value
ERG_TO_EV = 624150912588.3258
SIGMA_T   = con.sigma_T.cgs.value
MPC2      = (con.m_p * con.c**2).to('eV')
MPC2_ERG  = MPC2.to('erg').value


# ── Gamma-gamma absorption ────────────────────────────────────────────────────

def sigma_gammagamma(Eph1: np.ndarray, Eph2: np.ndarray) -> u.Quantity:
    # Both Eph1 and Eph2 are plain numpy arrays (eV values, no units)
    CMene = Eph1 * Eph2 / (MEC2_EV**2)   # dimensionless — no .decompose() needed
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

'''
def sigma_gammagamma(Eph1: np.ndarray, Eph2: np.ndarray) -> u.Quantity:
	"""
	Angle-averaged gamma-gamma pair production cross section [cm^2].
	Approximation from Eungwanichayapant & Aharonian 2009 (Eq. 5),
	originally from Aharonian 2004. Accurate within 3%.
	Parameters
	----------
	Eph1 : array-like  Gamma-ray energy [eV]
	Eph2 : array-like  Target photon energy [eV]
	Returns
	-------
	cross_section : Quantity  [cm^2]
	"""
	CMene = (Eph1 * Eph2 / (MEC2_EV**2)).decompose().value
	mask  = CMene > 1.0
	res   = np.full(CMene.shape, 0.0)
	res[mask] = (3.0 / (2.0 * CMene[mask]**2) * SIGMA_T
	* (
	(CMene[mask] + 0.5 * np.log(CMene[mask])
	- 1.0/6.0 + 1.0 / (2.0 * CMene[mask]))
	* np.log(np.sqrt(CMene[mask]) + np.sqrt(CMene[mask] - 1.0))
	- (CMene[mask] + 4.0/9.0 - 1.0 / (9.0 * CMene[mask]))
	* np.sqrt(1.0 - 1.0 / CMene[mask])
	)
	)
	return res * u.cm**2

'''
def absorption_coeff(egamma, targetene, target):
	"""
	Absorption coefficient K(E) = integral sigma_gg(E,e) * dn/de de [cm^-1].
	Inner integral of Aharonian 2004 Eq. 3.24.
	Parameters
	----------
	egamma    : Quantity  Gamma-ray energy array [eV units]
	targetene : Quantity  Target photon energy array [eV units]
	target    : Quantity  Target photon density dn/de [1/(eV cm^3)]
	Returns
	-------
	abs_coeff : Quantity  [cm^-1]
	"""
	product = sigma_gammagamma(
	np.vstack(egamma.to('eV').value),   # strip units before vstack
	targetene.to('eV').value) * target
	return naima.utils.trapz_loglog(product, targetene, axis=1)

def tau_gammagamma(
    egamma: u.Quantity,
    targetene: u.Quantity,
    target: u.Quantity,
    size: u.Quantity,
) -> np.ndarray:
    """
    Optical depth for gamma-gamma absorption (homogeneous radiation field).

    tau = size * K(E)   (Aharonian 2004 Eq. 3.24)

    Parameters
    ----------
    egamma    : Quantity  Gamma-ray energy array
    targetene : Quantity  Target photon energy array
    target    : Quantity  Target photon density dn/de [1/(eV cm^3)]
    size      : Quantity  Absorption path length [cm units]

    Returns
    -------
    tau : np.ndarray  Dimensionless optical depth
    """
    coeff = absorption_coeff(egamma, targetene, target)
    return (size.to('cm') * coeff).decompose().value


def cutoff_limit(bfield: float) -> float:
    """
    Maximum synchrotron cutoff energy from synchrotron burn-off [log10 TeV].

    Balances acceleration and synchrotron cooling losses.
    Expression 18 from Aharonian 2000.

    Parameters
    ----------
    bfield : float  Magnetic field [Gauss]

    Returns
    -------
    float  log10 of cutoff energy [TeV]
    """
    eff    = 1.0
    cutoff = (
        (3.0/2.0)**(3.0/4.0)
        * np.sqrt(1.0 / (EL**3 * bfield))
        * (M_E**2 * C_LIGHT**4)
        * eff**(-0.5)
    ) * u.erg
    return np.log10(cutoff.value * ERG_TO_EV * 1e-12)


def synch_cooling_time(bfield: u.Quantity, partene: u.Quantity) -> u.Quantity:
    """
    Synchrotron cooling time for an electron [seconds].

    Equation 1 from Aharonian 2000.

    Parameters
    ----------
    bfield  : Quantity  Magnetic field [Gauss units]
    partene : Quantity  Electron energy [eV units]

    Returns
    -------
    tcool : Quantity  [seconds]
    """
    bf   = bfield.to('G').value
    epar = partene.to('erg').value
    tcool = (6.0 * np.pi * M_E**4 * C_LIGHT**3) / (
        SIGMA_T * M_E**2 * epar * bf**2
    )
    return tcool * u.s


def synch_char_energy(bfield: u.Quantity, partene: u.Quantity) -> u.Quantity:
    """
    Characteristic synchrotron photon energy for an electron [eV].

    Equation 3.30 from Aharonian 2004 (adapted for electrons).

    Parameters
    ----------
    bfield  : Quantity  Magnetic field [Gauss units]
    partene : Quantity  Electron energy [eV units]

    Returns
    -------
    charene : Quantity  [eV]
    """
    bf    = bfield.to('G').value
    epar  = partene.to('erg').value
    charene = (
        np.sqrt(3.0/2.0)
        * (H_PLANCK * EL * bf)
        / (2.0 * np.pi * M_E**3 * C_LIGHT**5)
        * epar**2
    )
    return charene * ERG_TO_EV * u.eV


# ── Shock dynamics ────────────────────────────────────────────────────────────

@dataclass
class GRBShock:
    """
    Blandford-McKee (1976) forward shock dynamics for a GRB afterglow.

    Computes the bulk Lorentz factor Gamma and shell radius R at the
    average observation time t_obs = (tstart + tstop) / 2.

    Parameters
    ----------
    eiso       : float  Isotropic equivalent energy [erg]
    density    : float  Circumburst density [cm^-3] (ISM/average)
    tstart     : float  Start of observation window [s after trigger]
    tstop      : float  End of observation window [s after trigger]
    redshift   : float  Source redshift
    scenario   : str    'ISM', 'Wind', or 'average'
    mass_loss  : float  Progenitor mass loss rate [Msun/yr] (Wind only)
    wind_speed : float  Progenitor wind speed [km/s] (Wind only)

    Derived attributes (set in __post_init__)
    -----------------------------------------
    avtime       : float          Average observation time [s]
    gamma        : float          Bulk Lorentz factor
    sizer        : float          Shell radius [cm]
    depthpar     : float          Depth parameter (R / depthpar / Gamma = shell depth)
    shock_energy : float          Shock energy density [erg/cm^3]
    dl           : u.Quantity     Luminosity distance
    """
    eiso:       float
    density:    float
    tstart:     float
    tstop:      float
    redshift:   float
    scenario:   str   = 'ISM'
    mass_loss:  float = 0.0
    wind_speed: float = 0.0

    avtime:       float      = field(init=False)
    gamma:        float      = field(init=False)
    sizer:        float      = field(init=False)
    depthpar:     float      = field(init=False)
    shock_energy: float      = field(init=False)
    dl:           u.Quantity = field(init=False)

    def __post_init__(self):
        self.avtime = (self.tstart + self.tstop) / 2.0
        self.dl     = cosmo.luminosity_distance(self.redshift)
        self._compute_dynamics()

    def _compute_dynamics(self):
        """Compute Gamma, R, depthpar from Blandford-McKee 1976."""
        if self.scenario == 'ISM':
            self.gamma = (
                (1.0/8.0)**(3.0/8.0)
                * (3.0 * self.eiso
                   / (4.0 * np.pi * self.density * MPC2_ERG
                      * (C_LIGHT * self.avtime)**3))**0.125
            )
            self.sizer    = 8.0 * C_LIGHT * self.avtime * self.gamma**2
            self.depthpar = 9.0

        elif self.scenario == 'Wind':
            if self.mass_loss == 0 or self.wind_speed == 0:
                raise ValueError(
                    "Wind scenario requires non-zero mass_loss and wind_speed."
                )
            self.gamma = (
                (3.0 * self.eiso * self.wind_speed * 1e5)
                / (4.0 * C_LIGHT**3 * self.avtime
                   * self.mass_loss * 2e33 / 3.15e7)
            )**0.25
            self.sizer    = 4.0 * self.gamma**2 * C_LIGHT * self.avtime
            self.depthpar = 3.0
            self.density  = (
                (self.mass_loss * 1.2e57 / 3.15e7)
                / (4.0 * np.pi * self.wind_speed * 1e5 * self.sizer**2)
            )

        elif self.scenario == 'average':
            self.gamma = (
                (1.0/6.0)**(3.0/8.0)
                * (3.0 * self.eiso
                   / (4.0 * np.pi * self.density * MPC2_ERG
                      * (C_LIGHT * self.avtime)**3))**0.125
            )
            self.sizer    = 6.0 * C_LIGHT * self.avtime * self.gamma**2
            self.depthpar = 9.0 / 2.0

        else:
            raise ValueError(
                f"Unknown scenario '{self.scenario}'. "
                "Choose 'ISM', 'Wind', or 'average'."
            )

        # Shock energy density [erg/cm^3]
        self.shock_energy = 2.0 * self.gamma**2 * self.density * MPC2_ERG

    def shell_volume(self) -> float:
        """Thin shell volume [cm^3]. Eq. 7 from H.E.S.S. GRB190829A paper."""
        return 4.0 * np.pi * self.sizer**2 * (
            self.sizer / (self.depthpar * self.gamma)
        )

    def summary(self) -> None:
        """Print shock dynamics summary."""
        print(f"GRBShock ({self.scenario})")
        print(f"  Eiso         = {self.eiso:.2e} erg")
        print(f"  density      = {self.density:.3e} cm^-3")
        print(f"  t_obs        = {self.avtime:.1f} s")
        print(f"  Gamma        = {self.gamma:.2f}")
        print(f"  R            = {self.sizer:.3e} cm")
        print(f"  E_shock/vol  = {self.shock_energy:.3e} erg/cm^3")
        print(f"  redshift     = {self.redshift}")
        print(f"  D_L          = {self.dl:.3f}")


# ── SSC emission model ────────────────────────────────────────────────────────

class SSCModel:
    """
    One-zone SSC emission model for GRB afterglows.

    Computes the broadband SED (synchrotron + inverse Compton + gamma-gamma
    absorption) given shock dynamics from a GRBShock instance and a
    set of microphysics parameters.

    Parameters
    ----------
    shock             : GRBShock  Shock dynamics object
    absorption_method : int       Gamma-gamma absorption method:
                                  1 = simple exp(-tau)
                                  2 = Rybicki & Lightman 1979 (default)

    After calling sed(), the following attributes are populated:
        synch_comp   : synchrotron SED without absorption
        ic_comp      : IC/SSC SED without absorption
        synch_compGG : synchrotron SED with gamma-gamma absorption
        ic_compGG    : IC/SSC SED with gamma-gamma absorption
        emin         : minimum injection energy of electron distribution
        eta_e_val    : electron energy fraction
        eta_b_val    : magnetic field energy fraction
    """

    def __init__(self, shock: GRBShock, absorption_method: int = 2):
        self.shock             = shock
        self.absorption_method = absorption_method
        self.synch_comp        = None
        self.ic_comp           = None
        self.synch_compGG      = None
        self.ic_compGG         = None
        self.emin              = None
        self.eta_e_val         = None
        self.eta_b_val         = None

    def _photon_density(
        self,
        lsy: u.Quantity,
        sizereg: u.Quantity,
    ) -> u.Quantity:
        """
        Synchrotron photon number density dn/dE [1/(eV cm^3)].
        Thin shell: n_ph = L_sy / (4*pi*R^2*c)
        """
        return lsy / (4.0 * np.pi * sizereg**2 * C_LIGHT * u.cm / u.s)

    def sed(
        self,
        energy: u.Quantity,
        params: dict,
    ) -> u.Quantity:
        """
        Compute broadband SED including gamma-gamma absorption [erg/s/cm^2].

        Parameters
        ----------
        energy : Quantity
            Photon energy array with astropy units (e.g. np.logspace(3,13,100)*u.eV)
        params : dict
            Keys:
              'log10_eta_e'  : log10 electron energy fraction
              'log10_ebreak' : log10 break energy [TeV]
              'alpha2'       : high-energy electron index
              'log10_ecut'   : log10 cutoff energy [TeV]
              'log10_B'      : log10 magnetic field [Gauss]

        Returns
        -------
        total_sed : Quantity  Total SED [erg/s/cm^2]
        """
        # ── Unpack parameters ─────────────────────────────────────────────
        eta_e  = 10.0 ** params['log10_eta_e']
        ebreak = 10.0 ** params['log10_ebreak'] * u.TeV
        alpha2 = params['alpha2']
        alpha1 = alpha2 - 1.0          # cooling break — fixed relation
        ecut   = 10.0 ** params['log10_ecut'] * u.TeV
        bfield = 10.0 ** params['log10_B'] * u.G

        # ── Shock-derived quantities ──────────────────────────────────────
        shock        = self.shock
        redf         = 1.0 + shock.redshift
        doppler      = shock.gamma
        size_reg     = shock.sizer * u.cm
        vol          = shock.shell_volume() * u.cm**3
        shock_e_dens = shock.shock_energy * u.erg / u.cm**3
        eemax        = ecut.to('eV').value * 1e13    # max electron energy [eV]

        self.eta_e_val = eta_e
        self.eta_b_val = (bfield.value**2 / (8.0 * np.pi)) / shock.shock_energy

        # ── Step 1: electron distribution with temporary amplitude ────────
        ampl_tmp = 1.0 / u.eV
        ECBPL = ExponentialCutoffBrokenPowerLaw(
            ampl_tmp, 1.0 * u.TeV, ebreak, alpha1, alpha2, ecut
        )

        # ── Step 2: self-consistent Emin from eta_e (energy conservation) ─
        ener_grid = np.logspace(9, np.log10(eemax), 100) * u.eV
        eldis     = ECBPL(ener_grid)
        mean_e    = (
            naima.utils.trapz_loglog(ener_grid * eldis, ener_grid)
            / naima.utils.trapz_loglog(eldis, ener_grid)
        )
        emin = (eta_e * shock.gamma * MPC2) / mean_e * 1e9 * u.eV
        self.emin = emin

        # ── Step 3: physical amplitude from energy conservation ───────────
        SYN_tmp = Synchrotron(
            ECBPL, B=bfield, Eemin=emin, Eemax=eemax * u.eV, nEed=20
        )
        we_tmp = SYN_tmp.compute_We(Eemin=emin, Eemax=eemax * u.eV)
        ampl   = ((eta_e * shock_e_dens * vol) / we_tmp) / u.eV

        ECBPL = ExponentialCutoffBrokenPowerLaw(
            ampl, 1.0 * u.TeV, ebreak, alpha1, alpha2, ecut
        )
        SYN = Synchrotron(
            ECBPL, B=bfield, Eemin=emin, Eemax=eemax * u.eV, nEed=20
        )

        # ── Step 4: synchrotron photon field for SSC seed ─────────────────
        cutoff_char  = np.log10(synch_char_energy(bfield, ecut).value)
        bins_per_dec = 20
        bins         = int((cutoff_char - (-4.0)) * bins_per_dec)
        Esy          = np.logspace(-4.0, cutoff_char + 1, bins) * u.eV
        Lsy          = SYN.flux(Esy, distance=0 * u.cm)   # photons/eV/s
        phn_sy       = self._photon_density(Lsy, size_reg)

        # ── Step 5: IC / SSC component ────────────────────────────────────
        IC = InverseCompton(
            ECBPL,
            seed_photon_fields=[['SSC', Esy, phn_sy]],
            Eemin=emin,
            Eemax=eemax * u.eV,
            nEed=20,
        )

        # ── Step 6: observed SED with Doppler boosting and redshift ───────
        obs_energy = energy / doppler * redf
        synch      = doppler**2 * SYN.sed(obs_energy, distance=shock.dl)
        ic         = doppler**2 * IC.sed(obs_energy,  distance=shock.dl)

        self.synch_comp = synch
        self.ic_comp    = ic

        # ── Step 7: gamma-gamma absorption ───────────────────────────────
        tau = tau_gammagamma(
            obs_energy,
            Esy,
            phn_sy,
            size_reg / (shock.depthpar * shock.gamma),
        )

        if self.absorption_method == 1:
            # Simple exponential attenuation (Method 1)
            self.synch_compGG = synch * np.exp(-tau)
            self.ic_compGG    = ic   * np.exp(-tau)

        else:
            # Rybicki & Lightman 1979 Eq. 1.29-1.30 (Method 2, default)
            # Accounts for photons produced inside the absorbing region
            synch_gg = synch.copy()
            ic_gg    = ic.copy()
            mask = tau > 1e-4
            synch_gg[mask] = synch[mask] / tau[mask] * (1.0 - np.exp(-tau[mask]))
            ic_gg[mask]    = ic[mask]    / tau[mask] * (1.0 - np.exp(-tau[mask]))
            self.synch_compGG = synch_gg
            self.ic_compGG    = ic_gg

        return self.synch_compGG + self.ic_compGG

    def cooling_time_at_break(self, params: dict) -> float:
        """
        Synchrotron cooling time at the break energy [s].
        Used to implement the cooling constraint prior.
        """
        bfield = 10.0 ** params['log10_B'] * u.G
        ebreak = 10.0 ** params['log10_ebreak'] * u.TeV
        return synch_cooling_time(bfield, ebreak).value

    def summary(self) -> None:
        """Print model summary."""
        print("SSCModel")
        print(f"  Absorption method : {self.absorption_method}")
        self.shock.summary()
        if self.emin is not None:
            print(f"  Emin             = {self.emin:.3e}")
            print(f"  eta_e            = {self.eta_e_val:.4f}")
            print(f"  eta_b            = {self.eta_b_val:.4e}")
