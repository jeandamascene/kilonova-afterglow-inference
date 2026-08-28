"""
debug_kai.py — diagnostic script for KAI model and likelihood
Run from repo root: python debug_kai.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from kai.models.kilonova import (
    KilonovaComponent, MultiComponentKilonova,
    heating_rate, thermalization_efficiency,
)
from kai.data.loaders import load_gw170817_photometry
from kai.inference.likelihood import KilonovaLikelihood

print("=" * 60)
print("STEP 1: heating rate")
t = np.array([0.5, 1.0, 2.0, 5.0])
eps = heating_rate(t)
print(f"  t        = {t}")
print(f"  epsilon  = {eps}")
print(f"  finite?  = {np.all(np.isfinite(eps))}")

print()
print("=" * 60)
print("STEP 2: KilonovaComponent")
blue = KilonovaComponent(mej=0.025, vej=0.27, kappa=0.5, name="blue")
print(f"  t_diff       = {blue.t_diff():.3f} days")
print(f"  t_transition = {blue.t_transition():.3f} days")

print()
print("STEP 3: bolometric luminosity")
L = blue.bolometric_luminosity(t)
print(f"  L = {L}")
print(f"  finite?   = {np.all(np.isfinite(L))}")
print(f"  all zero? = {np.all(L == 0)}")

print()
print("STEP 4: temperature")
T = blue.temperature(t)
print(f"  T = {T}")

print()
print("STEP 5: blackbody flux")
F = blue.blackbody_flux(t, wavelength_aa=6230.0, distance_mpc=40.0)
print(f"  F_nu = {F}")

print()
print("STEP 6: magnitude")
mag = blue.magnitude(t, band="r", distance_mpc=40.0)
print(f"  mag (r-band) = {mag}")
print(f"  finite? = {np.all(np.isfinite(mag))}")

print()
print("=" * 60)
print("STEP 7: MultiComponentKilonova")
red   = KilonovaComponent(mej=0.040, vej=0.15, kappa=10.0, name="red")
model = MultiComponentKilonova([blue, red])
mag2  = model.magnitude(t, band="r", distance_mpc=40.0)
print(f"  combined mag (r-band) = {mag2}")

print()
print("=" * 60)
print("STEP 8: data loader")
df = load_gw170817_photometry(bands=["r", "J"])
print(f"  data points loaded: {len(df)}")
print(df)

print()
print("=" * 60)
print("STEP 9: likelihood at best-fit params")
lk = KilonovaLikelihood(data=df, n_components=2, distance_mpc=40.0)
lk.summary()

best_fit = {
    "blue_mej":   0.025,
    "blue_vej":   0.27,
    "blue_kappa": 0.5,
    "red_mej":    0.040,
    "red_vej":    0.15,
    "red_kappa":  10.0,
}
for k, v in best_fit.items():
    lk.parameters[k] = v

ln_l = lk.log_likelihood()
print(f"  log_likelihood = {ln_l:.4f}")
print(f"  finite?        = {np.isfinite(ln_l)}")