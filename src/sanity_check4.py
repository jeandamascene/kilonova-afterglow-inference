import sys
sys.path.insert(0, "src")
import numpy as np
from kai.models.ssc import GRBShock
from kai.inference.ssc_likelihood import SSCLikelihood, ssc_priors_grb211211a
from kai.data.grb211211a_loaders import load_xrt_lightcurve, load_xrt_as_sed

df_xrt = load_xrt_lightcurve("data/GRB211211A/xrt_flux_lightcurve.qdp")
df_sed  = load_xrt_as_sed(df_xrt, t_center=4200, dt=800)
shock   = GRBShock(eiso=1.25e52, density=1e-3, tstart=3519, tstop=4972,
                   redshift=0.0763, scenario='ISM')
lk      = SSCLikelihood(data=df_sed, shock=shock)
priors  = ssc_priors_grb211211a()

finite = 0
for i in range(50):
    s = priors.sample()
    for k, v in s.items():
        lk.parameters[k] = v
    ln_l = lk.log_likelihood()
    if np.isfinite(ln_l):
        finite += 1
        print(f"  FINITE {i}: ln_L={ln_l:.2f} "
              f"B={s['log10_B']:.2f} ebreak={s['log10_ebreak']:.2f} "
              f"ecut={s['log10_ecut']:.2f}")

print(f"\nFinite: {finite}/50")

finite = 0
for i in range(50):
    s = priors.sample()
    for k, v in s.items():
        lk.parameters[k] = v
    ln_l = lk.log_likelihood()
    if np.isfinite(ln_l):
        finite += 1
print(f"Finite: {finite}/50")

# Also test reference params
ref = {'log10_eta_e': -0.04, 'log10_ebreak': -1.513,
       'alpha2': 3.15, 'log10_ecut': 1.7, 'log10_B': -0.448}
for k, v in ref.items():
    lk.parameters[k] = v
print(f"Reference ln_L: {lk.log_likelihood():.4f}")
