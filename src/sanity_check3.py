import sys
sys.path.insert(0, "src")
import numpy as np
from kai.models.ssc import GRBShock
from kai.inference.ssc_likelihood import SSCLikelihood, ssc_priors_grb211211a
from kai.data.grb211211a_loaders import load_xrt_lightcurve, load_xrt_as_sed

df_xrt = load_xrt_lightcurve("data/GRB211211A/xrt_flux_lightcurve.qdp")
df_sed  = load_xrt_as_sed(df_xrt, t_center=4200, dt=800)

shock = GRBShock(eiso=1.25e52, density=1e-3,
                 tstart=3519, tstop=4972,
                 redshift=0.0763, scenario='ISM')

lk = SSCLikelihood(data=df_sed, shock=shock)
priors = ssc_priors_grb211211a()

# Simulate exactly what dynesty does internally
print("Simulating dynesty parameter passing:")
for i in range(5):
    sample = priors.sample()
    # bilby sets parameters this way internally
    for k, v in sample.items():
        lk.parameters[k] = v
    print(f"  parameters dict: {lk.parameters}")
    ln_l = lk.log_likelihood()
    print(f"  ln_L = {ln_l:.4f}\n")

# Check if parameters are None when log_likelihood is called without setting them
print("Test with None parameters (what bilby v3 does):")
for k in lk.parameters:
    lk.parameters[k] = None
print(f"  parameters: {lk.parameters}")
try:
    ln_l = lk.log_likelihood()
    print(f"  ln_L = {ln_l}")
except Exception as e:
    print(f"  ERROR: {e}")
