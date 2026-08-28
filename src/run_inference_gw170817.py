"""
run_inference_gw170817.py

Full inference run on GW170817 photometry.
Run with:  python run_inference_gw170817.py
"""

import numpy as np
import matplotlib.pyplot as plt
from kai.data.loaders import load_gw170817_photometry, summarise_dataset
from kai.inference.likelihood import KilonovaLikelihood
from kai.inference.priors import gw170817_priors
from kai.inference.sampler import run_inference
from kai.models.kilonova import MultiComponentKilonova, KilonovaComponent

# ── Configuration ─────────────────────────────────────────────────────────────
BANDS         = ["g", "r", "i", "z", "J", "H"]
DISTANCE_MPC  = 40.0
N_COMPONENTS  = 2
NLIVE         = 1000     # NLIVE 500 increase to 1000+ for publication quality

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading GW170817 photometry...")
df = load_gw170817_photometry(bands=BANDS, t_max=15.0)
summarise_dataset(df)

# ── Set up likelihood and priors ──────────────────────────────────────────────
likelihood = KilonovaLikelihood(
    data         = df,
    n_components = N_COMPONENTS,
    distance_mpc = DISTANCE_MPC,
    sigma_floor  = 0.05,
)
likelihood.summary()

priors = gw170817_priors(n_components=N_COMPONENTS)
print("\nPriors:")
print(priors)

# ── Diagnostic: test likelihood before sampling ───────────────────────────────
import numpy as np
print("\n--- Pre-flight likelihood check ---")
test_params = {
    "blue_mej": 0.025, "blue_vej": 0.27, "blue_kappa": 0.5,
    "red_mej":  0.040, "red_vej":  0.15, "red_kappa":  10.0,
}
for k, v in test_params.items():
    likelihood.parameters[k] = v
ln_l = likelihood.log_likelihood()
print(f"log_likelihood at test params: {ln_l}")
print(f"finite: {np.isfinite(ln_l)}")

# Test 10 random prior samples
print("\nTesting 10 random prior samples:")
for i in range(10):
    sample = priors.sample()
    for k, v in sample.items():
        likelihood.parameters[k] = v
    ln_l = likelihood.log_likelihood()
    print(f"  sample {i}: {sample} -> ln_l = {ln_l:.2f}")
# ─────────────────────────────────────────────────────────────────────────────

# Test what bilby sees when it calls log_likelihood internally
print("\n--- Bilby internal parameter test ---")
# Simulate what dynesty does: set parameters then call log_likelihood
for k in likelihood.parameters:
    likelihood.parameters[k] = 0.025 if "mej" in k else (0.2 if "vej" in k else 1.0)
print(f"Parameters set to: {likelihood.parameters}")
print(f"log_likelihood called directly: {likelihood.log_likelihood()}")

########### end to remove #########

# ── Run inference ─────────────────────────────────────────────────────────────
print("\nRunning posterior sampling...")
result = run_inference(
    likelihood = likelihood,
    priors     = priors,
    outdir     = "results/gw170817",
    label      = f"kilonova_{N_COMPONENTS}comp",
    sampler    = "dynesty",
    nlive      = NLIVE,
)

# ── Print results ─────────────────────────────────────────────────────────────
print("\nPosterior summary:")
#print(result.summary())
# ── Print results ─────────────────────────────────────────────────────────────
print("\nPosterior credible intervals (median ± 1σ):")
for param in priors.keys():
    samples = result.posterior[param]
    lo, med, hi = np.percentile(samples, [16, 50, 84])
    print(f"  {param:20s} = {med:.4f} + {hi-med:.4f} - {med-lo:.4f}")

print(f"\nlog Evidence (logZ) = {result.log_evidence:.2f} ± {result.log_evidence_err:.2f}")

# ── Corner plot ───────────────────────────────────────────────────────────────
result.plot_corner()

# ── Posterior predictive lightcurves ─────────────────────────────────────────
t_plot = np.linspace(0.3, 16, 200)
band_colors = {"g": "blue", "r": "green", "i": "orange",
               "z": "red",  "J": "purple", "H": "brown"}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for ax, band in zip(axes, BANDS):
    # Plot posterior predictive (100 samples)
    samples = result.posterior.sample(100)
    for _, row in samples.iterrows():
        from kai.inference.likelihood import build_model_from_params
        from kai.inference.likelihood import COMPONENT_NAMES
        model = build_model_from_params(
            row.to_dict(), COMPONENT_NAMES[N_COMPONENTS]
        )
        mag = model.magnitude(t_plot, band=band, distance_mpc=DISTANCE_MPC)
        ax.plot(t_plot, mag, color=band_colors[band], alpha=0.08, lw=0.7)

    # Plot data
    band_data = df[df["band"] == band]
    ax.errorbar(
        band_data["t_days"], band_data["mag"], yerr=band_data["mag_err"],
        fmt="ko", ms=5, capsize=3, zorder=5
    )

    ax.invert_yaxis()
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("AB magnitude")
    ax.set_title(f"{band}-band")

plt.suptitle("GW170817 kilonova — posterior predictive", fontsize=13)
plt.tight_layout()
plt.savefig("results/gw170817/posterior_lightcurves.png", dpi=150)
plt.show()

print("\nDone. Results saved to results/gw170817/")