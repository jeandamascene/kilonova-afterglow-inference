"""
kai/inference/sampler.py
Bilby v3 compatible sampler wrapper.
"""

import bilby
from pathlib import Path


def run_inference(
    likelihood,
    priors,
    outdir: str = "results/gw170817",
    label: str  = "kilonova_2comp",
    sampler: str = "dynesty",
    nlive: int   = 500,
    clean: bool  = False,
    **sampler_kwargs,
) -> bilby.result.Result:
    """Run posterior sampling with bilby."""
    Path(outdir).mkdir(parents=True, exist_ok=True)

    result = bilby.run_sampler(
        likelihood = likelihood,
        priors     = priors,
        sampler    = sampler,
        nlive      = nlive,
        outdir     = outdir,
        label      = label,
        clean      = clean,
        save       = True,
        **sampler_kwargs,
    )
    return result
