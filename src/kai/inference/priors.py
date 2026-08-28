"""
kai/inference/priors.py

Prior definitions for kilonova parameter inference.

All priors are physically motivated:

    mej   : Log-uniform 1e-4 -- 0.5 Msun  (spans sub-dynamical to wind ejecta)
    vej   : Uniform      0.05 -- 0.50 c    (sub-relativistic to mildly relativistic)
    kappa : Log-uniform  0.1  -- 20  cm2/g (lanthanide-free to lanthanide-rich)

The GW170817 informed priors use tighter ranges from Villar+2017.
"""

import bilby
from bilby.core.prior import PriorDict, LogUniform, Uniform


def gw170817_priors(n_components: int = 2) -> PriorDict:
    """
    Physically motivated priors for GW170817 kilonova inference.

    Parameters
    ----------
    n_components : int  Number of ejecta components (1, 2, or 3).

    Returns
    -------
    PriorDict
    """
    from kai.inference.likelihood import COMPONENT_NAMES
    comp_names = COMPONENT_NAMES[n_components]

    priors = PriorDict()

    for name in comp_names:

        # Ejecta mass: log-uniform, wide range
        priors[f"{name}_mej"] = LogUniform(
            minimum=1e-4, maximum=0.5,
            name=f"{name}_mej",
            latex_label=rf"$M_{{ej,\rm {name}}}\ [M_\odot]$",
        )

        # Ejecta velocity: uniform
        priors[f"{name}_vej"] = Uniform(
            minimum=0.05, maximum=0.50,
            name=f"{name}_vej",
            latex_label=rf"$v_{{ej,\rm {name}}}\ [c]$",
        )

        # Opacity: log-uniform
        # Blue component: constrain to low-opacity regime
        if name == "blue":
            priors[f"{name}_kappa"] = LogUniform(
                minimum=0.1, maximum=2.0,
                name=f"{name}_kappa",
                latex_label=rf"$\kappa_{{\rm {name}}}\ [\rm cm^2/g]$",
            )
        # Red component: constrain to high-opacity regime
        elif name == "red":
            priors[f"{name}_kappa"] = LogUniform(
                minimum=2.0, maximum=20.0,
                name=f"{name}_kappa",
                latex_label=rf"$\kappa_{{\rm {name}}}\ [\rm cm^2/g]$",
            )
        # Purple (intermediate): full range
        else:
            priors[f"{name}_kappa"] = LogUniform(
                minimum=0.1, maximum=20.0,
                name=f"{name}_kappa",
                latex_label=rf"$\kappa_{{\rm {name}}}\ [\rm cm^2/g]$",
            )

    return priors


def broad_priors(n_components: int = 2) -> PriorDict:
    """
    Broad uninformative priors — useful for testing or
    applying the model to new events.
    """
    from kai.inference.likelihood import COMPONENT_NAMES
    comp_names = COMPONENT_NAMES[n_components]
    priors = PriorDict()
    for name in comp_names:
        priors[f"{name}_mej"]   = LogUniform(1e-4, 0.5,  name=f"{name}_mej")
        priors[f"{name}_vej"]   = Uniform(0.05,   0.50,  name=f"{name}_vej")
        priors[f"{name}_kappa"] = LogUniform(0.1,  20.0,  name=f"{name}_kappa")
    return priors