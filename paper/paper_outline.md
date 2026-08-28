# KAI: Bayesian multiwavelength inference of kilonova ejecta properties
# applied to GW170817
# Jean [Surname] et al. 2026 — A&A submission outline

---

## Abstract (~150 words)

Motivate: kilonova observations constrain r-process nucleosynthesis and
neutron star merger physics. State the gap: no lightweight, open-source,
well-tested Bayesian inference tool exists for multiwavelength kilonova
lightcurve fitting. Present KAI: a multi-component kilonova model coupled
to nested sampling via bilby. Apply to GW170817 photometry across g, r,
i, z, J, H bands. State key results: recovered ejecta masses, velocities,
and opacities consistent with Villar+2017 at X-sigma. Conclude: KAI is
publicly available and designed for application to future events with
LSST, DSA-2000, and next-generation GW detectors.

---

## 1. Introduction (~600 words)

### 1.1 Kilonovae as multimessenger probes
- GW170817/AT2017gfo as the golden event
- Kilonova emission: r-process heating, ejecta components, opacity
- Scientific goals: ejecta mass → nucleosynthesis yields,
  opacity → lanthanide fraction, velocity → merger dynamics

### 1.2 Existing modelling approaches
- Analytical/semi-analytical models: Arnett 1982, Metzger+2010
- Multi-component models: Kasen+2017, Villar+2017
- Radiative transfer: POSSIS (Bulla 2019), CMFGEN
- Gap: no open, pip-installable, tested Bayesian inference code

### 1.3 This work
- Present KAI: open-source, Bayesian, multi-component kilonova inference
- Validated against GW170817 public photometry
- Designed for scalability to future events

---

## 2. Physical model (~800 words)

### 2.1 Ejecta components
- Motivate blue (lanthanide-poor, κ ~ 0.5 cm²/g) and
  red (lanthanide-rich, κ ~ 10 cm²/g) components
- Physical origin: dynamical vs wind ejecta
- Reference to 3-component model (purple) as optional extension

### 2.2 Radioactive heating rate
- Korobkin et al. 2012 parameterisation
- Barnes et al. 2016 thermalization efficiency
- Equations: ε(t), f_th(t)

### 2.3 Bolometric lightcurve
- Arnett-like diffusion integral (Villar+2017 Eq. 1–3)
- Thick-to-thin transition at t_tr
- Diffusion timescale t_diff(mej, vej, κ)

### 2.4 Spectral energy distribution
- Blackbody approximation at photospheric temperature T(t)
- Bolometric correction to per-band AB magnitudes
- Filter effective wavelengths (Table 1)

### 2.5 Model limitations
- Grey opacity approximation
- Homologous expansion assumption
- No relativistic corrections (vej < 0.5c assumed)

---

## 3. Bayesian inference framework (~500 words)

### 3.1 Likelihood
- Gaussian likelihood on AB magnitudes
- Effective uncertainty: σ_eff² = σ_phot² + σ_floor²
- Systematic floor σ_floor = 0.05 mag (justify)

### 3.2 Priors (Table 2)
- Log-uniform on mej: [1e-4, 0.5] Msun
- Uniform on vej: [0.05, 0.5] c
- Log-uniform on κ, split by component
- Physically motivated ranges — discuss

### 3.3 Posterior sampling
- bilby + dynesty nested sampling
- Convergence: nlive=500, dlogZ < 0.1
- Posterior diagnostics: corner plots, trace plots

### 3.4 Model comparison
- Bayes factor: 1-component vs 2-component
- Interpret: evidence for blue+red decomposition in GW170817

---

## 4. Data (~300 words)

### 4.1 GW170817 photometry
- Source: Villar et al. 2017 (VizieR table)
- Bands: g, r, i, z, J, H
- Time range: 0.47 -- 14.5 days
- Total data points: N
- Data table in appendix

### 4.2 Data selection
- Exclusion of very early (<0.4 d) and very late (>15 d) data
- Upper limits treatment (excluded from likelihood, shown in figures)

---

## 5. Results (~700 words)

### 5.1 Posterior distributions (Figure 3)
- Corner plot: all 6 parameters
- Report medians and 68% credible intervals (Table 3)
- Compare to Villar+2017 MAP estimates

### 5.2 Posterior predictive lightcurves (Figure 4)
- Model fits data well in all bands
- Blue component dominates g, r at <2 days
- Red component dominates J, H at >3 days
- Residuals discussion

### 5.3 Model comparison
- Log Bayes factor ln B₂₁ = X ± Y
- Strong evidence for 2-component model over 1-component

### 5.4 Derived quantities
- Total ejecta mass: M_tot = M_blue + M_red
- Kinetic energy: E_kin ~ 0.5 * M_ej * v_ej²
- Implied r-process yield — comparison to solar abundances

---

## 6. Discussion (~500 words)

### 6.1 Comparison with Villar+2017
- Results consistent at 1-sigma
- Differences: priors, sigma_floor, sampler

### 6.2 Model limitations and systematic uncertainties
- Grey opacity is a strong approximation
- Distance uncertainty (40 ± 3 Mpc) — propagate?
- Host extinction (E(B-V) = 0.1 for NGC 4993)

### 6.3 Applicability to future events
- LSST: expected ~10s of kilonovae per year post-O5
- DSA-2000, SKA: radio afterglow extension
- How to extend KAI: hadronic component, neutrino flux

---

## 7. Conclusions (~200 words)

- Presented KAI: open-source Bayesian kilonova inference package
- Applied to GW170817: recovered ejecta parameters consistent with
  literature
- Model comparison: strong evidence for 2-component ejecta
- Package is public, pip-installable, fully tested, and documented
- Designed for the multimessenger era: LSST + next-gen GW detectors

---

## Acknowledgements

- HESS collaboration (data experience, not used here)
- Villar et al. 2017 for public photometry table
- bilby and dynesty development teams
- Funded by: [my future institute]

---

## Appendix A: Data table
Full GW170817 photometry used in this analysis.

## Appendix B: Convergence diagnostics
Dynesty trace plots, evidence convergence, sampling efficiency.

## Appendix C: 3-component model
Results with blue + purple + red components — comparison to 2-component.

---

## Figures list

| Figure | Description                          | Notebook cell |
|--------|--------------------------------------|---------------|
| 1      | GW170817 multiwavelength data        | Cell 2        |
| 2      | Model component decomposition        | Cell 3        |
| 3      | Posterior corner plot                | Cell 5        |
| 4      | Posterior predictive lightcurves     | Cell 6        |

## Tables list

| Table | Description                          |
|-------|--------------------------------------|
| 1     | Filter effective wavelengths         |
| 2     | Prior definitions                    |
| 3     | Posterior credible intervals         |
| 4     | Model comparison Bayes factors       |

---

## Target journal & format

- Journal   : Astronomy & Astrophysics (A&A)
- Template  : aa.cls (ESO LaTeX template)
- Word limit: none (Letters <4 pages; full article preferred here)
- Figures   : PDF vector format, max 10
- Data policy: posterior samples on Zenodo, code on GitHub
- Deadline  : aim for arXiv submission before DZA application (May 2026)