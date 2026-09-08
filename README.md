# BENDER: Cross-taxa IDP contact asymmetry

Code to reproduce all figures in the manuscript:  
**"Arginine out-contacts lysine across six kingdoms: chain physics amplifies a pairwise asymmetry into a structural outcome"**

---

## Repository structure

```
BENDER-github/
├── data/               ← input data (see below)
├── figures/            ← output PDFs and PNGs (generated)
├── fig1.py             ← dataset overview
├── fig2.py             ← dissociation / contact asymmetry
├── fig3.py             ← trajectory / SCD breakpoint
├── fig4.py             ← charge symmetry
├── fig5.py             ← synthesis: lambda-sweep, cross-FF, amplification
├── fig6.py             ← cross-FF validation
├── environment.yml
└── README.md
```

## Data files

Place the following files in `data/` before running:

| File | Description |
|------|-------------|
| `protein_metadata.csv` | ν, kingdom, sequence length per UniProt ID |
| `substitution_results.csv` | K→R and R→K substitution contact efficiency results |
| `contact_results.csv` | Raw contact results (CALVADOS-2) |
| `mpipi_results.csv` | Raw contact results (MPIPI) |
| `lambda_sweep.csv` | λ_R sweep: ν and contact efficiency per protein per λ |
| `delta_nu_substitution.csv` | Δν per protein per substitution variant |
| `pae_concordance.tsv` | CALVADOS-2 vs AlphaFold PAE concordance |

Raw data and simulation trajectories are deposited at Zenodo: **[DOI — to be added]**

## Environment

```bash
conda env create -f environment.yml
conda activate bender
```

## Reproducing figures

```bash
python fig1.py
python fig2.py
python fig3.py
python fig4.py
python fig5.py
python fig6.py
```

Figures are written to `figures/` as both PDF and PNG.

## Key result

Arg/Lys contact efficiency ratio = **1.37× (95% CI: 1.33–1.42)** across 340 protein pairs.  
Chain physics amplifies the pairwise advantage **7.1× (95% CI: 6.3–8.0)**,  
consistent across all six kingdoms independently.
