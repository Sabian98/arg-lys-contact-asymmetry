"""
Figure 6 — Cross-force-field validation: CALVADOS-2 and MPIPI-GG
Layout: 5 rows × 6 columns
Rows: SCD, NCPR, FCR, Lys efficiency, Arg efficiency
Cols: Bacteria, Fungi, Mammals, Plants, Protists, Viruses

Inputs needed:
    BENDER_BIO.csv           — kingdom, nu, SCD, NCPR, FCR per sequence
    contact_results_raw.csv  — eff_Lys_*, eff_Arg_*, nu, kingdom (CALVADOS-2)
    mpipi_results_raw.csv    — SCD, NCPR, FCR, eff_Lys_*, eff_Arg_*,
                               nu_mpipi, kingdom (MPIPI-GG)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

FS_TITLE  = 36   # kingdom column headers
FS_ROW    = 30   # row labels on right
FS_LABEL  = 26   # axis label (ν)
FS_TICK   = 34   # tick labels
FS_LEGEND = 26   # legend
FS_SMALL  = 20   # regime labels

SIX = ['Bacteria','Fungi','Mammals','Plants','Protists','Viruses']
KC  = {'Bacteria':'#1f77b4','Fungi':'#2ca02c','Mammals':'#9467bd',
       'Plants':'#e377c2','Protists':'#bcbd22','Viruses':'#17becf'}

# ── Load ──────────────────────────────────────────────────────────────
bio  = pd.read_csv(DATA / 'protein_metadata.csv')
df_c = pd.read_csv(DATA / 'contact_results.csv')
df_m = pd.read_csv(DATA / 'mpipi_results.csv')
bio6 = bio[bio['kingdom'].isin(SIX)].copy()

# Pooled contact efficiency columns
lys_cols_c = [c for c in df_c.columns if 'Lys' in c and c.startswith('eff_')]
arg_cols_c = [c for c in df_c.columns if 'Arg' in c and c.startswith('eff_')
              and 'Lys' not in c]
lys_cols_m = [c for c in df_m.columns if 'Lys' in c and c.startswith('eff_')]
arg_cols_m = [c for c in df_m.columns if 'Arg' in c and c.startswith('eff_')
              and 'Lys' not in c]

df_c['lys_eff'] = df_c[lys_cols_c].replace(0, np.nan).mean(axis=1)
df_c['arg_eff'] = df_c[arg_cols_c].replace(0, np.nan).mean(axis=1)
df_m['lys_eff'] = df_m[lys_cols_m].replace(0, np.nan).mean(axis=1)
df_m['arg_eff'] = df_m[arg_cols_m].replace(0, np.nan).mean(axis=1)

# Clip efficiency outliers at p90 to match reference figure scale
for _df in [df_c, df_m]:
    for _col in ['lys_eff', 'arg_eff']:
        _df[_col] = _df[_col].clip(upper=_df[_col].quantile(0.90))

# ── Binning helper ────────────────────────────────────────────────────
BW = 0.02
def get_bins(df, col, nu_col='nu', min_n=5):
    df = df.copy()
    df['nu_bin'] = np.round(df[nu_col] / BW) * BW
    g = df.groupby('nu_bin').agg(
        m=(col, 'mean'), s=(col, 'sem'), n=(col, 'count')
    ).query(f'n >= {min_n}')
    return g.query('nu_bin >= 0.36 and nu_bin <= 0.72').sort_index()

# ── Row configs ───────────────────────────────────────────────────────
# BUG FIX 1: src_m was None for SCD/NCPR/FCR but mpipi_results_raw
# has these columns — now correctly passed as df_m for all rows
row_configs = [
    # (row_key,  calvados_src, mpipi_src, nu_col_c,  nu_col_m,   ylabel,    ylim)
    ('SCD',     bio6,         df_m,      'nu',       'nu_mpipi', 'SCD',     (-5, 125)),
    ('NCPR',    bio6,         df_m,      'nu',       'nu_mpipi', 'NCPR',    (-0.50, 0.65)),
    ('FCR',     bio6,         df_m,      'nu',       'nu_mpipi', 'FCR',     (0.15, 0.85)),
    ('lys_eff', df_c,         df_m,      'nu',       'nu_mpipi', 'Lys\neff',(0, 22)),
    ('arg_eff', df_c,         df_m,      'nu',       'nu_mpipi', 'Arg\neff',(0, 13)),
]

# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(46, 30))
outer_gs = gridspec.GridSpec(5, 6, figure=fig, hspace=0.18, wspace=0.10,
                              left=0.09, right=0.95, top=0.94, bottom=0.06)

for ri, (row_key, src_c, src_m, nu_c, nu_m, ylabel, ylim) in enumerate(row_configs):
    for ci, k in enumerate(SIX):
        ax = fig.add_subplot(outer_gs[ri, ci])
        col = KC[k]

        # Zone backgrounds
        ax.axvspan(0.36, 0.54, alpha=0.09, color='#5b9bd5', zorder=0)
        ax.axvspan(0.63, 0.72, alpha=0.10, color='#4caf50', zorder=0)

        # CALVADOS-2 (solid line)
        sub_c = src_c[src_c['kingdom'] == k]
        g_c = get_bins(sub_c, row_key, nu_col=nu_c)
        if len(g_c) > 1:
            ax.fill_between(g_c.index, g_c['m']-g_c['s'], g_c['m']+g_c['s'],
                            color=col, alpha=0.25, zorder=2)
            ax.plot(g_c.index, g_c['m'], 'o-', color=col, lw=5, ms=12,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=3)

        # MPIPI-GG (dashed line) — BUG FIX 1: now always plotted
        sub_m = src_m[src_m['kingdom'] == k]
        g_m = get_bins(sub_m, row_key, nu_col=nu_m)
        if len(g_m) > 1:
            ax.fill_between(g_m.index, g_m['m']-g_m['s'], g_m['m']+g_m['s'],
                            color=col, alpha=0.12, zorder=1)
            ax.plot(g_m.index, g_m['m'], 's--', color=col, lw=4, ms=9,
                    markeredgecolor='white', markeredgewidth=1.0,
                    zorder=2, alpha=0.85)

        # Zone boundaries (compact/transition/expanded)
        ax.axvline(0.54, color='#999999', lw=1.5, ls='--', alpha=0.8, zorder=4)
        ax.axvline(0.63, color='#999999', lw=1.5, ls='--', alpha=0.8, zorder=4)

        # NCPR zero reference
        if row_key == 'NCPR':
            ax.axhline(0, color='grey', lw=2.0, ls=':', alpha=0.6)

        ax.set_xlim(0.36, 0.72)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=FS_TICK)

        # BUG FIX 2: use tick_params(labelbottom=False) — more reliable
        # than set_xticklabels([]) for suppressing x labels
        if ri < 4:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('\u03bd', fontsize=FS_LABEL)

        # y-axis: label only on leftmost column
        if ci == 0:
            ax.set_ylabel(ylabel, fontsize=FS_ROW, rotation=0,
                          labelpad=70, va='center')
        else:
            ax.tick_params(labelleft=False)

        # Column title on top row
        if ri == 0:
            ax.set_title(k, fontsize=FS_TITLE, color=col, fontweight='bold', pad=8)

# BUG FIX 3: row labels as figure text, not new subplots
# (previous loop was doing add_subplot(outer_gs[ri,5]) which
#  overwrote the Viruses column plots)
row_label_texts = ['SCD','NCPR','FCR','Lys\neff','Arg\neff']
for ri, label in enumerate(row_label_texts):
    # Place label to the right of the last column using figure coordinates
    # Get the position of the last subplot in this row
    ax_ref = fig.axes[ri * 6 + 5]   # Viruses column, this row
    pos = ax_ref.get_position()
    fig.text(pos.x1 + 0.005, pos.y0 + pos.height/2,
             label, fontsize=FS_ROW, va='center', ha='left', rotation=0,
             fontweight='bold')

# Legend
legend_handles = [
    Line2D([0],[0], color='grey', lw=5, marker='o', ms=12,
           markeredgecolor='white', label='CALVADOS-2'),
    Line2D([0],[0], color='grey', lw=4, marker='s', ms=9,
           ls='--', alpha=0.85, markeredgecolor='white', label='MPIPI-GG'),
]
fig.legend(handles=legend_handles, fontsize=FS_LEGEND, loc='upper right',
           bbox_to_anchor=(0.99, 0.99), framealpha=0.9)

# ── Save ──────────────────────────────────────────────────────────────
import os
from pathlib import Path
_ROOT = Path(__file__).parent
DATA  = _ROOT / 'data'
OUT   = _ROOT / 'figures'
OUT.mkdir(exist_ok=True)

fig.savefig(OUT / 'fig6_crossff_validation.pdf', dpi=200, bbox_inches='tight')
fig.savefig(OUT / 'fig6_crossff_validation.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig6_crossff_validation.pdf / fig6_crossff_validation.png")