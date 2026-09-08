"""
Figure 4 — Charge symmetry: the architecture operates on both charge signs

Inputs:
    run1_results.csv   — all 6 variants: R_to_K, K_to_R, shuffle_control,
                         E_to_A, D_to_A, ED_to_A
    BENDER_BIO.csv     — nu and kingdom per UniProt_ID
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
import warnings
from pathlib import Path
_ROOT = Path(__file__).parent
DATA  = _ROOT / 'data'
OUT   = _ROOT / 'figures'
OUT.mkdir(exist_ok=True)

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

FS_PANEL  = 70
FS_LABEL  = 58
FS_TICK   = 45
FS_LEGEND = 40
FS_ANNOT  = 40
FS_SMALL  = 38

# ── Data ─────────────────────────────────────────────────────────────
df  = pd.read_csv(DATA / 'substitution_results.csv')
bio = pd.read_csv(DATA / 'protein_metadata.csv')

bio_nu = bio.set_index('UniProt_ID')['nu'].to_dict()
bio_k  = bio.set_index('UniProt_ID')['kingdom'].to_dict()

df['nu_wt']    = df['UniProt_ID'].map(bio_nu)
df['kingdom']  = df['UniProt_ID'].map(bio_k)
df['delta_nu'] = df['nu'] - df['nu_wt']

def zone(nu):
    if pd.isna(nu): return None
    if nu < 0.54:   return 'below_threshold'
    if nu <= 0.63:  return 'near_threshold'
    return 'above_threshold'
df['nu_zone'] = df['nu_wt'].apply(zone)

def sig_str(pval):
    if pval < 0.001: return '***'
    if pval < 0.01:  return '**'
    if pval < 0.05:  return '*'
    return 'ns'

def lbl(ax, l):
    ax.text(0.02, 0.97, l, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', va='top')

np.random.seed(42)
fig = plt.figure(figsize=(60, 46))
gs  = gridspec.GridSpec(3, 6, figure=fig, hspace=0.45, wspace=0.55,
                         left=0.14, right=0.97, top=0.95, bottom=0.07)

ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])
ax_d = fig.add_subplot(gs[1, 0:2])
ax_e = fig.add_subplot(gs[1, 2:4])
ax_f = fig.add_subplot(gs[1, 4:6])
ax_g = fig.add_subplot(gs[2, 1:3])
ax_h = fig.add_subplot(gs[2, 3:5])

# ══ Panel a: forest plot — negative charge removal ════════════════════
ax = ax_a
variants_a = [
    ('E_to_A',          'Fungi',  'E Fungi',    '#2ca02c'),
    ('E_to_A',          'Plants', 'E Plants',   '#e377c2'),
    ('E_to_A',          None,     'E All',      '#4477aa'),
    ('D_to_A',          'Fungi',  'D Fungi',    '#2ca02c'),
    ('D_to_A',          'Plants', 'D Plants',   '#e377c2'),
    ('D_to_A',          None,     'D All',      '#4477aa'),
    ('ED_to_A',         'Fungi',  'ED Fungi',   '#2ca02c'),
    ('ED_to_A',         'Plants', 'ED Plants',  '#e377c2'),
    ('ED_to_A',         None,     'ED All',     '#4477aa'),
    ('shuffle_control', 'Fungi',  'Shu Fungi',  '#2ca02c'),
    ('shuffle_control', 'Plants', 'Shu Plants', '#e377c2'),
    ('shuffle_control', None,     'Shu All',    '#4477aa'),
]
for yi, (var, kingdom, label, col) in enumerate(variants_a):
    sub = df[df['variant'] == var].copy()
    if kingdom:
        sub = sub[sub['kingdom'] == kingdom]
    elif var in ['E_to_A', 'D_to_A', 'ED_to_A']:
        sub = sub[sub['kingdom'].isin(['Fungi', 'Plants'])]
    sub = sub['delta_nu'].dropna()
    if len(sub) < 3:
        continue
    m  = sub.mean()
    se = sub.sem()
    _, pv = stats.ttest_1samp(sub, 0)
    ci = 1.96 * se
    ax.errorbar(m, yi, xerr=ci, fmt='o', color=col, ms=25,
                capsize=12, lw=5, markeredgecolor='white', zorder=4)
    ax.text(0.98, yi, sig_str(pv), va='center', ha='right', fontsize=FS_SMALL,
            color=col, fontweight='bold',
            transform=ax.get_yaxis_transform())

ax.axvline(0, color='grey', ls='--', lw=5, alpha=0.7)
ax.set_yticks(range(len(variants_a)))
ax.set_yticklabels([v[2] for v in variants_a], fontsize=FS_SMALL)
ax.set_xlabel('$\\mathrm{\\Delta\\nu}$ (variant − wt)', fontsize=FS_LABEL)
ax.tick_params(axis='x', labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'a')

# ══ Panel b: positive charge substitution bars ════════════════════════
ax = ax_b
zone_keys  = ['below_threshold', 'near_threshold', 'above_threshold']
zone_xlbls = ['Compact', 'Transition', 'Expanded']
for vi, (var, col, label) in enumerate([
        ('R_to_K', '#e07040', 'R→K'),
        ('K_to_R', '#5b8db8', 'K→R')]):
    means2, sems2 = [], []
    for zk in zone_keys:
        sub = df[(df['variant'] == var) & (df['nu_zone'] == zk)]['delta_nu'].dropna()
        means2.append(sub.mean() if len(sub) > 0 else 0)
        sems2.append(sub.sem()   if len(sub) > 0 else 0)
    x = np.arange(3) + (vi - 0.5) * 0.35
    ax.bar(x, means2, 0.32, color=col, alpha=0.85,
           yerr=[1.96 * s for s in sems2], capsize=10,
           error_kw={'lw': 5}, label=label, zorder=3)
ax.axhline(0, color='grey', ls='--', lw=5)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(zone_xlbls, fontsize=FS_TICK)
ax.set_ylabel('$\\mathrm{\\Delta\\nu}$', fontsize=FS_LABEL)
ax.tick_params(axis='y', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, framealpha=0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'b')

# ══ Panel c: carrier-specific vs composition ══════════════════════════
ax = ax_c
groups = [
    (df[(df['variant'] == 'E_to_A') &
        (df['kingdom'].isin(['Fungi', 'Plants']))]['delta_nu'].dropna(),
     'E→\nNeg.', '#5b8db8'),
    (df[(df['variant'] == 'shuffle_control') &
        (df['kingdom'].isin(['Fungi', 'Plants']))]['delta_nu'].dropna(),
     'Shuf', '#aaaaaa'),
    (df[df['variant'] == 'R_to_K']['delta_nu'].dropna(),
     'R→K\nPos.', '#e07040'),
    (df[df['variant'] == 'shuffle_control']['delta_nu'].dropna(),
     'Shuf', '#aaaaaa'),
]
for xi, (sub, label, col) in enumerate(groups):
    if len(sub) < 2:
        continue
    parts = ax.violinplot(sub, positions=[xi], widths=0.6,
                          showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(col)
        pc.set_alpha(0.45)
    jit = np.random.uniform(-0.12, 0.12, len(sub))
    ax.scatter(xi + jit, sub.clip(-0.18, 0.35), color=col, alpha=0.25, s=110, zorder=2)
    ax.errorbar(xi, sub.mean(), yerr=1.96 * sub.sem(), fmt='o', color='white',
                ms=28, capsize=12, lw=5, markeredgecolor='black', zorder=6)
ax.axhline(0, color='grey', ls='--', lw=5)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[1] for g in groups], fontsize=FS_TICK)
ax.set_ylabel('$\\mathrm{\\Delta\\nu}$ (variant − wt)', fontsize=FS_LABEL)
ax.tick_params(axis='y', labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'c')

# ══ Panel d: individual sequences Fungi+Plants ════════════════════════
ax = ax_d
variants_d = ['E_to_A', 'D_to_A', 'ED_to_A', 'shuffle_control']
labels_d   = ['E→A', 'D→A', 'ED→A', 'Shuffle']
for xi, var in enumerate(variants_d):
    for k, col in [('Fungi', '#2ca02c'), ('Plants', '#e377c2')]:
        sub = df[(df['variant'] == var) & (df['kingdom'] == k)]['delta_nu'].dropna()
        jit = np.random.uniform(-0.12, 0.12, len(sub))
        ax.scatter(xi + jit, sub.values, color=col, alpha=0.5, s=125, zorder=2)
    sub_all = df[(df['variant'] == var) &
                 (df['kingdom'].isin(['Fungi', 'Plants']))]['delta_nu'].dropna()
    if len(sub_all) > 0:
        ax.errorbar(xi, sub_all.mean(), yerr=1.96 * sub_all.sem(), fmt='s',
                    color='black', ms=28, capsize=12, lw=5,
                    zorder=5, markeredgecolor='white')
ax.axhline(0, color='grey', ls='--', lw=5)
ax.set_xticks(range(len(labels_d)))
ax.set_xticklabels(labels_d, fontsize=FS_TICK)
ax.set_ylabel('$\\mathrm{\\Delta\\nu}$ (variant − wt)', fontsize=FS_LABEL)
ax.tick_params(axis='y', labelsize=FS_TICK)
ax.legend(handles=[Patch(color='#2ca02c', label='Fungi'),
                   Patch(color='#e377c2', label='Plants')],
          fontsize=FS_LEGEND, loc='upper right', framealpha=0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'd')

# ══ Panel e: compaction by nu tertile ════════════════════════════════
ax = ax_e
ed_sub = df[(df['variant'] == 'ED_to_A') &
            (df['kingdom'].isin(['Fungi', 'Plants']))].copy()
ed_sub = ed_sub.dropna(subset=['delta_nu', 'nu_wt'])
q33 = ed_sub['nu_wt'].quantile(0.33)
q67 = ed_sub['nu_wt'].quantile(0.67)
def tertile(nu):
    if pd.isna(nu): return None
    if nu <= q33:   return 'Lower $\\mathrm{\\nu}$'
    if nu <= q67:   return 'Mid $\\mathrm{\\nu}$'
    return 'Upper $\\mathrm{\\nu}$'
ed_sub['tertile'] = ed_sub['nu_wt'].apply(tertile)

tert_order = ['Lower $\\mathrm{\\nu}$', 'Mid $\\mathrm{\\nu}$', 'Upper $\\mathrm{\\nu}$']
means_t, xi_t = [], []
for xi, t in enumerate(tert_order):
    sub_t = ed_sub[ed_sub['tertile'] == t]['delta_nu'].dropna()
    if len(sub_t) < 2:
        continue
    parts = ax.violinplot(sub_t, positions=[xi], widths=0.6,
                          showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#5b8db8')
        pc.set_alpha(0.35)
    for k, col in [('Fungi', '#2ca02c'), ('Plants', '#e377c2')]:
        sub_k = ed_sub[(ed_sub['tertile'] == t) & (ed_sub['kingdom'] == k)]['delta_nu'].dropna()
        jit = np.random.uniform(-0.12, 0.12, len(sub_k))
        ax.scatter(xi + jit, sub_k.values, color=col, alpha=0.55, s=125, zorder=2)
    ax.errorbar(xi, sub_t.mean(), yerr=1.96 * sub_t.sem(), fmt='o',
                color='white', ms=22, capsize=10, lw=4,
                markeredgecolor='black', zorder=6)
    means_t.append(sub_t.mean())
    xi_t.append(xi)

if len(means_t) > 1:
    ax.plot(xi_t, means_t, 'k--', lw=7, alpha=0.6, zorder=3)
ax.axhline(0, color='grey', ls='--', lw=5)
ax.set_xticks(range(3))
ax.set_xticklabels(tert_order, fontsize=FS_TICK)
ax.set_ylabel('$\\mathrm{\\Delta\\nu}$ (E→A − wt)', fontsize=FS_LABEL)
ax.tick_params(axis='y', labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'e')

# ══ Panel f: Bland-Altman additivity ══════════════════════════════════
ax = ax_f
e_idx  = df[df['variant'] == 'E_to_A'].set_index('UniProt_ID')['delta_nu']
d_idx  = df[df['variant'] == 'D_to_A'].set_index('UniProt_ID')['delta_nu']
ed_idx = df[df['variant'] == 'ED_to_A'].set_index('UniProt_ID')['delta_nu']
common = sorted(set(e_idx.index) & set(d_idx.index) & set(ed_idx.index))

if len(common) >= 3:
    pred   = e_idx[common] + d_idx[common]
    meas   = ed_idx[common]
    diff   = meas - pred
    avg    = (meas + pred) / 2
    bias   = diff.mean()
    loa_lo = bias - 1.96 * diff.std()
    loa_hi = bias + 1.96 * diff.std()

    kingdom_map = df[df['variant'] == 'ED_to_A'].set_index('UniProt_ID')['kingdom'].to_dict()
    for k, col in [('Fungi', '#2ca02c'), ('Plants', '#e377c2')]:
        mask = [kingdom_map.get(uid) == k for uid in common]
        ax.scatter(avg[mask], diff[mask], color=col, alpha=0.7,
                   s=130, zorder=3, label=k)

    ax.axhline(0,      color='grey',      ls=':', lw=5)
    ax.axhline(bias,   color='steelblue', lw=7,   label='Mean bias')
    ax.axhline(loa_lo, color='lightblue', lw=5,   ls='--')
    ax.axhline(loa_hi, color='lightblue', lw=5,   ls='--')
    ax.text(avg.min() + 0.005, bias + 0.005,
            f'Mean bias: +{bias:.4f}\n95% LOA: [{loa_lo:.3f}, +{loa_hi:.3f}]',
            fontsize=FS_SMALL, color='steelblue',
            bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    ax.set_xlabel('Mean of predicted and measured $\\mathrm{\\Delta\\nu}$', fontsize=FS_LABEL)
    ax.set_ylabel('Measured − Predicted', fontsize=FS_LABEL)
    ax.legend(handles=[Patch(color='#2ca02c', label='Fungi'),
                       Patch(color='#e377c2', label='Plants')],
              fontsize=FS_LEGEND, loc='upper right', framealpha=0.85)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'f')

# ══ Panel g: symmetry violin |$\\mathrm{\\Delta\\nu}$| ════════════════════════════════════
ax = ax_g
for xi, (var, col) in enumerate([('ED_to_A', '#5b8db8'), ('R_to_K', '#e07040')]):
    if var == 'R_to_K':
        sub = df[(df['variant'] == var) &
                 (df['kingdom'].isin(['Fungi', 'Plants']))]['delta_nu'].dropna().abs()
    else:
        sub = df[df['variant'] == var]['delta_nu'].dropna().abs()
    if len(sub) < 2:
        continue
    parts = ax.violinplot(sub, positions=[xi], widths=0.6,
                          showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(col)
        pc.set_alpha(0.45)
    jit = np.random.uniform(-0.12, 0.12, len(sub))
    ax.scatter(xi + jit, sub.values, color=col, alpha=0.3, s=110, zorder=2)
    ax.errorbar(xi, sub.mean(), yerr=1.96 * sub.sem(), fmt='o', color='white',
                ms=30, capsize=12, lw=5, markeredgecolor='black', zorder=6)
    ax.text(xi, sub.mean() + 0.007, f'{sub.mean():.3f}', ha='center',
            fontsize=FS_SMALL, fontweight='bold', color=col)

ax.set_xticks([0, 1])
ax.set_xticklabels(['ED→A\n(neg. charge)', 'R→K\n(pos. charge)'], fontsize=FS_TICK)
ax.set_ylabel('|$\\mathrm{\\Delta\\nu}$| (effect size)', fontsize=FS_LABEL)
ax.tick_params(axis='y', labelsize=FS_TICK)
ax.set_ylim(0, 0.22)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'g')

# ══ Panel h: taxon-specific compaction ════════════════════════════════
ax = ax_h
for xi, (k, col) in enumerate([('Fungi', '#2ca02c'), ('Plants', '#e377c2')]):
    sub = df[(df['variant'] == 'ED_to_A') & (df['kingdom'] == k)]['delta_nu'].dropna()
    if len(sub) < 2:
        continue
    parts = ax.violinplot(sub, positions=[xi], widths=0.6,
                          showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(col)
        pc.set_alpha(0.5)
    jit = np.random.uniform(-0.15, 0.15, len(sub))
    ax.scatter(xi + jit, sub.values, color=col, alpha=0.45, s=125, zorder=2)
    ax.errorbar(xi, sub.mean(), yerr=1.96 * sub.sem(), fmt='o', color='white',
                ms=30, capsize=12, lw=5, markeredgecolor='black', zorder=6)
    _, pv = stats.ttest_1samp(sub, 0)
    ax.text(xi, -0.15,
            f'n={len(sub)}\n' + '$\\mathrm{\\Delta\\nu}$=' + f'{sub.mean():.3f}\np={sig_str(pv)}',
            ha='center', fontsize=FS_SMALL, color=col, va='top')
ax.axhline(0, color='grey', ls='--', lw=5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Fungi', 'Plants'], fontsize=FS_TICK)
ax.set_ylabel('$\\mathrm{\\Delta\\nu}$ (ED→A − wildtype)', fontsize=FS_LABEL)
ax.tick_params(axis='y', labelsize=FS_TICK)
ax.set_ylim(-0.25, 0.09)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'h')

# ── Save ─────────────────────────────────────────────────────────────
fig.savefig(OUT / 'fig4_charge_symmetry.pdf', dpi=200, bbox_inches='tight')
fig.savefig(OUT / 'fig4_charge_symmetry.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig4_charge_symmetry.pdf / fig4_charge_symmetry.png")
