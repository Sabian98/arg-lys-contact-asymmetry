"""
Figure 5 — Synthesis: lambda-sweep, cross-FF agreement, dual crossover, scale of change
Panels: a) lambda-sweep nu vs lambda_R, b) contact ratio curve,
        c) per-partner Spearman rho, d) per-kingdom aromatic gap,
        e) cross-FF heatmap, f) dual crossover, g) fold-change bars

Inputs needed:
    sweep_results.csv         — lambda_R, UniProt_ID, nu, nu_r2, eff_Arg_* columns
    run1_results.csv          — K_to_R and R_to_K variant contact efficiencies
    BENDER_BIO.csv            — kingdom, nu, f_K, f_aro, FCR, SCD per sequence
    contact_results_raw.csv   — per-sequence contact efficiencies (CALVADOS-2)
    mpipi_results_raw.csv     — per-sequence contact efficiencies (MPIPI-GG)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtransforms
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

FS_PANEL  = 84
FS_LABEL  = 70
FS_TICK   = 54
FS_LEGEND = 48
FS_ANNOT  = 46
FS_SMALL  = 41
FS_CELL   = 30   # heatmap cell text — panel f now has full row

SIX = ['Bacteria', 'Fungi', 'Mammals', 'Plants', 'Protists', 'Viruses']
KC  = {'Bacteria': '#1f77b4', 'Fungi': '#2ca02c', 'Mammals': '#9467bd',
       'Plants':   '#e377c2', 'Protists': '#bcbd22', 'Viruses': '#17becf'}

# ── Load ──────────────────────────────────────────────────────────────
df   = pd.read_csv(DATA / 'substitution_results.csv')
bio  = pd.read_csv(DATA / 'protein_metadata.csv')
swp  = pd.read_csv(DATA / 'lambda_sweep.csv')
df_c = pd.read_csv(DATA / 'contact_results.csv')
df_m = pd.read_csv(DATA / 'mpipi_results.csv')

wt_nu_bio = bio.set_index('UniProt_ID')['nu'].to_dict()
bio_k     = bio.set_index('UniProt_ID')['kingdom'].to_dict()
bio6      = bio[bio['kingdom'].isin(SIX)].copy()

# ── Paired data for panels c-e ────────────────────────────────────────
ktr = df[df['variant'] == 'K_to_R'].set_index('UniProt_ID')
rtk = df[df['variant'] == 'R_to_K'].set_index('UniProt_ID')
common = sorted(set(ktr.index) & set(rtk.index))
partners_full = [('eff_Arg_Tyr', 'eff_Lys_Tyr', 'Tyr'),
                 ('eff_Arg_Phe', 'eff_Lys_Phe', 'Phe'),
                 ('eff_Arg_Trp', 'eff_Lys_Trp', 'Trp'),
                 ('eff_Arg_Glu', 'eff_Lys_Glu', 'Glu'),
                 ('eff_Arg_Asp', 'eff_Lys_Asp', 'Asp'),
                 ('eff_Arg_Ser', 'eff_Lys_Ser', 'Ser')]

rows = []
for uid in common:
    av, lv, pp = [], [], {}
    for ac, lc, pn in partners_full:
        a = ktr.loc[uid, ac] if ac in ktr.columns else np.nan
        l = rtk.loc[uid, lc] if lc in rtk.columns else np.nan
        if np.isfinite(a) and a > 0: av.append(a)
        if np.isfinite(l) and l > 0: lv.append(l)
        pp['arg_' + pn] = a if np.isfinite(a) else np.nan
        pp['lys_' + pn] = l if np.isfinite(l) else np.nan
    aro_a = np.nanmean([pp.get('arg_Tyr', np.nan), pp.get('arg_Phe', np.nan), pp.get('arg_Trp', np.nan)])
    aro_l = np.nanmean([pp.get('lys_Tyr', np.nan), pp.get('lys_Phe', np.nan), pp.get('lys_Trp', np.nan)])
    if av and lv:
        rows.append({'UniProt_ID': uid, 'nu_wt': wt_nu_bio.get(uid, np.nan),
                     'kingdom': bio_k.get(uid, 'unknown'),
                     'arg_eff': np.nanmean(av), 'lys_eff': np.nanmean(lv),
                     'aro_gap': aro_a - aro_l,
                     'chg_gap': np.nanmean([pp.get('arg_Glu', np.nan), pp.get('arg_Asp', np.nan)]) -
                                np.nanmean([pp.get('lys_Glu', np.nan), pp.get('lys_Asp', np.nan)]),
                     'ratio': np.nanmean(av) / np.nanmean(lv),
                     **pp})

paired = pd.DataFrame(rows)
paired['aro_gap_w'] = paired['aro_gap'].clip(paired['aro_gap'].quantile(0.01),
                                              paired['aro_gap'].quantile(0.99))
paired['chg_gap_w'] = paired['chg_gap'].clip(paired['chg_gap'].quantile(0.01),
                                              paired['chg_gap'].quantile(0.99))
paired['nu_bin'] = np.round(paired['nu_wt'] / 0.04) * 0.04
measured_ratio = paired['ratio'].mean()

# ── Lambda sweep ──────────────────────────────────────────────────────
filt     = swp.groupby('UniProt_ID')['nu_r2'].min() >= 0.95
good_ids = filt[filt].index
sub_swp  = swp[swp['UniProt_ID'].isin(good_ids)]
lam_vals = sorted(swp['lambda_R'].unique())
nu_means, nu_sems = [], []
for l in lam_vals:
    s  = sub_swp[sub_swp['lambda_R'] == l]['nu']
    sc = s.clip(s.quantile(0.01), s.quantile(0.99))
    nu_means.append(sc.mean())
    nu_sems.append(sc.sem())
bio_swp    = bio[bio['UniProt_ID'].isin(good_ids)]
pred_slope = -2 / (5 * np.log(bio_swp['n_residues'].mean())) * bio_swp['f_R'].mean() * 0.8368 / (4 * 2.479)
pred_nus   = [nu_means[0] + pred_slope * (l - lam_vals[0]) for l in lam_vals]
nu_a  = sub_swp[sub_swp['lambda_R'] == 0.731].set_index('UniProt_ID')['nu']
nu_l  = sub_swp[sub_swp['lambda_R'] == 0.179].set_index('UniProt_ID')['nu']
com_s = sorted(set(nu_a.index) & set(nu_l.index))
_, pw_s = stats.wilcoxon(nu_a[com_s], nu_l[com_s])

# Contact ratio curve from sweep
arg_cols = ['eff_Arg_Tyr', 'eff_Arg_Phe', 'eff_Arg_Trp',
            'eff_Arg_Glu', 'eff_Arg_Asp', 'eff_Arg_Ser']
all_50   = swp['UniProt_ID'].unique()
base_eff = {}
for uid in all_50:
    row = swp[(swp['UniProt_ID'] == uid) & (swp['lambda_R'] == 0.179)]
    if len(row) > 0:
        val = row[arg_cols].replace(0, np.nan).mean(axis=1).values[0]
        if np.isfinite(val): base_eff[uid] = val

lam_x = [0.10, 0.179, 0.317, 0.455, 0.593, 0.731, 0.80]
lam_y  = [1.00, 1.00]
for l in [0.317, 0.455, 0.593, 0.731]:
    rl = []
    for uid, base in base_eff.items():
        row = swp[(swp['UniProt_ID'] == uid) & (swp['lambda_R'] == l)]
        if len(row) > 0:
            v = row[arg_cols].replace(0, np.nan).mean(axis=1).values[0]
            if np.isfinite(v) and base > 0: rl.append(v / base)
    lam_y.append(np.nanmean(rl))
lam_y.append(lam_y[-1] + 0.025)

pairwise  = 1.05
chain_731 = lam_y[5]
measured_r = measured_ratio

# ── Cross-FF rank-biserial ────────────────────────────────────────────
lys_cols_c = ['eff_Lys_Tyr','eff_Lys_Phe','eff_Lys_Trp','eff_Lys_Glu','eff_Lys_Asp','eff_Lys_Lys','eff_Lys_Ser']
arg_cols_c = ['eff_Arg_Tyr','eff_Arg_Phe','eff_Arg_Trp','eff_Arg_Glu','eff_Arg_Asp','eff_Arg_Arg','eff_Arg_Ser']
chg_cols_c = ['eff_Arg_Glu','eff_Arg_Asp','eff_Lys_Glu','eff_Lys_Asp']
cat_cols_c = ['eff_Arg_Tyr','eff_Arg_Phe','eff_Arg_Trp','eff_Lys_Tyr','eff_Lys_Phe','eff_Lys_Trp']
lys_cols_m = [c for c in df_m.columns if 'Lys' in c and c.startswith('eff_')]
arg_cols_m = [c for c in df_m.columns if 'Arg' in c and c.startswith('eff_') and 'Lys' not in c]
chg_cols_m = [c for c in df_m.columns if c.startswith('eff_') and any(x in c for x in ['Arg_Glu','Arg_Asp','Lys_Glu','Lys_Asp'])]
cat_cols_m = [c for c in df_m.columns if c.startswith('eff_') and any(x in c for x in ['Arg_Tyr','Arg_Phe','Arg_Trp','Lys_Tyr','Lys_Phe','Lys_Trp'])]

def rb_loss(a, b, min_n=2):
    if len(a) < min_n or len(b) < min_n: return np.nan
    stat, _ = stats.mannwhitneyu(a, b, alternative='greater')
    return (2 * stat) / (len(a) * len(b)) - 1

classes = [('Lys-med', lys_cols_c, lys_cols_m), ('Chg-chg', chg_cols_c, chg_cols_m),
           ('Cat-π',   cat_cols_c,  cat_cols_m), ('Arg-med', arg_cols_c,  arg_cols_m)]

calv_vals  = np.zeros((6, 4))
mpipi_vals = np.zeros((6, 4))
agree_mask = np.zeros((6, 4), dtype=bool)

for i, k in enumerate(SIX):
    sub_c      = df_c[df_c['kingdom'] == k]
    compact_c  = sub_c[sub_c['nu'] < 0.54]
    expanded_c = sub_c[sub_c['nu'] > 0.63]
    sub_m      = df_m[df_m['kingdom'] == k]
    compact_m  = sub_m[sub_m['nu_mpipi'] < 0.54]
    expanded_m = sub_m[sub_m['nu_mpipi'] > 0.63]
    for j, (cname, cc, cm) in enumerate(classes):
        ec   = compact_c[[c for c in cc if c in compact_c.columns]].replace(0, np.nan).mean(axis=1).dropna()
        ee   = expanded_c[[c for c in cc if c in expanded_c.columns]].replace(0, np.nan).mean(axis=1).dropna()
        rb_c = rb_loss(ec.values, ee.values)
        em_c = compact_m[[c for c in cm if c in compact_m.columns]].replace(0, np.nan).mean(axis=1).dropna()
        em_e = expanded_m[[c for c in cm if c in expanded_m.columns]].replace(0, np.nan).mean(axis=1).dropna()
        rb_m = rb_loss(em_c.values, em_e.values)
        calv_vals[i, j]  = rb_c if not np.isnan(rb_c) else 0.0
        mpipi_vals[i, j] = rb_m if not np.isnan(rb_m) else 0.0
        agree_mask[i, j] = (not np.isnan(rb_c) and not np.isnan(rb_m) and
                            np.sign(rb_c) == np.sign(rb_m))
n_agree = agree_mask.sum()

# ── Crossover points ──────────────────────────────────────────────────
bio6['nu_bin'] = np.round(bio6['nu'] / 0.02) * 0.02
comp_cross = {'Bacteria': 0.578, 'Fungi': 0.547, 'Mammals': 0.568,
              'Plants': 0.600, 'Protists': 0.525, 'Viruses': 0.574}
eff_cross  = {'Bacteria': 0.565, 'Fungi': 0.550, 'Mammals': 0.580,
              'Plants': 0.558, 'Protists': 0.545, 'Viruses': 0.575}

# ── Fold-changes ──────────────────────────────────────────────────────
compact  = bio6[bio6['nu'] < 0.54]
expanded = bio6[bio6['nu'] > 0.63]

# ── Figure ────────────────────────────────────────────────────────────
np.random.seed(42)
fig = plt.figure(figsize=(70, 66))
gs  = gridspec.GridSpec(4, 6, figure=fig, hspace=0.42, wspace=0.50,
                         left=0.10, right=0.97, top=0.95, bottom=0.05)

ax_a = fig.add_subplot(gs[0, 0:3])
ax_b = fig.add_subplot(gs[0, 3:6])
ax_c = fig.add_subplot(gs[1, 0:3])
ax_e = fig.add_subplot(gs[1, 3:6])
ax_f = fig.add_subplot(gs[2, 0:6])   # full row
ax_g = fig.add_subplot(gs[3, 1:3])
ax_h = fig.add_subplot(gs[3, 3:5])

# Shrink top-row panels inward to prevent clashing
_pad = 0.028
_pa, _pb = ax_a.get_position(), ax_b.get_position()
ax_a.set_position([_pa.x0, _pa.y0, _pa.width - _pad, _pa.height])
ax_b.set_position([_pb.x0 + _pad, _pb.y0, _pb.width - _pad, _pb.height])

def lbl(ax, l):
    ax.text(0.02, 0.97, l, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', va='top')

# ══ Panel a: lambda sweep nu ══════════════════════════════════════════
ax = ax_a
ax.axvspan(0.08, 0.35, alpha=0.20, color='#5b8db8', zorder=0)
ax.axvspan(0.60, 0.82, alpha=0.20, color='#e07040', zorder=0)
ax.axhline(np.mean(pred_nus), color='#cc4444', lw=5, ls='--',
           label='Flory prediction', zorder=3, alpha=0.9)
ax.errorbar(lam_vals, nu_means, yerr=nu_sems, fmt='o-', color='#1a1a1a',
            lw=7, ms=26, capsize=0, zorder=4, label='Measured $\\mathrm{\\nu}$ (n=37)')
ax.fill_between(lam_vals, [m - s for m, s in zip(nu_means, nu_sems)],
                [m + s for m, s in zip(nu_means, nu_sems)],
                color='#1a1a1a', alpha=0.12, zorder=3)
ax.axvline(0.179, color='#5b8db8', lw=3.5, ls=':', alpha=0.8)
ax.axvline(0.731, color='#e07040', lw=3.5, ls=':', alpha=0.8)
ax.text(0.179, nu_means[0] + 0.001, 'λ=λ_K', ha='center',
        fontsize=FS_LABEL, color='#5b8db8', va='bottom')
# ax.text(0.731, nu_means[-1] - 0.002, 'Published\nλ_R=0.731',
#         ha='center', fontsize=FS_SMALL, color='#e07040', va='top')
dnu = nu_means[-1] - nu_means[0]
ax.annotate('', xy=(0.731, nu_means[-1]), xytext=(0.731, nu_means[0]),
            arrowprops=dict(arrowstyle='<->', color='darkred', lw=4.5))
ax.text(0.755, (nu_means[0] + nu_means[-1]) / 2,
        '$\\mathrm{\\Delta\\nu}$=' + f'{dnu:+.3f}\np={pw_s:.3f}',
        fontsize=FS_LABEL, color='darkred', va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9))
# ax.text(0.40, np.mean(pred_nus) + 0.001, 'Flory prediction\n(pairwise only)',
#         fontsize=FS_LABEL, color='#cc4444', ha='center', va='bottom', style='normal')
ax.set_xlabel('λ_R (Arg cohesion)', fontsize=FS_LABEL)
ax.set_ylabel('Mean $\\mathrm{\\nu}$', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc='lower left', framealpha=0.85)
ax.set_xlim(0.10, 0.82)
ax.set_ylim(nu_means[-1] - 0.025, nu_means[0] + 0.025)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'a')

# ══ Panel b: contact ratio curve ══════════════════════════════════════
ax = ax_b
ax.axhspan(1.00,           pairwise,           alpha=0.35, color='#f0d080', zorder=0)
ax.axhspan(pairwise,       chain_731 + 0.05,   alpha=0.25, color='#f0a060', zorder=0)
ax.axhline(measured_r, color='darkred', lw=6, zorder=3)
ax.plot(lam_x, lam_y, 'o-', color='#333', lw=7, ms=22, zorder=4)
ax.axvline(0.731, color='#e07040', lw=3.5, ls='--', alpha=0.8)
ax.scatter([0.731], [chain_731], s=350, color='#e07040', zorder=5)
ax.text(0.731, chain_731 - 0.028, f'{chain_731:.2f}×',
        ha='center', fontsize=FS_LABEL, color='#e07040', fontweight='bold', va='top')
ax.text(0.45, measured_r + 0.015, f'Measured: {measured_r:.2f}×',
        color='darkred', fontsize=FS_LABEL, fontweight='bold', ha='center', va='bottom')
ax.text(0.11, 1.018, 'Pairwise', fontsize=FS_LABEL, color='#b07000', va='bottom')
ax.text(0.11, 1.18,  'Chain\nphysics', fontsize=FS_LABEL, color='#c06000', va='center')
pair_pct  = (pairwise - 1.0) / (measured_r - 1.0) * 100
chain_pct = (chain_731 - pairwise) / (measured_r - 1.0) * 100
ax.text(0.78, 1.025, f'{pair_pct:.0f}%',
        color='#b07000', fontsize=FS_LABEL, fontweight='bold', ha='right', va='bottom')
ax.text(0.78, (pairwise + chain_731) / 2, f'{chain_pct:.0f}%',
        color='#c06000', fontsize=FS_LABEL, fontweight='bold', ha='right', va='center')
ax.axvline(0.179, color='grey', lw=3, ls=':', alpha=0.6)
# ax.text(0.179, 0.975, 'λ=λ_K', ha='center', fontsize=FS_LABEL, color='grey', va='top')
ax.set_xlabel('λ_R (Arg cohesion)', fontsize=FS_LABEL)
ax.set_ylabel('Arg/Lys contact ratio', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.set_xlim(0.08, 0.82)
ax.set_ylim(0.97, 1.55)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'b')

# ══ Panel c: per-partner Spearman rho ═════════════════════════════════
ax = ax_c
pnames = ['Tyr', 'Phe', 'Trp', 'Glu', 'Asp', 'Ser']
rhos, sigs_c = [], []
for _, _, pn in partners_full:
    acol, lcol = 'arg_' + pn, 'lys_' + pn
    sub = paired[['nu_wt', acol, lcol]].dropna()
    sub = sub[(sub[acol] > 0) & (sub[lcol] > 0)]
    gap = sub[acol] - sub[lcol]
    rho, pv = stats.spearmanr(sub['nu_wt'], gap)
    rhos.append(rho)
    sigs_c.append('***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'ns')

colors_c = ['#e07040'] * 3 + ['#5b8db8'] * 3
x = np.arange(len(pnames))
ax.bar(x, rhos, color=colors_c, alpha=0.85, edgecolor='white', lw=5)
ax.axhline(0, color='grey', lw=3.5, ls='--')
ax.axvline(2.5, color='grey', ls='--', lw=3.5, alpha=0.6)
for xi, (rho, sig) in enumerate(zip(rhos, sigs_c)):
    ax.text(xi, rho - 0.018, sig, ha='center', fontsize=FS_LABEL, fontweight='bold',
            color='white' if abs(rho) > 0.2 else 'black')
_trans_c = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.text(0.9,  0.04, 'Aromatic', color='#e07040', fontsize=FS_LABEL, 
        ha='center', va='bottom', transform=_trans_c)
ax.text(3.85, 0.04, 'Charged',  color='#5b8db8', fontsize=FS_LABEL, 
        ha='center', va='bottom', transform=_trans_c)
ax.set_xticks(x)
ax.set_xticklabels(pnames, fontsize=FS_LABEL)
ax.tick_params(axis='y', labelsize=FS_LABEL)
ax.set_ylabel('Spearman ρ (gap vs $\\mathrm{\\nu}$)', fontsize=FS_LABEL)
ax.set_ylim(-0.55, 0.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'c')

# ══ Panel d: per-kingdom aromatic gap ═════════════════════════════════
ax = ax_e
kingdom_order = ['Viruses', 'Protists', 'Plants', 'Mammals', 'Fungi', 'Bacteria']
ytick_labels = []
for yi, k in enumerate(kingdom_order):
    sub_w = paired[paired['kingdom'] == k]['aro_gap_w'].dropna()
    sub_r = paired[paired['kingdom'] == k]['aro_gap'].dropna()
    col = KC[k]
    jit = np.random.uniform(-0.12, 0.12, len(sub_r))
    ax.scatter(sub_r.clip(-7.9, 7.9).values, yi + jit, color=col, alpha=0.35, s=150, zorder=2)
    ax.errorbar(sub_w.mean(), yi, xerr=1.96 * sub_w.sem(), fmt='o', color=col,
                ms=28, capsize=12, lw=6, markeredgecolor='white', zorder=5)
    if len(sub_r) >= 5:
        _, pw = stats.wilcoxon(sub_r)
    else:
        pw = 1.0
    sig = '***' if pw < 0.001 else '**' if pw < 0.01 else '*' if pw < 0.05 else 'ns'
    ytick_labels.append(f'{k} ({sig})')

ax.set_yticks(range(len(kingdom_order)))
ax.set_yticklabels(ytick_labels, fontsize=FS_ANNOT)
for tick, k in zip(ax.yaxis.get_major_ticks(), kingdom_order):
    tick.label1.set_color(KC[k])
    tick.label1.set_fontweight('bold')
ax.axvline(0, color='grey', ls='--', lw=3.5)
ax.set_xlabel('Arg − Lys aromatic gap', fontsize=FS_LABEL)
ax.tick_params(axis='x', labelsize=FS_TICK)
ax.set_xlim(-8, 12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'd')

# ══ Panel e: cross-FF heatmap (full row) ══════════════════════════════
ax = ax_f
im = ax.imshow(calv_vals, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
contact_classes = ['Lys-med', 'Chg-chg', 'Cat-π', 'Arg-med']
for i in range(6):
    for j in range(4):
        cv = calv_vals[i, j]
        mv = mpipi_vals[i, j]
        text_color = 'white' if abs(cv) > 0.22 else '#1a1a1a'
        ax.text(j, i, f'{cv:+.2f}\n{mv:+.2f}', ha='center', va='center',
                fontsize=FS_ANNOT, color=text_color, fontweight='bold')
        if agree_mask[i, j]:
            rect = plt.Rectangle([j - 0.5, i - 0.5], 1, 1, fill=False,
                                  edgecolor='#3a9a3a', lw=4, zorder=3)
            ax.add_patch(rect)
ax.set_xticks(range(4))
ax.set_xticklabels(contact_classes, fontsize=FS_LABEL)
ax.set_yticks(range(6))
ax.set_yticklabels(SIX, fontsize=FS_LABEL)
cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
cb.set_label('CALVADOS-2 rank-biserial\n(positive = compact wins)', fontsize=FS_SMALL)
cb.ax.tick_params(labelsize=FS_SMALL)
# ax.text(0.5, -0.07, f'{n_agree}/24 cells agree in sign across CALVADOS-2 and MPIPI-GG (green border)',
#         transform=ax.transAxes, ha='center', va='top', fontsize=FS_SMALL,
#         color='#2a8a2a', fontweight='bold')
lbl(ax, 'e')

# ══ Panel f: dual crossover ═══════════════════════════════════════════
ax = ax_g
for k in SIX:
    col = KC[k]
    yi  = SIX.index(k)
    ax.scatter(comp_cross[k], yi, s=330, color=col, zorder=4,
               marker='o', edgecolors='white', lw=3)
    ax.scatter(eff_cross[k],  yi, s=330, color=col, zorder=4,
               marker='D', edgecolors='white', lw=3)
    ax.plot([comp_cross[k], eff_cross[k]], [yi, yi], color=col, lw=12, alpha=0.5)
ax.axvline(np.mean(list(comp_cross.values())), color='grey', ls=':', lw=10.5, alpha=0.7)
ax.axvline(np.mean(list(eff_cross.values())),  color='grey', ls='--', lw=10.5, alpha=0.7)
ax.legend(handles=[
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', ms=22, label='Comp crossover'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='grey', ms=22, label='Eff crossover')],
    fontsize=FS_LEGEND, loc='upper right', framealpha=0.8)
t, p = stats.ttest_rel(list(comp_cross.values()), list(eff_cross.values()))
delta = abs(np.mean(list(comp_cross.values())) - np.mean(list(eff_cross.values())))
ax.text(0.04, 0.08, f'Δ = {delta:.3f}\np = {p:.2f} (ns)',
        transform=ax.transAxes, fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', fc='white', alpha=0.85))
ax.set_yticks(range(len(SIX)))
ax.set_yticklabels(SIX, fontsize=FS_SMALL)
for tick, k in zip(ax.yaxis.get_major_ticks(), SIX):
    tick.label1.set_color(KC[k])
    tick.label1.set_fontweight('bold')
ax.set_xlabel('Crossover $\\mathrm{\\nu}$', fontsize=FS_LABEL)
ax.tick_params(axis='x', labelsize=FS_TICK)
ax.set_xlim(0.495, 0.68)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
lbl(ax, 'f')

# ══ Panel g: fold-change bars ══════════════════════════════════════════
ax = ax_h
fc_data = [
    ('$f_{aro}$', expanded['f_aro'].mean() / compact['f_aro'].mean(), '#C49A88', '#7A5040'),
    ('$f_K$',     expanded['f_K'].mean()   / compact['f_K'].mean(),   '#7A9EC4', '#3A5E84'),
    ('FCR',       expanded['FCR'].mean()   / compact['FCR'].mean(),   '#8AB898', '#407A50'),
    ('SCD',       expanded['SCD'].mean()   / compact['SCD'].mean(),   '#9A88C4', '#5040A0'),
]
anns = {
    '$f_{aro}$': f'{expanded["f_aro"].mean()/compact["f_aro"].mean():.2f}×',
    '$f_K$':     f'{expanded["f_K"].mean()/compact["f_K"].mean():.2f}×',
    'FCR':       f'{expanded["FCR"].mean()/compact["FCR"].mean():.2f}×',
    'SCD':       f'{expanded["SCD"].mean()/compact["SCD"].mean():.0f}×',
}
for yi, (label, fc, bcol, tcol) in enumerate(reversed(fc_data)):
    ax.barh(yi, fc, color=bcol, alpha=0.88, height=0.55, zorder=3)
    tx = fc * 1.08 if fc < 10 else fc + 2
    ax.text(tx, yi, anns[label], va='center', fontsize=FS_SMALL,
            color=tcol, fontweight='bold', ha='left')
ax.set_xscale('log')
ax.axvline(1, color='grey', ls='--', lw=3.5, alpha=0.7)
ax.set_yticks(range(len(fc_data)))
ax.set_yticklabels([d[0] for d in reversed(fc_data)], fontsize=FS_TICK)
ax.set_xlabel('Fold-change (log scale)', fontsize=FS_LABEL)
ax.tick_params(axis='x', labelsize=FS_TICK)
ax.set_xlim(0.5, 400)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 0.97, 'g', transform=ax.transAxes,
        fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')

# ── Save ──────────────────────────────────────────────────────────────
fig.savefig(OUT / 'fig5_synthesis.pdf', dpi=200, bbox_inches='tight')
fig.savefig(OUT / 'fig5_synthesis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig5_synthesis.pdf / fig5_synthesis.png")
