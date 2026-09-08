"""
Figure 2 — Arginine out-contacts Lysine: composition-controlled direct test
Layout: 3-3-2 (rows 0,1 three panels each; row 2 two centred panels)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import blended_transform_factory
from scipy import stats
import warnings, os
from pathlib import Path
_ROOT = Path(__file__).parent
DATA  = _ROOT / 'data'
OUT   = _ROOT / 'figures'
OUT.mkdir(exist_ok=True)

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

SIX = ['Bacteria','Fungi','Mammals','Plants','Protists','Viruses']
KC  = {'Bacteria':'#1f77b4','Fungi':'#2ca02c','Mammals':'#9467bd',
       'Plants':'#e377c2','Protists':'#bcbd22','Viruses':'#17becf'}

FS_LABEL  = 72
FS_TICK   = 56
FS_LEGEND = 48
FS_ANNOT  = 56
FS_PANEL  = 62
FS_ZONE   = 54
FS_TITLE  = 56

# ── Load ──────────────────────────────────────────────────────────────
df  = pd.read_csv(DATA / 'substitution_results.csv')
bio = pd.read_csv(DATA / 'protein_metadata.csv')
dnu = pd.read_csv(DATA / 'delta_nu_substitution.csv')

wt_nu_bio = bio.set_index('UniProt_ID')['nu'].to_dict()
bio_k     = bio.set_index('UniProt_ID')['kingdom'].to_dict()

ktr = df[df['variant']=='K_to_R'].set_index('UniProt_ID')
rtk = df[df['variant']=='R_to_K'].set_index('UniProt_ID')
common = sorted(set(ktr.index) & set(rtk.index))

partners_full = [('eff_Arg_Tyr','eff_Lys_Tyr','Tyr'),
                 ('eff_Arg_Phe','eff_Lys_Phe','Phe'),
                 ('eff_Arg_Trp','eff_Lys_Trp','Trp'),
                 ('eff_Arg_Glu','eff_Lys_Glu','Glu'),
                 ('eff_Arg_Asp','eff_Lys_Asp','Asp'),
                 ('eff_Arg_Ser','eff_Lys_Ser','Ser')]

rows = []
for uid in common:
    av, lv, pp = [], [], {}
    for ac, lc, pn in partners_full:
        a = ktr.loc[uid,ac] if ac in ktr.columns else np.nan
        l = rtk.loc[uid,lc] if lc in rtk.columns else np.nan
        if np.isfinite(a) and a > 0: av.append(a)
        if np.isfinite(l) and l > 0: lv.append(l)
        pp['arg_'+pn] = a if np.isfinite(a) else np.nan
        pp['lys_'+pn] = l if np.isfinite(l) else np.nan
    if av and lv:
        rows.append({'UniProt_ID':uid, 'nu_wt':wt_nu_bio.get(uid,np.nan),
                     'kingdom':bio_k.get(uid,'unknown'),
                     'arg_eff':np.nanmean(av), 'lys_eff':np.nanmean(lv), **pp})

paired = pd.DataFrame(rows)
paired['diff']   = paired['arg_eff'] - paired['lys_eff']
paired['diff_w'] = paired['diff'].clip(paired['diff'].quantile(0.01),
                                        paired['diff'].quantile(0.99))
paired['nu_wt_bio'] = paired['UniProt_ID'].map(wt_nu_bio)

def zone(nu):
    if pd.isna(nu): return 'unknown'
    if nu < 0.54:   return 'Compact'
    if nu <= 0.63:  return 'Transition'
    return 'Expanded'
paired['zone'] = paired['nu_wt_bio'].apply(zone)

paired['nu_bin'] = np.round(paired['nu_wt'] / 0.04) * 0.04
bins = paired.groupby('nu_bin').agg(
    m=('diff_w','mean'), s=('diff_w','sem'), n=('UniProt_ID','count')
).query('n>=5 and nu_bin>=0.38 and nu_bin<=0.70').sort_index()

np.random.seed(42)
rand_paired = paired.copy()
swap = np.random.rand(len(rand_paired)) > 0.5
rand_paired.loc[swap,'diff'] = -rand_paired.loc[swap,'diff']
rand_paired['diff_w'] = rand_paired['diff'].clip(
    rand_paired['diff'].quantile(0.01), rand_paired['diff'].quantile(0.99))
rand_paired['nu_bin'] = np.round(rand_paired['nu_wt'] / 0.04) * 0.04
bins_rand = rand_paired.groupby('nu_bin').agg(
    m=('diff_w','mean'), s=('diff_w','sem'), n=('UniProt_ID','count')
).query('n>=5 and nu_bin>=0.38 and nu_bin<=0.70').sort_index()

fig_n = {'Compact':120, 'Transition':117, 'Expanded':103}

# ── Figure: 3-3-2 layout via 3×6 GridSpec ─────────────────────────────
np.random.seed(42)
fig = plt.figure(figsize=(76, 58))
gs  = gridspec.GridSpec(3, 6, figure=fig, hspace=0.60, wspace=0.46,
                         left=0.06, right=0.98, top=0.93, bottom=0.07)

ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])
ax_d = fig.add_subplot(gs[1, 0:2])
ax_e = fig.add_subplot(gs[1, 2:4])
ax_f = fig.add_subplot(gs[1, 4:6])
ax_g = fig.add_subplot(gs[2, 0:3])
ax_h = fig.add_subplot(gs[2, 3:6])

def lbl(ax, l, title=''):
    ax.text(0.02, 0.97, l, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', va='top', clip_on=False)
    if title:
        ax.text(0.0, 1.16, title, transform=ax.transAxes,
                fontsize=FS_LABEL, fontweight='normal', va='bottom', clip_on=False)

def draw_zones(ax):
    """Vertical zone dividers with labels floated just above the top spine."""
    ax.axvline(0.54, color='#aaaaaa', lw=1.9, ls='-', alpha=0.8, zorder=1)
    ax.axvline(0.63, color='#aaaaaa', lw=1.9, ls='-', alpha=0.8, zorder=1)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.455, 1.05, 'Compact',    color='grey', fontsize=FS_ZONE,
            style='normal', alpha=0.9, ha='center', transform=trans,
            clip_on=False, va='bottom')
    ax.text(0.585, 1.05, 'Transition', color='grey', fontsize=FS_ZONE,
            style='normal', alpha=0.9, ha='center', transform=trans,
            clip_on=False, va='bottom')
    ax.text(0.705, 1.05, 'Expanded',   color='grey', fontsize=FS_ZONE,
            style='normal', alpha=0.9, ha='right', transform=trans,
            clip_on=False, va='bottom')

# ── Panel a ───────────────────────────────────────────────────────────
ax = ax_a; draw_zones(ax)
ax.grid(axis='y', color='#dddddd', lw=0.8, zorder=0)
for _, row in paired.iterrows():
    ax.scatter(row['nu_wt'], row['diff'], color=KC.get(row['kingdom'],'#888'),
               alpha=0.60, s=500, zorder=2, edgecolors='none')
ax.fill_between(bins.index, bins['m']-bins['s'], bins['m']+bins['s'],
                color='red', alpha=0.22, zorder=3)
ax.plot(bins.index, bins['m'], color='red', lw=12, zorder=4)
ax.scatter(bins.index, bins['m'], s=550, color='red', edgecolors='white', lw=6, zorder=5)
ax.axhline(0, color='grey', ls='--', lw=2, alpha=0.7)
ax.set_xlim(0.37, 0.71); ax.set_ylim(-3.5, 7.5)
ax.set_xlabel('Wildtype $\\mathrm{\\nu}$', fontsize=FS_LABEL)
ax.set_ylabel('Arg eff − Lys eff', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
lbl(ax, 'a')

# ── Panel b ───────────────────────────────────────────────────────────
ax = ax_b; draw_zones(ax)
ax.grid(axis='y', color='#dddddd', lw=0.8, zorder=0)
pct_r = (rand_paired['diff'] > 0).mean() * 100
for _, row in rand_paired.iterrows():
    ax.scatter(row['nu_wt'], row['diff'], color=KC.get(row['kingdom'],'#888'),
               alpha=0.60, s=500, zorder=2, edgecolors='none')
ax.fill_between(bins_rand.index, bins_rand['m']-bins_rand['s'],
                bins_rand['m']+bins_rand['s'], color='salmon', alpha=0.22, zorder=3)
ax.plot(bins_rand.index, bins_rand['m'], color='salmon', lw=11,
        linestyle='dotted', zorder=4)
ax.scatter(bins_rand.index, bins_rand['m'], s=550, color='salmon',
           edgecolors='white', lw=6, zorder=5)
ax.axhline(0, color='steelblue', lw=10, zorder=3)
ax.fill_between([0.37,0.71], [-0.05]*2, [0.05]*2,
                color='steelblue', alpha=0.15, zorder=2)
ax.axhline(0, color='grey', ls='--', lw=2, alpha=0.7)
ax.set_xlim(0.37, 0.71); ax.set_ylim(-3.5, 7.5)
ax.set_xlabel('Wildtype $\\mathrm{\\nu}$', fontsize=FS_LABEL)
ax.set_ylabel('Arg eff − Lys eff\n(randomised)', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
lbl(ax, 'b')

# ── Panel c: per-partner bars ──────────────────────────────────────────
ax = ax_c
pnames = ['Tyr','Phe','Trp','Glu','Asp','Ser']
arg_means, lys_means, arg_sems, lys_sems = [], [], [], []
for _, _, pn in partners_full:
    acol, lcol = 'arg_'+pn, 'lys_'+pn
    sub = paired[[acol,lcol]].dropna()
    sub = sub[(sub[acol]>0) & (sub[lcol]>0)]
    a_w = sub[acol].clip(sub[acol].quantile(0.02), sub[acol].quantile(0.98))
    l_w = sub[lcol].clip(sub[lcol].quantile(0.02), sub[lcol].quantile(0.98))
    arg_means.append(a_w.mean()); arg_sems.append(a_w.sem())
    lys_means.append(l_w.mean()); lys_sems.append(l_w.sem())

x = np.arange(len(pnames)); w = 0.35
ax.grid(axis='y', color='#dddddd', lw=0.8, zorder=0)
ax.bar(x-w/2, arg_means, w, color='#e07040', label='Arg (K→R)',
       yerr=[1.96*s for s in arg_sems], capsize=18,
       error_kw={'lw':8,'capthick':8}, alpha=0.88, zorder=3)
ax.bar(x+w/2, lys_means, w, color='#5b8db8', label='Lys (R→K)',
       yerr=[1.96*s for s in lys_sems], capsize=18,
       error_kw={'lw':8,'capthick':8}, alpha=0.88, zorder=3)
for xi, (_,_,pn) in enumerate(partners_full):
    acol, lcol = 'arg_'+pn, 'lys_'+pn
    sub = paired[[acol,lcol]].dropna(); sub = sub[(sub[acol]>0)&(sub[lcol]>0)]
    _, pw = stats.wilcoxon(sub[acol], sub[lcol])
    sig = '***' if pw<0.001 else '**' if pw<0.01 else '*' if pw<0.05 else 'ns'
    ymax = max(arg_means[xi]+1.96*arg_sems[xi], lys_means[xi]+1.96*lys_sems[xi])
    ax.text(xi, ymax+0.4, sig, ha='center', fontsize=FS_ANNOT, fontweight='bold')
ax.axvline(2.5, color='grey', ls='--', lw=1.9, alpha=0.6)
# Float category labels centred over their respective bar groups
ax.text(0.25, 1.05, 'aromatic',      color='#e07040', fontsize=FS_ZONE, ha='center', transform=ax.transAxes,
        clip_on=False, va='bottom')
ax.text(0.75, 1.05, 'charged/polar', color='#5b8db8', fontsize=FS_ZONE, ha='center', transform=ax.transAxes,
        clip_on=False, va='bottom')
ax.set_xticks(x); ax.set_xticklabels(pnames, fontsize=FS_TICK)
ax.tick_params(axis='y', labelsize=FS_TICK)
ax.set_ylabel('Contact efficiency', fontsize=FS_LABEL)
ax.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.85)
ax.set_ylim(0, 18)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
lbl(ax, 'c')

# ── Panel d: by compaction zone ───────────────────────────────────────
ax = ax_d
zone_order  = ['Expanded','Transition','Compact']
zone_labels = ['Expanded\n($\\mathrm{\\nu}$>0.63)','Transition\n(0.54–0.63)','Compact\n($\\mathrm{\\nu}$<0.54)']
zone_colors = ['#2ca02c','#888888','#4878d0']
for yi, (zname, zlabel, zcol) in enumerate(zip(zone_order, zone_labels, zone_colors)):
    sub = paired[paired['zone']==zname].dropna(subset=['arg_eff','lys_eff'])
    diffs = sub['arg_eff'] - sub['lys_eff']
    jit = np.random.uniform(-0.15, 0.15, len(sub))
    ax.scatter(diffs.values, yi+jit, color=zcol, alpha=0.35, s=280, zorder=2)
    ax.errorbar(diffs.mean(), yi, xerr=1.96*diffs.sem(), fmt='s', color=zcol,
                ms=44, capsize=20, lw=10, zorder=5, markeredgecolor='white')
    _, pw3 = stats.wilcoxon(sub['arg_eff'], sub['lys_eff'])
    sig = '***' if pw3<0.001 else '**' if pw3<0.01 else '*' if pw3<0.05 else 'ns'
    ax.text(5.2, yi, f'{sig}  n={fig_n[zname]}', va='center', fontsize=FS_ANNOT)
ax.axvline(0, color='grey', ls='--', lw=1.8)
ax.set_xlim(-5.5, 7.5)
ax.set_yticks(range(3)); ax.set_yticklabels(zone_labels, fontsize=40)
ax.tick_params(axis='x', labelsize=FS_TICK)
ax.set_xlabel('Arg − Lys efficiency', fontsize=FS_LABEL)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
lbl(ax, 'd')

# ── Panel e: violin summary ───────────────────────────────────────────
ax = ax_e
ax.grid(axis='y', color='#dddddd', lw=0.8, zorder=0)
for xi, (data, col, pct_val, p_str) in enumerate([
        (paired['diff'].dropna(),      '#e07040', 84, '4×10$^{-31}$'),
        (rand_paired['diff'].dropna(), '#5b8db8', round(pct_r), '0.59(ns)')]):
    parts = ax.violinplot(data, positions=[xi], widths=0.75,
                          showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(col); pc.set_alpha(0.45)
    q25, med, q75 = np.percentile(data, [25, 50, 75])
    ax.vlines(xi, q25, q75, color=col, lw=26, alpha=0.6, zorder=4)
    ax.hlines(med, xi-0.18, xi+0.18, color='black', lw=14, zorder=5)
    if xi == 0:
        for _, row in paired.iterrows():
            jv = np.random.uniform(-0.18, 0.18)
            ax.scatter(xi+jv, row['diff'], color=KC.get(row['kingdom'],'grey'),
                       alpha=0.45, s=200, zorder=3)
    else:
        jit = np.random.uniform(-0.18, 0.18, len(data))
        ax.scatter(xi+jit, data, color=col, alpha=0.40, s=200, zorder=3)
    ax.errorbar(xi, data.mean(), yerr=1.96*data.sem(), fmt='o', color='white',
                ms=36, capsize=20, lw=10, markeredgecolor='black', zorder=6)
    pcol = 'darkred' if xi==0 else 'steelblue'
    ax.text(xi, -4.0, f'{pct_val}%>0\np={p_str}', ha='center', va='bottom',
            fontsize=44, color=pcol, fontweight='bold')
ax.axhline(0, color='grey', ls='--', lw=1.8)
ax.set_xticks([0,1])
ax.set_xticklabels(['Matched\npositions','Randomised\npositions'], fontsize=FS_TICK)
ax.tick_params(axis='y', labelsize=FS_TICK)
ax.set_ylabel('Arg eff − Lys eff', fontsize=FS_LABEL)
ax.set_ylim(-5.5, 5.0)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
lbl(ax, 'e')

# ── Panel f: substitution $\\mathrm{\\Delta\\nu}$ by zone ─────────────────────────────────
ax = ax_f
zone_keys   = ['below_threshold','near_threshold','above_threshold']
zone_xlbls  = ['Compact','Transition','Expanded']
for vi, (var, col, label) in enumerate([
        ('R_to_K',          '#e07040', 'R→K'),
        ('shuffle_control',  '#5b8db8', 'Shuffle')]):
    means2, sems2 = [], []
    for zk in zone_keys:
        sub = dnu[(dnu['variant']==var)&(dnu['nu_zone']==zk)]['delta_nu'].dropna()
        means2.append(sub.mean()); sems2.append(sub.sem())
        jit = np.random.uniform(-0.12, 0.12, len(sub))
        ax.scatter(zone_keys.index(zk)+(vi-0.5)*0.35+jit*0.3,
                   sub.values, color=col, alpha=0.35, s=200, zorder=2)
    ax.bar(np.arange(3)+(vi-0.5)*0.35, means2, 0.30, color=col, alpha=0.88,
           yerr=[1.96*s for s in sems2], capsize=18,
           error_kw={'lw':8,'capthick':8}, label=label, zorder=3)
ax.annotate('p=1.6×10$^{-4}$', xy=(1, 0.015), fontsize=FS_ANNOT,
            ha='center', color='black')
ax.axhline(0, color='grey', ls='--', lw=1.8)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(zone_xlbls, fontsize=FS_TICK)
ax.tick_params(axis='y', labelsize=FS_TICK)
ax.set_ylabel('$\\mathrm{\\Delta\\nu}$ (variant − wt)', fontsize=FS_LABEL)
ax.legend(fontsize=FS_LEGEND, framealpha=0.8)
ax.set_ylim(-0.12, 0.22)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
lbl(ax, 'f')

# ── Panel g: per-taxon strip ──────────────────────────────────────────
ax = ax_g
kingdom_order = ['Viruses','Protists','Plants','Mammals','Fungi','Bacteria']
XMIN, XMAX = -6, 10
XPAD = 2.5   # extra padding for edge labels
for yi, k in enumerate(kingdom_order):
    sub_r = paired[paired['kingdom']==k]['diff'].dropna()
    sub_w = sub_r.clip(sub_r.quantile(0.01), sub_r.quantile(0.99))
    col = KC[k]
    disp = sub_r.clip(XMIN, XMAX)
    jit  = np.random.uniform(-0.15, 0.15, len(sub_r))
    ax.scatter(disp.values, yi+jit, color=col, alpha=0.40, s=320,
               zorder=2, edgecolors='none')
    ax.errorbar(sub_w.mean(), yi, xerr=1.96*sub_w.sem(), fmt='o', color=col,
                ms=54, capsize=20, lw=10, markeredgecolor='white',
                markeredgewidth=4, zorder=5)
    pct = (sub_r > 0).mean() * 100
    _, pw = stats.wilcoxon(sub_r)
    sig = '***' if pw<0.001 else '**' if pw<0.01 else '*' if pw<0.05 else 'ns'
    ax.text(XMIN-XPAD+0.2, yi, f'{pct:.0f}%', va='center', ha='left',
            fontsize=FS_ANNOT, color=col, fontweight='bold')
    ax.text(XMAX+XPAD-0.2, yi, sig, va='center', ha='right',
            fontsize=FS_ANNOT, color=col, fontweight='bold')
ax.axvline(0, color='grey', ls='--', lw=1.9, alpha=0.7)
ax.set_yticks(range(len(kingdom_order)))
ax.set_yticklabels(kingdom_order, fontsize=FS_TICK)
ax.tick_params(axis='x', labelsize=FS_TICK)
ax.set_xlabel('Arg − Lys efficiency', fontsize=FS_LABEL)
ax.set_xlim(XMIN-XPAD, XMAX+XPAD)
ax.set_ylim(-0.7, 6.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
lbl(ax, 'g')

# ── Panel h: CDF ──────────────────────────────────────────────────────
ax = ax_h
all_diff = paired['diff'].dropna().sort_values()
cdf = np.arange(1, len(all_diff)+1) / len(all_diff)
ax.grid(color='#dddddd', lw=0.8, zorder=0)
ax.plot(all_diff, cdf, 'k-', lw=14, label='Overall', zorder=5)
for k in SIX:
    sub = paired[paired['kingdom']==k]['diff'].dropna().sort_values()
    cdf_k = np.arange(1, len(sub)+1) / len(sub)
    ax.plot(sub, cdf_k, color=KC[k], lw=10, alpha=0.88, label=k)
ax.axvline(0, color='grey', ls='--', lw=1.8)
pct_pos = (paired['diff'] > 0).mean()
ax.annotate('84%', xy=(0, 1-pct_pos), xytext=(2.0, 1-pct_pos-0.12),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
            color='red', fontsize=FS_ANNOT, fontweight='bold', va='center')
ax.set_xlabel('Arg eff − Lys eff', fontsize=FS_LABEL)
ax.set_ylabel('Cumulative fraction', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.set_xlim(-4.5, 4.5); ax.set_ylim(0, 1)
ax.legend(fontsize=FS_LEGEND, loc='upper left', framealpha=0.8, ncol=1)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
lbl(ax, 'h')

# ── Trim panels g and h to 2.5 cols (remove 0.5-col outer margin each) ─
_pg = ax_g.get_position(); _ph = ax_h.get_position()
_shrink = _pg.width * (0.5 / 3)
ax_g.set_position([_pg.x0 + _shrink, _pg.y0, _pg.width - _shrink, _pg.height])
ax_h.set_position([_ph.x0, _ph.y0, _ph.width - _shrink, _ph.height])

# ── Save ──────────────────────────────────────────────────────────────
fig.savefig(OUT / 'fig2_dissociation.pdf', dpi=200, bbox_inches='tight')
fig.savefig(OUT / 'fig2_dissociation.png', dpi=200, bbox_inches='tight')
plt.close()
print(f\"Saved: {OUT}/fig2_dissociation.pdf / fig2_dissociation.png\")
