"""
Figure 3 — Compositional trajectory, apparent efficiency crossover, decomposition
Layout: 3-3-2  (rows 0,1 three panels each; row 2 two centred panels)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
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

FS_LABEL  = 60
FS_TICK   = 48
FS_LEGEND = 42
FS_ANNOT  = 44
FS_PANEL  = 62
FS_SMALL  = 40
FS_TITLE  = 48

# ── Load ──────────────────────────────────────────────────────────────
bio  = pd.read_csv(DATA / 'protein_metadata.csv')
df_c = pd.read_csv(DATA / 'contact_results.csv')
bio6 = bio[bio['kingdom'].isin(SIX)].copy()

bp = pd.DataFrame({
    'kingdom':    ['Bacteria','Fungi','Mammals','Plants','Protists','Viruses','pooled'],
    'descriptor': ['SCD']*7,
    'breakpoint': [0.55, 0.57, 0.56, 0.54, 0.55, 0.57, 0.56],
    'ci_lo_95':   [0.53, 0.55, 0.54, 0.52, 0.53, 0.55, 0.55],
    'ci_hi_95':   [0.57, 0.59, 0.58, 0.56, 0.57, 0.59, 0.57],
})

# ── Helpers ───────────────────────────────────────────────────────────
BW = 0.02
def binned(df, col, nu_col='nu', min_n=10):
    df = df.copy()
    df['nu_bin'] = np.round(df[nu_col]/BW)*BW
    g = df.groupby('nu_bin').agg(m=(col,'mean'), s=(col,'sem'),
                                  n=(col,'count')).query(f'n>={min_n}')
    return g.query('nu_bin>=0.37 and nu_bin<=0.72').sort_index()

def zone_lines(ax):
    """Vertical zone dividers + labels just above the spine (clear of titles)."""
    ax.axvline(0.54, color='#aaaaaa', lw=3.5, ls='-', alpha=0.7, zorder=1)
    ax.axvline(0.63, color='#aaaaaa', lw=3.5, ls='-', alpha=0.7, zorder=1)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    kw = dict(fontsize=FS_SMALL, color='#888888', style='normal',
              ha='center', transform=trans, clip_on=False, va='bottom')
    ax.text(0.455, 1.03, 'Compact',    **kw)
    ax.text(0.585, 1.03, 'Transition', **kw)
    ax.text(0.665, 1.03, 'Expanded',   **kw)

def lbl(ax, letter, title=''):
    ax.text(0.02, 0.97, letter, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', va='top', clip_on=False)
    if title:
        ax.text(0.0, 1.14, title, transform=ax.transAxes,
                fontsize=FS_LABEL, fontweight='normal', va='bottom', clip_on=False)

# ── Figure: 3-3-2 via 3×6 GridSpec ───────────────────────────────────
fig = plt.figure(figsize=(64, 50))
gs  = gridspec.GridSpec(3, 6, figure=fig, hspace=0.55, wspace=0.44,
                         left=0.06, right=0.98, top=0.93, bottom=0.06)

# Row 0
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])
# Row 1
ax_d = fig.add_subplot(gs[1, 0:2])
ax_e = fig.add_subplot(gs[1, 2:4])
ax_f = fig.add_subplot(gs[1, 4:6])
# Row 2 centred
ax_g = fig.add_subplot(gs[2, 1:3])
ax_h = fig.add_subplot(gs[2, 3:5])

for ax in [ax_a,ax_b,ax_c,ax_d,ax_e,ax_f,ax_g,ax_h]:
    ax.tick_params(labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ══ Panel a: f_R and f_K per kingdom ══════════════════════════════════
ax = ax_a
for k in SIX:
    sub = bio6[bio6['kingdom']==k].copy()
    sub['nu_bin'] = np.round(sub['nu']/BW)*BW
    g_R = sub.groupby('nu_bin')['f_R'].agg(['mean','sem','count'])
    g_K = sub.groupby('nu_bin')['f_K'].agg(['mean','sem','count'])
    g_R = g_R.query('count>=10 and nu_bin>=0.37 and nu_bin<=0.72')
    g_K = g_K.query('count>=10 and nu_bin>=0.37 and nu_bin<=0.72')
    ax.fill_between(g_R.index, g_R['mean']-g_R['sem'], g_R['mean']+g_R['sem'],
                    color=KC[k], alpha=0.12)
    ax.plot(g_R.index, g_R['mean'], color=KC[k], lw=7)
    ax.fill_between(g_K.index, g_K['mean']-g_K['sem'], g_K['mean']+g_K['sem'],
                    color=KC[k], alpha=0.12)
    ax.plot(g_K.index, g_K['mean'], color=KC[k], lw=7, ls='--')

zone_lines(ax)
ax.set_xlim(0.37, 0.72); ax.set_ylim(0, 0.42)
ax.set_xlabel('$\\mathrm{\\nu}$', fontsize=FS_LABEL)
ax.set_ylabel('Residue fraction', fontsize=FS_LABEL)
handles_a = ([Line2D([0],[0], color='grey', lw=7,       label='$f_R$ (Arg)'),
              Line2D([0],[0], color='grey', lw=7, ls='--', label='$f_K$ (Lys)')] +
             [Line2D([0],[0], color=KC[k], lw=7, label=k) for k in SIX])
ax.legend(handles=handles_a, fontsize=FS_LEGEND, ncol=2,
          loc='upper left', framealpha=0.0, handlelength=1.4,
          borderpad=0.4, labelspacing=0.3)
lbl(ax, 'a')

# ══ Panel b: apparent efficiency crossover ════════════════════════════
ax = ax_b
lys_cols = ['eff_Lys_Tyr','eff_Lys_Phe','eff_Lys_Trp','eff_Lys_Glu','eff_Lys_Asp','eff_Lys_Lys']
arg_cols = ['eff_Arg_Tyr','eff_Arg_Phe','eff_Arg_Trp','eff_Arg_Glu','eff_Arg_Asp','eff_Arg_Arg']
df6 = df_c[df_c['kingdom'].isin(SIX)].copy()
df6['nu_bin'] = np.round(df6['nu']/BW)*BW
df6['lys_eff'] = df6[[c for c in lys_cols if c in df6.columns]].replace(0,np.nan).mean(axis=1)
df6['arg_eff'] = df6[[c for c in arg_cols if c in df6.columns]].replace(0,np.nan).mean(axis=1)
g_lys = df6.groupby('nu_bin').agg(m=('lys_eff','mean'),s=('lys_eff','sem'),n=('lys_eff','count'))
g_arg = df6.groupby('nu_bin').agg(m=('arg_eff','mean'),s=('arg_eff','sem'),n=('arg_eff','count'))
g_lys = g_lys.query('n>=20 and nu_bin>=0.37 and nu_bin<=0.72')
g_arg = g_arg.query('n>=20 and nu_bin>=0.37 and nu_bin<=0.72')

ax.fill_between(g_lys.index, g_lys['m']-g_lys['s'], g_lys['m']+g_lys['s'],
                color='#5b8db8', alpha=0.2)
ax.plot(g_lys.index, g_lys['m'], 's-', color='#5b8db8', lw=10, ms=16,
        label='Lys-mediated')
ax.fill_between(g_arg.index, g_arg['m']-g_arg['s'], g_arg['m']+g_arg['s'],
                color='#e07040', alpha=0.2)
ax.plot(g_arg.index, g_arg['m'], 'o-', color='#e07040', lw=10, ms=16,
        label='Arg-mediated')
# ax.text(0.97, 0.97,
        # 'Apparent crossover:\nLys efficiency falls,\nArg efficiency rises',
        # fontsize=FS_ANNOT, color='grey', style='normal',
        # ha='right', va='top', transform=ax.transAxes,
        # bbox=dict(boxstyle='round', fc='white', alpha=0.85, lw=0))
zone_lines(ax)
ax.set_xlim(0.37, 0.72); ax.set_ylim(0, 55)
ax.set_xlabel('$\\mathrm{\\nu}$', fontsize=FS_LABEL)
ax.set_ylabel('Contact efficiency', fontsize=FS_LABEL)
ax.legend(fontsize=FS_LEGEND, framealpha=0.85, loc='upper left')
lbl(ax, 'b')

# ══ Panel c: log-decomposition ════════════════════════════════════════
ax = ax_c
df6m = df6.merge(bio[['UniProt_ID','f_K','f_F']].rename(
    columns={'f_F':'f_Phe'}), on='UniProt_ID', how='left')
compact_mean = df6m[df6m['nu']<0.54]
ref_logE  = np.log(compact_mean['eff_Lys_Phe'].replace(0,np.nan).dropna()).mean()
ref_logR  = (np.log(compact_mean['raw_Lys_Phe'].replace(0,np.nan).dropna()).mean()
             if 'raw_Lys_Phe' in df6m.columns else np.nan)
ref_logfK = np.log(compact_mean['f_K'].replace(0,np.nan).dropna()).mean()

bins_c = []
for nb in sorted(df6m['nu_bin'].unique()):
    sub = df6m[df6m['nu_bin']==nb]
    if len(sub) < 20: continue
    logE  = np.log(sub['eff_Lys_Phe'].replace(0,np.nan).dropna()).mean()
    logfK = np.log(sub['f_K'].replace(0,np.nan).dropna()).mean()
    logR  = (np.log(sub['raw_Lys_Phe'].replace(0,np.nan).dropna()).mean()
             if 'raw_Lys_Phe' in sub.columns else np.nan)
    bins_c.append({'nu_bin':nb, 'dlogE':logE-ref_logE,
                   'dlogR': logR-ref_logR if not np.isnan(ref_logR) else np.nan,
                   'dlogfK':logfK-ref_logfK})
bc = pd.DataFrame(bins_c).query('nu_bin>=0.37 and nu_bin<=0.72')

ax.axhline(0, color='grey', lw=3.5, ls='--', alpha=0.6)
ax.plot(bc['nu_bin'], bc['dlogE'],  'o--', color='black',   lw=8,  ms=14,
        label='log(efficiency)')
if not bc['dlogR'].isna().all():
    ax.plot(bc['nu_bin'], bc['dlogR'], 's--', color='#cc3333', lw=6, ms=14,
            label='log(raw contacts)')
ax.plot(bc['nu_bin'], bc['dlogfK'], '^:',  color='#4499cc', lw=6, ms=14,
        label='log($f_K$) [inflates denominator]')
ax.text(0.97, 0.55, 'denominator\ngap', fontsize=FS_ANNOT, color='#4499cc',
        style='normal', ha='right', va='top', transform=ax.transAxes)
ax.text(0.97, 0.28, 'genuine\ndecline',  fontsize=FS_ANNOT, color='#cc3333',
        style='normal', ha='right', va='top', transform=ax.transAxes)
zone_lines(ax)
ax.set_xlim(0.37, 0.72)
ax.set_xlabel('$\\mathrm{\\nu}$', fontsize=FS_LABEL)
ax.set_ylabel('Δ log (relative to compact)', fontsize=FS_LABEL)
ax.legend(fontsize=FS_LEGEND, loc='lower left', framealpha=0.85)
lbl(ax, 'c')

# ══ Panel d: SCD per kingdom ══════════════════════════════════════════
ax = ax_d
for k in SIX:
    sub = bio6[bio6['kingdom']==k]
    g = binned(sub, 'SCD')
    ax.fill_between(g.index, g['m']-g['s'], g['m']+g['s'], color=KC[k], alpha=0.15)
    ax.plot(g.index, g['m'], color=KC[k], lw=10, label=k)
ax.axvline(0.56, color='grey', lw=3.5, ls=':', alpha=0.8)
trans_d = blended_transform_factory(ax.transData, ax.transAxes)
ax.text(0.575, 0.96, 'onset\n$\\mathrm{\\nu}$=0.56', fontsize=FS_ANNOT, color='grey',
        style='normal', ha='left', va='top', transform=trans_d)
zone_lines(ax)
ax.set_xlim(0.37, 0.72); ax.set_ylim(-5, 110)
ax.set_xlabel('$\\mathrm{\\nu}$', fontsize=FS_LABEL); ax.set_ylabel('SCD', fontsize=FS_LABEL)
ax.legend(fontsize=FS_LEGEND, loc='upper left', framealpha=0.0, ncol=2,
          handlelength=1.2, labelspacing=0.3)
lbl(ax, 'd')

# ══ Panel e: NCPR per kingdom ═════════════════════════════════════════
ax = ax_e
ax.axhline(0, color='grey', lw=3.5, ls=':', alpha=0.6)
for k in SIX:
    sub = bio6[bio6['kingdom']==k]
    g = binned(sub, 'NCPR')
    ax.fill_between(g.index, g['m']-g['s'], g['m']+g['s'], color=KC[k], alpha=0.15)
    ax.plot(g.index, g['m'], color=KC[k], lw=10, label=k)
ax.text(0.97, 0.97, '+ Bact, Prot, Vir', fontsize=FS_ANNOT, color='grey',
        ha='right', va='top', transform=ax.transAxes)
ax.text(0.97, 0.30, '− Fungi, Plants',   fontsize=FS_ANNOT, color='grey',
        ha='right', va='top', transform=ax.transAxes)
ax.text(0.97, 0.58, '0 Mammals',         fontsize=FS_ANNOT, color='grey',
        ha='right', va='top', transform=ax.transAxes)
zone_lines(ax)
ax.set_xlim(0.37, 0.72); ax.set_ylim(-0.45, 0.50)
ax.set_xlabel('$\\mathrm{\\nu}$', fontsize=FS_LABEL); ax.set_ylabel('NCPR', fontsize=FS_LABEL)
lbl(ax, 'e')

# ══ Panel f: FCR per kingdom ══════════════════════════════════════════
ax = ax_f
for k in SIX:
    sub = bio6[bio6['kingdom']==k]
    g = binned(sub, 'FCR')
    ax.fill_between(g.index, g['m']-g['s'], g['m']+g['s'], color=KC[k], alpha=0.15)
    ax.plot(g.index, g['m'], color=KC[k], lw=10, label=k)
# ax.text(0.05, 0.08, '1.95× increase\ncompact → expanded',
#         fontsize=FS_ANNOT, color='grey', style='normal',
#         transform=ax.transAxes, va='bottom')
zone_lines(ax)
ax.set_xlim(0.37, 0.72); ax.set_ylim(0.15, 0.85)
ax.set_xlabel('$\\mathrm{\\nu}$', fontsize=FS_LABEL); ax.set_ylabel('FCR', fontsize=FS_LABEL)
lbl(ax, 'f')

# ══ Panel g: aromatic fraction ════════════════════════════════════════
ax = ax_g
for k in SIX:
    sub = bio6[bio6['kingdom']==k]
    g = binned(sub, 'f_aro')
    ax.fill_between(g.index, g['m']-g['s'], g['m']+g['s'], color=KC[k], alpha=0.15)
    ax.plot(g.index, g['m'], color=KC[k], lw=10, label=k)
zone_lines(ax)
ax.set_xlim(0.37, 0.72); ax.set_ylim(0.01, 0.09)
ax.set_xlabel('$\\mathrm{\\nu}$', fontsize=FS_LABEL)
ax.set_ylabel('Aromatic fraction', fontsize=FS_LABEL, labelpad=100)
ax.legend(fontsize=FS_LEGEND, ncol=1, loc='center left',
          bbox_to_anchor=(-0.52, 0.5), framealpha=0.0,
          handlelength=1.5, labelspacing=0.45, borderaxespad=0)
lbl(ax, 'g')

# ══ Panel h: SCD breakpoint per kingdom ══════════════════════════════
ax = ax_h
bp_scd = bp[bp['descriptor']=='SCD'].copy()
kingdom_order_h = ['Viruses','Protists','Plants','Mammals','Fungi','Bacteria']
pooled_row = bp_scd[bp_scd['kingdom']=='pooled']

for yi, k in enumerate(kingdom_order_h):
    row = bp_scd[bp_scd['kingdom']==k]
    if len(row) == 0: continue
    row = row.iloc[0]
    col = KC[k]
    ax.errorbar(row['breakpoint'], yi,
                xerr=[[row['breakpoint']-row['ci_lo_95']],
                      [row['ci_hi_95']-row['breakpoint']]],
                fmt='o', color=col, ms=30, capsize=14, lw=10,
                markeredgecolor='white', markeredgewidth=3, zorder=4)
    ax.text(row['ci_hi_95']+0.003, yi, '$\\mathrm{\\nu}$=' + f'{row["breakpoint"]:.2f}',
            va='center', fontsize=FS_ANNOT, color=col,
            fontweight='bold', clip_on=False)

if len(pooled_row) > 0:
    pr = pooled_row.iloc[0]
    yi_p = len(kingdom_order_h)
    ax.errorbar(pr['breakpoint'], yi_p, fmt='o', color='#333333',
                ms=34, capsize=14, lw=10,
                markeredgecolor='white', markeredgewidth=3, zorder=5)
    ax.text(pr.get('ci_hi_95', pr['breakpoint'])+0.003, yi_p,
            'Pooled $\\mathrm{\\nu}$=' + f'{pr["breakpoint"]:.2f}',
            va='center', fontsize=FS_ANNOT, fontweight='bold', clip_on=False)

ax.axvline(0.588, color='grey', lw=4, ls='--', alpha=0.7)

# ax.text(0.03, 0.97, 'All kingdoms: onset\nbefore Flory boundary',
#         fontsize=FS_ANNOT, color='grey', style='normal', va='top',
#         transform=ax.transAxes,
#         bbox=dict(boxstyle='round', fc='white', alpha=0.85,
#                   lw=0.8, ec='#cccccc'))
ax.set_yticks(list(range(len(kingdom_order_h))) + [len(kingdom_order_h)])
ax.set_yticklabels(kingdom_order_h + ['Pooled'], fontsize=FS_TICK)
ax.set_xlabel('$\\mathrm{\\nu}$ (SCD breakpoint)', fontsize=FS_LABEL)
ax.set_xlim(0.445, 0.655)
lbl(ax, 'h')

# ── Save ──────────────────────────────────────────────────────────────
fig.savefig(OUT / 'fig3_trajectory.pdf', dpi=200, bbox_inches='tight')
fig.savefig(OUT / 'fig3_trajectory.png', dpi=200, bbox_inches='tight')
plt.close()
print(f\"Saved: {OUT}/fig3_trajectory.pdf / fig3_trajectory.png\")
