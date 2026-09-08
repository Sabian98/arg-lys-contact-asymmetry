"""
Figure 1 — BENDER cross-taxon IDP ensemble dataset
Panels: a) KDE of nu, b) count histogram by nu bin,
        c) PAE concordance violins, d) representative SD_dist + PAE maps

Inputs needed:
    BENDER_BIO.csv           — columns: kingdom, nu (panels a, b)
    pae_concordance.csv      — columns: UniProt_ID, kingdom, pearson_r (panel c)
                               (computed on HPC: per-sequence Pearson r of SD_dist vs PAE)
    sd_maps/                 — directory with {UniProt_ID}_sd.npy files (panel d)
    pae_maps/                — directory with {UniProt_ID}_pae.npy files (panel d)

Panel d proteins (global median r=0.805):
    N9ZEX2 (Bacteria), Q2HAA5 (Fungi), Q8CF14 (Mammals), Q0PFA9 (Viruses)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from matplotlib.transforms import blended_transform_factory
import os, sys
from pathlib import Path
_ROOT = Path(__file__).parent
DATA  = _ROOT / 'data'
OUT   = _ROOT / 'figures'
OUT.mkdir(exist_ok=True)


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

# ── Colours (match original) ──────────────────────────────────────────
KC = {'Bacteria':'#e05050', 'Viruses':'#9990cc', 'Mammals':'#2a9d6f',
      'Plants':'#1a5fa8',   'Fungi':'#40c0c0',   'Protists':'#f0a050'}

SIX = ['Bacteria','Fungi','Mammals','Plants','Protists','Viruses']
FS_LABEL  = 20
FS_TICK   = 16
FS_LEGEND = 16
FS_ANNOT  = 16
FS_PANEL  = 20

# ── Load data ─────────────────────────────────────────────────────────
bio = pd.read_csv(DATA / 'protein_metadata.csv')
bio6 = bio[bio['kingdom'].isin(SIX)].copy()

# Panel c: load real PAE concordance if available
pae_path = DATA / 'pae_concordance.tsv'
if os.path.exists(pae_path):
    pae_df = pd.read_csv(pae_path, sep='\t')
else:
    print("WARNING: pae_concordance.csv not found — panel c will use placeholder")
    pae_df = None

# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                         left=0.07, right=0.97, top=0.95, bottom=0.07)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])
ax_d.axis('off')

# ═══════════════════════════════════════════════════════════════════════
# PANEL a: KDE of nu
# ═══════════════════════════════════════════════════════════════════════
nu_range = np.linspace(0.33, 0.70, 300)
kingdom_order_a = ['Bacteria','Viruses','Mammals','Plants','Fungi','Protists']

for k in kingdom_order_a:
    sub = bio6[bio6['kingdom']==k]['nu'].dropna()
    sub = sub[(sub >= 0.33) & (sub <= 0.72)]
    kde = gaussian_kde(sub, bw_method=0.12)
    ax_a.plot(nu_range, kde(nu_range), color=KC[k], lw=2.2, label=k)

ax_a.axvspan(0.50, 0.58, alpha=0.25, color='#f5e642', zorder=0)
ax_a.axvspan(0.44, 0.50, alpha=0.12, color='#999999', zorder=0)
ax_a.axvline(0.50, color='grey', lw=0.8, ls=':', alpha=0.7)
ax_a.axvline(0.58, color='grey', lw=0.8, ls=':', alpha=0.7)
ax_a.annotate('', xy=(0.50, 14.6), xytext=(0.58, 14.6),
              arrowprops=dict(arrowstyle='<->', color='#b09000', lw=1.4))
ax_a.text(0.54, 14.8, '\u03b8-solvent\ntransition',
          ha='center', fontsize=13, color='#b09000', style='italic', va='bottom',
          clip_on=False)

ax_a.set_xlabel('Scaling Exponent (\u03bd)', fontsize=FS_LABEL)
ax_a.set_ylabel('Density', fontsize=FS_LABEL)
ax_a.tick_params(labelsize=FS_TICK)
ax_a.set_xlim(0.33, 0.70); ax_a.set_ylim(0, 15.5)
ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)
handles = [plt.Line2D([0],[0],color=KC[k],lw=2.2,label=k) for k in kingdom_order_a]
ax_a.legend(handles=handles, fontsize=FS_LEGEND, ncol=2,
            loc='upper left', framealpha=0.0, handlelength=1.5)
ax_a.text(-0.08, 1.02, 'a', transform=ax_a.transAxes,
          fontsize=FS_PANEL, fontweight='bold', va='bottom')

# ═══════════════════════════════════════════════════════════════════════
# PANEL b: count histogram by nu bin
# ═══════════════════════════════════════════════════════════════════════
bins_edges = [0.33,0.40,0.45,0.50,0.53,0.56,0.58,0.60,0.65,0.70]
bin_labels  = ['0.33-\n0.40','0.40-\n0.45','0.45-\n0.50','0.50-\n0.53',
               '0.53-\n0.56','0.56-\n0.58','0.58-\n0.60','0.60-\n0.65','0.65-\n0.70']
kingdom_order_b = ['Bacteria','Plants','Mammals','Viruses','Fungi','Protists']
n_k = len(kingdom_order_b)
bar_w = 0.8 / n_k
offsets = np.linspace(-0.4+bar_w/2, 0.4-bar_w/2, n_k)

for ki, k in enumerate(kingdom_order_b):
    sub = bio6[bio6['kingdom']==k]['nu'].dropna()
    counts, _ = np.histogram(sub, bins=bins_edges)
    x = np.arange(len(bin_labels))
    ax_b.bar(x + offsets[ki], counts, bar_w*0.92, color=KC[k],
             label=k, alpha=0.88, zorder=3)

for xi in range(3, 6):   # theta-solvent bins 0.50-0.58
    ax_b.axvspan(xi-0.5, xi+0.5, alpha=0.18, color='#f5e642', zorder=0)

ax_b.set_xticks(np.arange(len(bin_labels)))
ax_b.set_xticklabels(bin_labels, fontsize=FS_TICK-1)
ax_b.set_ylabel('Count', fontsize=FS_LABEL)
ax_b.set_xlabel('\u03bd bin', fontsize=FS_LABEL)
ax_b.tick_params(axis='y', labelsize=FS_TICK)
ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
handles_b = [plt.Rectangle((0,0),1,1,color=KC[k],alpha=0.88,label=k)
             for k in kingdom_order_b]
ax_b.legend(handles=handles_b, fontsize=FS_LEGEND, ncol=1,
            loc='upper right', framealpha=0.0, handlelength=1.2, handletextpad=0.5)
ax_b.text(-0.08, 1.02, 'b', transform=ax_b.transAxes,
          fontsize=FS_PANEL, fontweight='bold', va='bottom')

# ═══════════════════════════════════════════════════════════════════════
# PANEL c: PAE concordance violins
# ═══════════════════════════════════════════════════════════════════════
kingdom_order_c = ['Bacteria','Viruses','Mammals','Plants','Fungi','Protists']
vcolors = [KC[k] for k in kingdom_order_c]

if pae_df is not None:
    vdata = [pae_df[pae_df['kingdom']==k]['pearson_r'].dropna().values
             for k in kingdom_order_c]
    global_median = pae_df['pearson_r'].median()
else:
    # Placeholder using known medians — replace with real data
    pae_medians = {'Bacteria':0.831,'Viruses':0.819,'Mammals':0.798,
                   'Plants':0.795,'Fungi':0.780,'Protists':0.779}
    pae_n = {'Bacteria':2850,'Viruses':1025,'Mammals':1361,
             'Plants':2480,'Fungi':1507,'Protists':1049}
    np.random.seed(42)
    vdata = []
    for k in kingdom_order_c:
        med = pae_medians[k]; n = pae_n[k]
        s = np.random.beta(8, 2, n) * 0.24 + (med - 0.12)
        s = np.clip(s, 0.30, 1.0)
        s = s + (med - np.median(s))
        vdata.append(s)
    global_median = 0.804

parts = ax_c.violinplot(vdata, positions=range(len(kingdom_order_c)),
                         widths=0.7, showmedians=False, showextrema=False)
for pc, col in zip(parts['bodies'], vcolors):
    pc.set_facecolor(col); pc.set_alpha(0.75); pc.set_edgecolor('white')

for i, (k, d) in enumerate(zip(kingdom_order_c, vdata)):
    med = np.median(d)
    ax_c.hlines(med, i-0.25, i+0.25, color='black', lw=2.5, zorder=5)
    ax_c.text(i, med+0.015, f'{med:.3f}', ha='center',
              fontsize=FS_ANNOT, fontweight='bold', zorder=6)
    n = len(d)
    trans_c = blended_transform_factory(ax_c.transData, ax_c.transAxes)
    ax_c.text(i, -0.14, f'n={n:,}', ha='center', fontsize=FS_TICK-2,
              color='#666666', va='top', transform=trans_c, clip_on=False)

ax_c.axhline(global_median, color='grey', ls='--', lw=1.2, alpha=0.8)
ax_c.text(0.98, 0.97, f'– – –  Global median = {global_median:.3f}',
          fontsize=FS_TICK-2, color='grey', va='top', ha='right',
          transform=ax_c.transAxes)
ax_c.set_xticks(range(len(kingdom_order_c)))
ax_c.set_xticklabels(kingdom_order_c, fontsize=FS_TICK)
ax_c.set_ylabel('Pearson r (SD$_{dist}$ vs PAE)', fontsize=FS_LABEL)
ax_c.set_ylim(0.30, 1.02)
ax_c.tick_params(axis='y', labelsize=FS_TICK)
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
ax_c.text(-0.08, 1.02, 'c', transform=ax_c.transAxes,
          fontsize=FS_PANEL, fontweight='bold', va='bottom')

# ═══════════════════════════════════════════════════════════════════════
# PANEL d: representative SD_dist + PAE contact maps
# Load real numpy arrays from HPC output
# ═══════════════════════════════════════════════════════════════════════
proteins = [('N9ZEX2','Bacteria'), ('Q2HAA5','Fungi'),
            ('Q8CF14','Mammals'),  ('Q0PFA9','Viruses')]

inner = gridspec.GridSpecFromSubplotSpec(
    2, 4, subplot_spec=gs[1,1], hspace=0.08, wspace=0.08)

cmap_sd  = 'magma'
cmap_pae = LinearSegmentedColormap.from_list(
    'pae', ['#00441b','#41ab5d','#d9f0d3','#ffffff'])

for col_i, (uid, kingdom) in enumerate(proteins):
    sd_file  = f'/Volumes/User Homes/Nu Project/BENDER-BIO/PAE/pae_analysis/sd_maps/{uid}_sd.npy'
    pae_file = f'/Volumes/User Homes/Nu Project/BENDER-BIO/PAE/pae_analysis/pae_maps/{uid}_pae.npy'

    if os.path.exists(sd_file) and os.path.exists(pae_file):
        sd_map  = np.load(sd_file)
        pae_map = np.load(pae_file)
    else:
        print(f"WARNING: maps for {uid} not found — using placeholder")
        n = 80
        sd_map = np.fromfunction(
            lambda i,j: 0.3 + 1.8*(1-np.exp(-np.abs(i-j)/20)), (n,n))
        pae_map = np.fromfunction(
            lambda i,j: 2 + 25*(1-np.exp(-np.abs(i-j)/15)), (n,n))

    r_val = (pae_df[pae_df['uniprot_id']==uid]['pearson_r'].values[0]
             if pae_df is not None else 0.805)

    ax_top = fig.add_subplot(inner[0, col_i])
    im1 = ax_top.imshow(sd_map, cmap=cmap_sd, vmin=0, vmax=2.2,
                         aspect='equal', interpolation='nearest')
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.set_title(f'{uid}\n({kingdom}, r = {r_val:.3f})',
                     fontsize=13, pad=3)
    if col_i == 3:
        cbar1 = fig.colorbar(im1, ax=ax_top, fraction=0.08, pad=0.02)
        cbar1.set_label('SD (nm)', fontsize=13)
        cbar1.ax.tick_params(labelsize=12)
    if col_i == 0:
        ax_top.set_ylabel('SD$_{dist}$', fontsize=14, rotation=0,
                          labelpad=38, va='center')

    ax_bot = fig.add_subplot(inner[1, col_i])
    im2 = ax_bot.imshow(pae_map, cmap=cmap_pae, vmin=0, vmax=32,
                         aspect='equal', interpolation='nearest')
    ax_bot.set_xticks([]); ax_bot.set_yticks([])
    if col_i == 3:
        cbar2 = fig.colorbar(im2, ax=ax_bot, fraction=0.08, pad=0.02)
        cbar2.set_label('PAE (Å)', fontsize=13)
        cbar2.ax.tick_params(labelsize=12)
    if col_i == 0:
        ax_bot.set_ylabel('PAE', fontsize=14, rotation=0,
                          labelpad=28, va='center')

ax_d.text(-0.08, 1.02, 'd', transform=ax_d.transAxes,
          fontsize=FS_PANEL, fontweight='bold', va='bottom')

# ── Save ──────────────────────────────────────────────────────────────
fig.savefig(OUT / 'fig1_dataset.pdf', dpi=200, bbox_inches='tight')
fig.savefig(OUT / 'fig1_dataset.png', dpi=200, bbox_inches='tight')
plt.close()
print(f\"Saved: {OUT}/fig1_dataset.pdf / fig1_dataset.png\")
