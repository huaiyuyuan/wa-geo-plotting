#!/usr/bin/env python3
"""
plot_misfit_compare.py - data misfit (RMS, Fvs key 'Misfit', one value per node) of two or
more TransD models: maps on one shared colour scale, histograms, node-by-node scatter
(B interpolated linearly to A's nodes - the two models need not share node locations - kept
where a B node lies within --match-deg) and per-domain medians.

  source env_stage6.sh
  python3 plotting/plot_misfit_compare.py --fvs $FVS2:"v2 iter2" $FVSV3:"v3 iter1" \\
      --sites examples/yilgarn_sites_19.txt --out $FIG/compare_v3_iter2/misfit.png

Caveat for the caption: the models fit different data (v2 band80 vs v3 band120 maps,
other period ranges / errors), so a lower misfit means 'fits its own data better', not
'is closer to the truth'; compare spatial patterns more than absolute levels.
Nodes with Ndat < --min-ndat are left out. Writes <out> (+ .pdf) and <stem>.txt (medians).
"""
import argparse, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'basemap'))
import tomo2d as t2                                           # noqa: E402

DOMAINS = ['wac', 'yilgarn', 'pilbara', 'capricorn', 'albany_fraser', 'perth', 'canning', 'officer', 'eucla']


def model_label(spec):
    path, _, label = spec.partition(':')
    if not label:
        m = re.search(r'iter\.?(\d+)', os.path.basename(path))
        label = f'iter{m.group(1)}' if m else os.path.basename(path).replace('.npz', '')
    return path, label


def load(spec, min_ndat):
    path, label = model_label(spec)
    d = np.load(path, allow_pickle=True)
    if 'Misfit' not in d.files:
        sys.exit(f"{os.path.basename(path)} has no 'Misfit' (keys: {', '.join(d.files)})")
    lon, lat = np.asarray(d['Lon'], float), np.asarray(d['Lat'], float)
    mis = np.asarray(d['Misfit'], float)
    if mis.ndim > 1:                                   # one value per node expected
        mis = np.nanmean(mis.reshape(len(lon), -1), axis=1)
    fin = np.isfinite(mis)
    keep = fin.copy()
    if 'Ndat' in d.files and min_ndat:
        nd = np.asarray(d['Ndat'], float)
        keep &= nd >= min_ndat
        if fin.any() and not keep.any():                # Ndat on another scale (e.g. v3)
            print(f"  ({label}: no node has Ndat >= {min_ndat} (Ndat {np.nanmin(nd):g}-{np.nanmax(nd):g}); "
                  f"Ndat filter skipped - set --min-ndat for this model's scale)")
            keep = fin
    if not fin.any():
        print(f"  ({label}: Misfit is all NaN/inf - nothing to plot)")
    if keep.any():
        print(f"{label}: {os.path.basename(path)}  {keep.sum()} nodes, misfit median {np.nanmedian(mis[keep]):.4f}, "
              f"10-90% {np.nanpercentile(mis[keep], 10):.4f}-{np.nanpercentile(mis[keep], 90):.4f}")
    return dict(label=label, lon=lon[keep], lat=lat[keep], mis=mis[keep])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs', nargs='+', required=True, metavar='PATH[:LABEL]')
    ap.add_argument('--min-ndat', type=int, default=35)
    ap.add_argument('--vlim', nargs=2, type=float, default=None, help='map colour limits (default 2-98%% of all)')
    ap.add_argument('--match-deg', type=float, default=0.3,
                    help='scatter: use A nodes with a B node within this distance (deg)')
    ap.add_argument('--sites', default=None, help='mark study sites on the maps')
    ap.add_argument('--density', action='store_true',
                    help='histogram as density (default: number of nodes per bin)')
    ap.add_argument('--domains', nargs='+', default=DOMAINS)
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--extent', default='112,129.6,-35.8,-13.2')
    ap.add_argument('--out', default='figures/compare/misfit.png')
    a = ap.parse_args()

    M = [load(s, a.min_ndat) for s in a.fvs]
    empty = [m['label'] for m in M if not len(m['mis'])]
    if empty:
        print(f"  left out (no finite Misfit): {', '.join(empty)}")
    M = [m for m in M if len(m['mis'])]
    if not M:
        sys.exit('no model with a finite Misfit')
    allm = np.concatenate([m['mis'] for m in M])
    lo, hi = a.vlim or np.nanpercentile(allm, [2, 98])
    norm = Normalize(lo, hi)
    cmap = matplotlib.colormaps['YlOrRd']
    ext = [float(x) for x in a.extent.split(',')]
    sites = t2.load_sites(a.sites) if a.sites else None

    n = len(M)
    fig = plt.figure(figsize=(4.2 * n + 4.4, 6.4))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [1.05], height_ratios=[1, 1],
                          wspace=0.22, hspace=0.32, left=0.04, right=0.98, top=0.9, bottom=0.16)
    for j, m in enumerate(M):
        ax = fig.add_subplot(gs[:, j])
        F = t2.MapFrame(ax, ext)
        x, y = F.xy(m['lon'], m['lat'])
        sc = ax.scatter(x, y, c=m['mis'], s=9, cmap=cmap, norm=norm, lw=0, zorder=3)
        if sites:
            sx, sy = F.xy(np.array([s['lon'] for s in sites]), np.array([s['lat'] for s in sites]))
            ax.scatter(sx, sy, marker='^', s=28, facecolors='none', edgecolors='k', lw=0.7, zorder=7)
        ax.set_title(f"{m['label']}  (median {np.nanmedian(m['mis']):.3f})", fontsize=10, loc='left')
        if j == n - 1:                                # one bar under all maps
            x0 = gs[:, 0].get_position(fig).x0; x1 = ax.get_position().x1
            cax = fig.add_axes([x0 + 0.15 * (x1 - x0), 0.085, 0.7 * (x1 - x0), 0.022])
            cb = fig.colorbar(sc, cax=cax, orientation='horizontal', extend='both')
            cb.set_label('RMS data misfit (Fvs Misfit)', fontsize=8.5)
    # histograms
    axh = fig.add_subplot(gs[0, n])
    bins = np.linspace(lo, hi, 40)
    for k, m in enumerate(M):
        axh.hist(np.clip(m['mis'], lo, hi), bins=bins, histtype='step', lw=1.6,
                 label=f"{m['label']} ({len(m['mis'])} nodes)", color=f'C{k}', density=a.density)
    axh.set_xlabel('RMS misfit (km/s)', fontsize=8.5)
    axh.set_ylabel('density' if a.density else 'number of nodes', fontsize=8.5)
    axh.legend(fontsize=8, frameon=False); axh.tick_params(labelsize=7.5)
    # scatter: B (interpolated to A's nodes) against A
    axs = fig.add_subplot(gs[1, n])
    if n >= 2:
        from scipy.spatial import cKDTree
        A, B = M[0], M[1]
        from scipy.interpolate import LinearNDInterpolator
        # different node sets: B interpolated (linear, in its triangulation) at A's nodes
        Bi = LinearNDInterpolator(np.column_stack([B['lon'], B['lat']]), B['mis'])(A['lon'], A['lat'])
        dist, _ = cKDTree(np.column_stack([B['lon'], B['lat']])).query(np.column_stack([A['lon'], A['lat']]))
        ok = (dist <= a.match_deg) & np.isfinite(Bi)
        idx = np.arange(len(A['lon']))                  # (B values now on A's nodes)
        Bv = Bi
        axs.scatter(A['mis'][ok], Bv[ok], s=4, c='0.3', lw=0, alpha=0.6)
        lim = [lo, hi]
        axs.plot(lim, lim, 'k--', lw=0.8)
        axs.set_xlim(lim); axs.set_ylim(lim)
        r = np.corrcoef(A['mis'][ok], Bv[ok])[0, 1] if ok.sum() > 2 else np.nan
        better = np.mean(Bv[ok] < A['mis'][ok]) * 100 if ok.any() else np.nan
        axs.set_title(f"{ok.sum()} {A['label']} nodes, {B['label']} interpolated: r = {r:.2f}, "
                      f"{B['label']} lower at {better:.0f}%", fontsize=8)
        axs.set_xlabel(f"{A['label']} misfit", fontsize=8.5); axs.set_ylabel(f"{B['label']} misfit", fontsize=8.5)
    else:
        axs.axis('off')
    axs.tick_params(labelsize=7.5)
    fig.suptitle('TransD data misfit per node (lower = better fit to its own dispersion data; '
                 'models fit different data sets)', fontsize=11)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t2.source_note(fig, [model_label(s)[0] for s in a.fvs])
    fig.savefig(a.out, dpi=220); fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")

    # per-domain medians
    stem = os.path.splitext(a.out)[0]
    try:
        from domains import Domains
        sys.path.insert(0, os.path.join(_HERE, '..', 'extract'))
        from export_domains import all_definitions
        D = Domains(a.tect_npz)
        D.defs = all_definitions(True)
        with open(stem + '.txt', 'w') as f:
            f.write('# median RMS misfit per domain (n nodes)\n# domain ' +
                    ' '.join(f"{m['label'].replace(' ', '_')}" for m in M) + '\n')
            for k in a.domains:
                if k not in D.defs:
                    continue
                cells = []
                for m in M:
                    sel = D.contains(k, m['lon'], m['lat'])
                    cells.append(f"{np.nanmedian(m['mis'][sel]):.4f}({sel.sum()})" if sel.any() else 'nan(0)')
                f.write(f"{k:<14s} " + ' '.join(cells) + '\n')
                print(f"  {D[k]['label']:<24s} " + '  '.join(cells))
        print(f"Table: {stem}.txt")
    except Exception as e:
        print(f"  (no per-domain table: {e})")


if __name__ == '__main__':
    main()
