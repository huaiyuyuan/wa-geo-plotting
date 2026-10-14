#!/usr/bin/env python3
"""
plot_site_dispersion.py - dispersion curves at study sites (e.g. the 19 Yilgarn sites in
examples/yilgarn_sites_19.txt) read from the v3b (band120) iter-2 group / phase velocity
maps (TransD.region.{group,phase}.npz): Rayleigh (Z) and Love (T), phase and group.

  source env_stage6.sh
  python3 plotting/plot_site_dispersion.py --root $V3B --sites examples/yilgarn_sites_19.txt \\
      --out $FIG/v3b/sites/site_dispersion.png

Value at a site = bilinear interpolation of the map at that period. A period counts as
'in mask' when the map cell nearest the site is inside the figure mask of the maps
(coverage density >= 10, holes filled, largest region; posterior std <= 0.80): filled
symbols; outside: hollow symbols on a thin line (kept, but not to be interpreted).

Outputs
  <out>               one panel per site, the four curves (blue Rayleigh, red Love;
                      solid phase, dashed group)
  <stem>.overlay.png  one panel per set, all sites coloured by terrane (in-mask values only)
  <stem>.csv          long table: site, terrane, lat, lon, comp, type, period, v, std2, in_mask
  <stem>.<Z|T>_<phase|group>.csv   wide tables: period x site, NaN outside the mask
                      (--keep-masked writes every value; the long table always has them)
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))
import tomo2d as t2                                           # noqa: E402

SET_STYLE = {('Z', 'phase'): dict(color='#1f4e9c', ls='-'), ('Z', 'group'): dict(color='#1f4e9c', ls='--'),
             ('T', 'phase'): dict(color='#c0392b', ls='-'), ('T', 'group'): dict(color='#c0392b', ls='--')}
XT = [2, 3, 5, 10, 20, 30, 40, 60]


def _logx(ax, pmin, pmax, labels=True):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(mticker.FixedLocator(XT))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g') if labels else mticker.NullFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(pmin * 0.9, pmax * 1.1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.environ.get('V3B'), help='v3b dir (default $V3B)')
    ap.add_argument('--npz', nargs='+', default=None, help='explicit TransD.region.*.npz files')
    ap.add_argument('--sets', nargs='+', default=['all'])
    ap.add_argument('--sites', default=os.path.join(_HERE, '..', 'examples', 'yilgarn_sites_19.txt'))
    ap.add_argument('--footprint', default=None)
    ap.add_argument('--density-min', type=float, default=10.0)
    ap.add_argument('--std-max', type=float, default=None)
    ap.add_argument('--keep-masked', action='store_true', help='wide tables: values outside the mask too')
    ap.add_argument('--ylim', nargs=2, type=float, default=None)
    ap.add_argument('--ncol', type=int, default=5)
    ap.add_argument('--tag', default='v3b')
    ap.add_argument('--out', default='figures/v3b/sites/site_dispersion.png')
    a = ap.parse_args()

    if a.npz:
        sets = [t2.set_from_path(p) + (p,) for p in a.npz]
    elif a.root:
        sets = t2.find_sets(a.root, t2.parse_sets(a.sets))
    else:
        sys.exit('give --root (v3b dir) or --npz')
    sites = t2.load_sites(a.sites)
    slat = np.array([s['lat'] for s in sites]); slon = np.array([s['lon'] for s in sites])
    print(f"{len(sites)} sites from {a.sites}")

    R = {}                                     # (comp, typ) -> dict(per, v[nsite,nper], std, inm)
    for comp, typ, path in sets:
        T = t2.load_tomo(path, comp, typ)
        cov, _ = t2.load_coverage(T['lat'], T['lon'], a.footprint, thr=a.density_min)
        fm = t2.figure_masks(T, cov, a.std_max)
        ii, jj = t2.nearest_cell(T['lat'], T['lon'], slat, slon)
        n = len(T['per'])
        v = np.array([t2.sample_grid(T['lat'], T['lon'], T['v'][p], slat, slon) for p in range(n)]).T
        s = np.array([t2.sample_grid(T['lat'], T['lon'], T['std'][p], slat, slon) for p in range(n)]).T
        inm = np.array([fm[p][ii, jj] for p in range(n)]).T
        R[(comp, typ)] = dict(per=T['per'], v=v, std=s, inm=inm)
        print(f"  {comp} {typ}: {n} periods; sites with every period in the mask: "
              f"{int(inm.all(1).sum())}/{len(sites)}")
        for k in np.where(~inm.any(1))[0]:
            print(f"     {sites[k]['no']} {sites[k]['name']}: outside the mask at all periods")
    order = [k for k in t2.SETS if k in R]
    pmin = min(R[k]['per'].min() for k in order); pmax = max(R[k]['per'].max() for k in order)
    allv = np.concatenate([R[k]['v'].ravel() for k in order])
    ylim = a.ylim or (np.floor(np.nanmin(allv) * 10 - 1) / 10, np.ceil(np.nanmax(allv) * 10 + 1) / 10)

    # ---- per-site panels
    ns = len(sites); nc = a.ncol; nr = int(np.ceil((ns + 1) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc + 0.4, 2.15 * nr + 0.6), sharex=True, sharey=True,
                            squeeze=False)
    for k, site in enumerate(sites):
        ax = axs.flat[k]
        for key in order:
            r, st = R[key], SET_STYLE[key]
            per, v, m = r['per'], r['v'][k], r['inm'][k]
            ax.plot(per, v, color=st['color'], ls=st['ls'], lw=0.6, alpha=0.5)
            ax.plot(per, np.where(m, v, np.nan), color=st['color'], ls=st['ls'], lw=1.4)
            ax.plot(per[m], v[m], 'o', ms=2.2, color=st['color'])
            ax.plot(per[~m], v[~m], 'o', ms=2.6, mfc='w', mec=st['color'], mew=0.6)
        ax.set_title(f"{site['no']} {site['name']}", fontsize=8.5, loc='left', pad=2,
                     color=t2.terrane_colour(site['terrane']), fontweight='bold')
        ax.text(0.98, 0.04, site['terrane'], transform=ax.transAxes, ha='right', va='bottom',
                fontsize=6.5, color='0.35')
        ax.grid(True, lw=0.3, color='0.85')
        ax.tick_params(labelsize=7)
        _logx(ax, pmin, pmax)
        ax.set_ylim(*ylim)
    for ax in list(axs.flat)[ns:]:
        ax.axis('off')
    lax = list(axs.flat)[ns] if ns < nr * nc else None
    h = [Line2D([], [], color=SET_STYLE[k]['color'], ls=SET_STYLE[k]['ls'], lw=1.4,
                label=f"{t2.WAVE[k[0]]} ({k[0]}) {k[1]}") for k in order]
    h += [Line2D([], [], ls='', marker='o', ms=4, color='0.3', label='inside map mask'),
          Line2D([], [], ls='', marker='o', ms=4, mfc='w', mec='0.3', label='outside mask')]
    if lax is not None:
        lax.legend(handles=h, loc='upper left', fontsize=7.5, frameon=False, bbox_to_anchor=(0, 1.08))
        lax.text(0.02, -0.02, 'mask: coverage density >= 10\n& posterior std <= 0.80\n(nearest map cell)\n'
                 'low confidence: Z group 32-40 s,\nT group ~30-32 s; Z phase 52-60 s\nsmooth by construction',
                 transform=lax.transAxes, fontsize=6.5, va='top', color='0.3')
    for ax in axs[:, 0]:
        ax.set_ylabel('velocity (km/s)', fontsize=8)
    for ax in axs[-1, :]:
        ax.set_xlabel('period (s)', fontsize=8)
    for c in range(nc):                       # x labels on the lowest used panel of each column
        col = [axs[r, c] for r in range(nr) if r * nc + c < ns]
        if col:
            col[-1].xaxis.set_tick_params(labelbottom=True)
            col[-1].set_xlabel('period (s)', fontsize=8)
    fig.suptitle(f"{a.tag}: dispersion at the study sites (from the group/phase velocity maps)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=220); fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")

    # ---- overlay: one panel per set, sites coloured by terrane
    stem = os.path.splitext(a.out)[0]
    fig, axs = plt.subplots(1, len(order), figsize=(3.4 * len(order) + 1.8, 4.2), squeeze=False)
    terr = []
    for c, key in enumerate(order):
        ax = axs[0, c]; r = R[key]
        for k, site in enumerate(sites):
            col = t2.terrane_colour(site['terrane'], k)
            m = r['inm'][k]
            ax.plot(r['per'], np.where(m, r['v'][k], np.nan), color=col, lw=1.1, alpha=0.9)
            if site['terrane'] not in terr:
                terr.append(site['terrane'])
        for lo, hi, txt in t2.CAVEATS.get(key, []):
            ax.axvspan(lo, hi, color='0.88', zorder=0, lw=0)
        _logx(ax, r['per'].min(), r['per'].max())
        ax.set_title(f"{t2.WAVE[key[0]]} ({key[0]}) {key[1]}", fontsize=10)
        ax.set_xlabel('period (s)', fontsize=9); ax.grid(True, lw=0.3, color='0.85')
        ax.tick_params(labelsize=8)
        if c == 0:
            ax.set_ylabel('velocity (km/s)', fontsize=9)
    hh = [Line2D([], [], color=t2.terrane_colour(t, i), lw=2, label=t) for i, t in enumerate(terr)]
    fig.legend(handles=hh, loc='center right', fontsize=8, frameon=False, title='terrane',
               title_fontsize=8.5)
    fig.suptitle(f"{a.tag}: {len(sites)} study sites, in-mask values only (grey: low-confidence ranges)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1 - 1.7 / (3.4 * len(order) + 1.8), 0.95))
    fig.savefig(stem + '.overlay.png', dpi=220); fig.savefig(stem + '.overlay.pdf')
    plt.close(fig)
    print(f"Saved {stem}.overlay.png (+ .pdf)")

    # ---- tables
    with open(stem + '.csv', 'w') as f:
        f.write(f"# {a.tag} dispersion at study sites ({os.path.basename(a.sites)}); v = bilinear from "
                f"the map; in_mask = nearest cell inside coverage & std<=0.80\n")
        f.write('site_no,name,terrane,lat,lon,comp,type,period_s,v_kms,std2,in_mask\n')
        for key in order:
            r = R[key]
            for k, s in enumerate(sites):
                for p, P in enumerate(r['per']):
                    f.write(f"{s['no']},{s['name']},{s['terrane']},{s['lat']:g},{s['lon']:g},{key[0]},{key[1]},"
                            f"{P:g},{r['v'][k, p]:.4f},{r['std'][k, p]:.4f},{int(r['inm'][k, p])}\n")
    for key in order:
        r = R[key]
        fn = f"{stem}.{key[0]}_{key[1]}.csv"
        with open(fn, 'w') as f:
            f.write(f"# {a.tag} {t2.WAVE[key[0]]} ({key[0]}) {key[1]} velocity (km/s) at study sites; "
                    + ("all values" if a.keep_masked else "NaN = outside the map mask") + "\n")
            f.write('period_s,' + ','.join(f"{s['no']}_{s['name']}" for s in sites) + '\n')
            for p, P in enumerate(r['per']):
                f.write(f"{P:g}," + ','.join(f"{r['v'][k, p]:.4f}" if (r['inm'][k, p] or a.keep_masked)
                                              else 'NaN' for k in range(len(sites))) + '\n')
    print(f"Tables: {stem}.csv + {stem}.<set>.csv")


if __name__ == '__main__':
    main()
