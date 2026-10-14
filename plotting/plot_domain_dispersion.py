#!/usr/bin/env python3
"""
plot_domain_dispersion.py - per-domain mean dispersion curves from the v3b (band120)
iter-2 group / phase velocity maps (TransD.region.{group,phase}.npz): Rayleigh (Z) and
Love (T), phase and group, averaged over the map cells of each tectonic domain (GSWA
1:10M units, basemap/domains.csv) that lie inside the figure mask at that period.

  source env_stage6.sh
  python3 plotting/plot_domain_dispersion.py --root $V3B --out $FIG/v3b/domain_dispersion.png

Mask per period = coverage (footprint 'density' >= 10, holes filled, largest region) AND
posterior std <= 0.80 (finite maski) AND finite velocity - the same cells as the maps of
plot_dispersion_maps.py, so node counts change with period. A domain's point is dropped
where it has fewer than --min-nodes cells. Domains overlap (the WAC contains the Yilgarn,
Pilbara, Capricorn, Albany-Fraser); each is averaged on its own.

Figure: one column per set (Z phase, T phase, Z group, T group); rows = mean velocity
(+-1 sigma spatial spread), dV (%) vs the --ref-domain (WAC) mean, cells per domain.
Grey bands: low-confidence / smooth-by-construction period ranges (tomo2d.CAVEATS).
Tables: <out>.csv (set, domain, period, n, mean, std, median, mean posterior std,
dV vs ref) and <out>.npz.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', 'basemap'))
sys.path.insert(0, os.path.join(_HERE, '..', 'extract'))
import tomo2d as t2                                           # noqa: E402

DEFAULT_DOMAINS = ['wac', 'yilgarn', 'pilbara', 'capricorn', 'albany_fraser',
                   'perth', 'canning', 'officer', 'eucla']


def domain_stats(T, fm, dmask, min_nodes):
    """{stat: [nper]} over cells in fm[p] & dmask."""
    nper = len(T['per'])
    out = {k: np.full(nper, np.nan) for k in ('mean', 'std', 'median', 'post_std')}
    out['n'] = np.zeros(nper, int)
    for p in range(nper):
        m = fm[p] & dmask
        n = int(m.sum())
        out['n'][p] = n
        if n >= max(1, min_nodes):
            v = T['v'][p][m]
            out['mean'][p], out['std'][p], out['median'][p] = v.mean(), v.std(), np.median(v)
            s = T['std'][p][m]
            out['post_std'][p] = np.nanmean(s) if np.isfinite(s).any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.environ.get('V3B'), help='v3b dir (default $V3B)')
    ap.add_argument('--npz', nargs='+', default=None, help='explicit TransD.region.*.npz files')
    ap.add_argument('--sets', nargs='+', default=['all'], help='Z.phase T.phase Z.group T.group | all')
    ap.add_argument('--domains', nargs='+', default=DEFAULT_DOMAINS)
    ap.add_argument('--ref-domain', default='wac', help="dV reference: domain key or 'mask'")
    ap.add_argument('--min-nodes', type=int, default=5)
    ap.add_argument('--footprint', default=None)
    ap.add_argument('--density-min', type=float, default=10.0)
    ap.add_argument('--std-max', type=float, default=None)
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--no-band', action='store_true', help='no +-1 sigma bands')
    ap.add_argument('--no-all', action='store_true', help="no whole-mask curve")
    ap.add_argument('--tag', default='v3b')
    ap.add_argument('--out', default='figures/v3b/domain_dispersion.png')
    a = ap.parse_args()

    if a.npz:
        sets = [t2.set_from_path(p) + (p,) for p in a.npz]
    elif a.root:
        sets = t2.find_sets(a.root, t2.parse_sets(a.sets))
    else:
        sys.exit('give --root (v3b dir) or --npz')
    if not sets:
        sys.exit('no input npz found')

    from domains import Domains
    from export_domains import all_definitions
    D = Domains(a.tect_npz)
    D.defs = all_definitions(True)
    keys = [k for k in a.domains if k in D.defs]
    for k in set(a.domains) - set(keys):
        print(f"  (domain '{k}' not in domains.csv - skipped)")
    if a.ref_domain not in ('mask',) and a.ref_domain not in keys:
        keys.append(a.ref_domain)

    res, rows = {}, []
    for comp, typ, path in sets:
        T = t2.load_tomo(path, comp, typ)
        cov, cdesc = t2.load_coverage(T['lat'], T['lon'], a.footprint, thr=a.density_min)
        fm = t2.figure_masks(T, cov, a.std_max)
        print(f"{comp} {typ}: {len(T['per'])} periods {T['per'].min():g}-{T['per'].max():g} s; "
              f"coverage {cov.sum()} cells")
        S = {}
        if not a.no_all:
            S['mask'] = domain_stats(T, fm, np.ones_like(cov), a.min_nodes)
        for k in keys:
            dm = D.contains(k, T['LON'].ravel(), T['LAT'].ravel()).reshape(cov.shape)
            S[k] = domain_stats(T, fm, dm, a.min_nodes)
            n = S[k]['n']
            print(f"   {D[k]['label']:<24s} cells in mask: {n.min():4d}-{n.max():4d}"
                  + ('' if n.max() >= a.min_nodes else '  (below --min-nodes: not plotted)'))
        ref = S.get(a.ref_domain) if a.ref_domain != 'mask' else domain_stats(
            T, fm, np.ones_like(cov), a.min_nodes)
        for k, s in S.items():
            s['dv'] = 100.0 * (s['mean'] - ref['mean']) / ref['mean']
            for p in range(len(T['per'])):
                rows.append((comp, typ, k, T['per'][p], s['n'][p], s['mean'][p], s['std'][p],
                             s['median'][p], s['post_std'][p], s['dv'][p]))
        res[(comp, typ)] = (T['per'], S)

    # ---------------------------------------------------------------- figure
    order = [s for s in t2.SETS if s in res]
    nc = len(order)
    fig = plt.figure(figsize=(3.3 * nc + 2.2, 8.4))
    gs = fig.add_gridspec(3, nc, height_ratios=[3.0, 2.0, 1.2], hspace=0.08, wspace=0.28,
                          left=0.06, right=1 - 2.1 / (3.3 * nc + 2.2), top=0.93, bottom=0.07)
    ref_lab = 'masked-area' if a.ref_domain == 'mask' else D[a.ref_domain]['label']

    def style(k):
        if k == 'mask':
            return dict(color='0.45', ls='--', lw=1.2), 'all masked cells'
        c = D[k]['colour']
        return dict(color=c, ls='-', lw=2.0 if k == a.ref_domain else 1.4), D[k]['label']

    handles = {}
    for c, (comp, typ) in enumerate(order):
        per, S = res[(comp, typ)]
        axv = fig.add_subplot(gs[0, c])
        axd = fig.add_subplot(gs[1, c], sharex=axv)
        axn = fig.add_subplot(gs[2, c], sharex=axv)
        for k, s in S.items():
            kw, lab = style(k)
            ok = s['n'] >= max(1, a.min_nodes)
            if not ok.any():
                continue
            mean = np.where(ok, s['mean'], np.nan)
            h, = axv.plot(per, mean, marker='o', ms=2.2, **kw)
            handles[k] = (h, lab)
            if not a.no_band and k != 'mask':
                axv.fill_between(per, mean - s['std'], mean + s['std'], color=kw['color'],
                                 alpha=0.08, lw=0)
            axd.plot(per, np.where(ok, s['dv'], np.nan), marker='o', ms=2.0, **kw)
            axn.plot(per, s['n'], **{**kw, 'lw': 1.0})
        axd.axhline(0, color='k', lw=0.5)
        for ax in (axv, axd, axn):
            for lo, hi, txt in t2.CAVEATS.get((comp, typ), []):
                ax.axvspan(lo, hi, color='0.85', zorder=0, lw=0)
            ax.set_xscale('log')
            ax.grid(True, which='major', lw=0.3, color='0.8')
            ax.tick_params(labelsize=7.5)
        for lo, hi, txt in t2.CAVEATS.get((comp, typ), []):
            axv.text(np.sqrt(lo * hi), 0.02, 'low conf.' if 'low' in txt else 'smooth',
                     transform=axv.get_xaxis_transform(), rotation=90, fontsize=6.5,
                     ha='center', va='bottom', color='0.35')
        axn.xaxis.set_major_locator(mticker.FixedLocator([2, 3, 5, 10, 20, 30, 40, 60]))
        axn.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
        axn.xaxis.set_minor_formatter(mticker.NullFormatter())
        axn.set_xlim(per.min() * 0.92, per.max() * 1.08)
        plt.setp(axv.get_xticklabels() + axd.get_xticklabels(), visible=False)
        axv.set_title(f"{t2.WAVE.get(comp, comp)} ({comp}) {typ}", fontsize=10)
        axn.set_xlabel('period (s)', fontsize=8.5)
        axn.set_yscale('log')
        axn.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
        if c == 0:
            axv.set_ylabel('mean velocity (km/s)\n(band: ±1σ across cells)', fontsize=8.5)
            axd.set_ylabel(f'dV vs {ref_lab}\nmean (%)', fontsize=8.5)
            axn.set_ylabel('cells in mask', fontsize=8.5)
    hl = [handles[k] for k in (['mask'] + keys) if k in handles]
    fig.legend([h for h, _ in hl], [l for _, l in hl], loc='center right', fontsize=8,
               frameon=False, bbox_to_anchor=(0.995, 0.6))
    fig.text(1 - 2.05 / (3.3 * nc + 2.2), 0.2,
             'cells: map cells of the domain inside\ncoverage (density >= 10) and\n'
             'posterior std <= 0.80 at that period;\n'
             f'points with < {a.min_nodes} cells dropped.\n'
             'grey: low-confidence / smooth\nperiod ranges.\n' + t2.DROPPED.replace(' (', '\n('),
             fontsize=6.8, va='top', color='0.25')
    fig.suptitle(f"{a.tag}: mean dispersion curves of tectonic domains (GSWA 1:10M units)",
                 fontsize=11.5)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=250)
    fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")

    stem = os.path.splitext(a.out)[0]
    with open(stem + '.csv', 'w') as f:
        f.write(f"# {a.tag} per-domain dispersion; mask = coverage (density>={a.density_min:g}, holes "
                f"filled, largest region) & std<=0.80 & finite; dv_pct vs {a.ref_domain} mean\n")
        f.write('comp,type,domain,period_s,n_cells,mean_kms,std_kms,median_kms,mean_post_std,dv_pct\n')
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:g},{r[4]},"
                    + ','.join('' if not np.isfinite(x) else f"{x:.4f}" for x in r[5:]) + '\n')
    arr = {}
    for (comp, typ), (per, S) in res.items():
        arr[f'{comp}_{typ}_period'] = per
        for k, s in S.items():
            for q in ('n', 'mean', 'std', 'median', 'post_std', 'dv'):
                arr[f'{comp}_{typ}_{k}_{q}'] = s[q]
    np.savez(stem + '.npz', **arr)
    print(f"Saved {stem}.csv, {stem}.npz")


if __name__ == '__main__':
    main()
