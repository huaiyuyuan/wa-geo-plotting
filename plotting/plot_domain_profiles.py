#!/usr/bin/env python3
"""
plot_domain_profiles.py - representative 1-D profiles of tectonic domains (Yilgarn,
West Australian Craton, Perth Basin, Albany-Fraser, Capricorn; see basemap/domains.csv).

For each model and domain: mean over the domain's nodes at each depth, with the lateral
spread (+/-1 sigma, or 16-84 % with --spread pct) as a band. One panel per quantity the
models carry - Vsv (true Vsv for xi models), Viso, xi, Vp/Vs - several models side by
side as line styles. A map shows which nodes went into each domain. Optional AR23 Moho:
domain mean as a short bar on the right edge of each panel.

  python3 plotting/plot_domain_profiles.py --fvs $FVS8:iter8 $FVS11:iter11 \\
      --moho $MOHO --d-max 60 --out $FIG/domains/domain_profiles.png

Writes <out>.png/.pdf and, per model, <out stem>.<label>.txt (depth, mean, std, p16, p50,
p84, n for every domain and field) and .npz (same arrays) - the Yilgarn means there are
what --ref-domain yilgarn uses as the dln reference in the maps and sections.
"""
import argparse, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
for _p in (os.path.join(_HERE, '..', 'basemap'), os.path.join(_HERE, '..')):
    if os.path.abspath(_p) not in map(os.path.abspath, sys.path):
        sys.path.insert(1, os.path.abspath(_p))
from domains import Domains, model_fields                     # noqa: E402

# the tectonic regions of the v3b domain dispersion curves (plot_domain_dispersion.py)
DISPERSION_SET = ['wac', 'yilgarn', 'pilbara', 'capricorn', 'albany_fraser', 'perth',
                  'canning', 'officer', 'eucla']

FIELD_ORDER = ('vsv', 'viso', 'xi', 'vpvs')
FIELD_LABEL = {'vsv': 'Vsv (km/s)', 'viso': 'Viso (km/s)', 'xi': 'xi = Vsh/Vsv', 'vpvs': 'Vp/Vs'}
STYLES = ['-', '--', ':', '-.']


def parse_model(spec):
    path, _, label = spec.partition(':')
    if not label:
        m = re.search(r'iter\.?(\d+)', os.path.basename(path))
        label = f'iter{m.group(1)}' if m else os.path.basename(path).replace('.npz', '')
    return path, label


def stats(A, spread):
    out = dict(mean=np.nanmean(A, 0), std=np.nanstd(A, 0), p16=np.nanpercentile(A, 16, 0),
               p50=np.nanmedian(A, 0), p84=np.nanpercentile(A, 84, 0))
    out['lo'], out['hi'] = ((out['mean'] - out['std'], out['mean'] + out['std'])
                            if spread == 'sigma' else (out['p16'], out['p84']))
    return out


def moho_at(path, lon, lat):
    from scipy.interpolate import RectBivariateSpline
    md = np.loadtxt(path, skiprows=11)
    mla, mlo = np.unique(md[:, 0]), np.unique(md[:, 1])
    f = RectBivariateSpline(mla, mlo, md[:, 2].reshape(len(mla), len(mlo)), kx=1, ky=1)
    return np.array([float(np.ravel(f(a, o))[0]) for a, o in zip(lat, lon)])


def draw_map(ax, D, keys, models, masks, tr, cov=None, sites=None, veil_alpha=0.6, veil_colour='white',
             stations=None, show_nodes=True):
    from plot_depth_slice import _add_states_ocean
    import tomo2d as t2
    for k in keys:                                   # unit outlines, filled faintly
        col = D[k]['colour']
        for unit, idx in D.unit_rings(k).items():
            for i in idx:
                r = D.rings[i]
                kw = {'transform': tr} if tr else {}
                ax.fill(r[:, 0], r[:, 1], color=col, alpha=0.10, lw=0, zorder=1, **kw)
                ax.plot(r[:, 0], r[:, 1], color=col, lw=0.6, alpha=0.8, zorder=2, **kw)
    d0 = models[0][2]
    lon, lat = np.asarray(d0['Lon']), np.asarray(d0['Lat'])
    kw = {'transform': tr} if tr else {}
    if show_nodes:                                   # model nodes, coloured by domain
        ax.scatter(lon, lat, s=2, c='0.75', lw=0, zorder=3, **kw)
        for k in sorted(keys, key=lambda k: -masks[(models[0][1], k)].sum()):   # big first
            m = masks[(models[0][1], k)]
            ax.scatter(lon[m], lat[m], s=5, c=D[k]['colour'], lw=0, zorder=4, **kw)
    if stations is not None and len(stations[0]):   # real stations (name lat lon ...)
        ax.scatter(stations[0], stations[1], marker='^', s=7, c='k', edgecolors='w', linewidths=0.2,
                   zorder=6, **kw)
    if cov is not None:                              # fade the land outside the coverage mask
        from matplotlib.colors import ListedColormap
        la, lo, cm = cov
        ax.pcolormesh(t2.edges(lo), t2.edges(la), np.ma.masked_where(cm, np.ones(cm.shape)),
                      cmap=ListedColormap([veil_colour]), alpha=veil_alpha, shading='flat',
                      zorder=3.5, rasterized=True, **kw)
        LO, LA = np.meshgrid(np.r_[2 * lo[0] - lo[1], lo, 2 * lo[-1] - lo[-2]],
                             np.r_[2 * la[0] - la[1], la, 2 * la[-1] - la[-2]])
        ax.contour(LO, LA, np.pad(cm.astype(float), 1), levels=[0.5], colors='0.25',
                   linewidths=0.8, linestyles='--', zorder=4.5, **kw)
    if sites:
        import matplotlib.patheffects as pe
        xyc = tr._as_mpl_transform(ax) if tr is not None else 'data'
        for k, st in enumerate(sites):
            ax.scatter([st['lon']], [st['lat']], marker='^', s=42, c=[t2.terrane_colour(st['terrane'], k)],
                       edgecolors='k', linewidths=0.6, zorder=8, **kw)
            ax.annotate(str(st['no']), (st['lon'], st['lat']), xycoords=xyc, xytext=(3.5, 2.5),
                        textcoords='offset points', fontsize=6.3, fontweight='bold', zorder=9,
                        path_effects=[pe.withStroke(linewidth=1.6, foreground='w')])
    _add_states_ocean(ax, ocean='white', states='#e8e8e8')
    if tr is not None:
        ax.set_extent([112.5, 129.5, -35.8, -13.5], crs=tr)
    else:
        ax.set_xlim(112.5, 129.5); ax.set_ylim(-35.8, -13.5); ax.set_aspect('equal')
    ax.set_title(f'Nodes per domain ({models[0][1]})' if show_nodes else
                 f'Domains and stations ({len(stations[0]) if stations is not None else 0})', fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs', nargs='+', required=True, metavar='PATH[:LABEL]')
    ap.add_argument('--domains', nargs='+', default=None,
                    help="domain keys from basemap/domains.csv (default: the active ones; "
                         "commented-out ones may be named too); 'dispersion' = the nine regions "
                         "of plot_domain_dispersion.py")
    ap.add_argument('--fields', nargs='+', default=None, choices=FIELD_ORDER,
                    help='panels (default: every quantity any model carries)')
    ap.add_argument('--d-max', type=float, default=60.0)
    ap.add_argument('--min-ndat', type=int, default=35)
    ap.add_argument('--spread', default='sigma', choices=['sigma', 'pct'])
    ap.add_argument('--moho', default=None, help='AR23 Moho: domain mean on each panel')
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--no-map', action='store_true')
    ap.add_argument('--footprint', action='store_true', help='only nodes inside the array footprint')
    ap.add_argument('--coverage', action='store_true',
                    help="only nodes inside the v3b coverage mask (footprint 'density' >= 10, holes "
                         "filled, largest region) - the cells of the dispersion maps/curves")
    ap.add_argument('--density-min', type=float, default=10.0)
    ap.add_argument('--no-veil', action='store_true',
                    help='with --coverage: do not fade the map outside the coverage mask')
    ap.add_argument('--veil-alpha', type=float, default=0.6)
    ap.add_argument('--veil-colour', default='white')
    ap.add_argument('--sites', default=None, help='study sites to mark on the map (e.g. examples/yilgarn_sites_19.txt)')
    ap.add_argument('--stations', default=None,
                    help="real stations on the map instead of the model-node dots ('name lat lon ...' file, "
                         "e.g. stations.phases.1.2.3.txt)")
    ap.add_argument('--show-nodes', action='store_true', help='with --stations: keep the node dots too')
    ap.add_argument('--rel-to', default=None, metavar='DOMAIN',
                    help='extra panel: d(first field) in %% relative to this domain mean (e.g. wac)')
    ap.add_argument('--footprint-npz', default=None)
    ap.add_argument('--footprint-outline', default=None)
    ap.add_argument('--out', default='figures/domains/domain_profiles.png')
    a = ap.parse_args()

    D = Domains(a.tect_npz)
    if a.domains:
        sys.path.insert(0, os.path.join(_HERE, '..', 'extract'))
        from export_domains import all_definitions
        D.defs = all_definitions(True)                # commented domains usable by name
        if a.domains == ['dispersion']:
            a.domains = DISPERSION_SET
    keys = a.domains or D.keys()
    for k in keys:
        D[k]                                          # unknown key -> clear error
    fp = None
    if a.footprint:
        from footprint import Footprint
        fp = Footprint(a.footprint_npz, a.footprint_outline)
    cov = None
    if a.coverage:
        import tomo2d as t2
        _fd = np.load(a.footprint_npz or t2._default_footprint(), allow_pickle=True)
        _f, _la, _lo = t2._orient(None, _fd['lat2d'], _fd['lon2d'])
        cov = (_la, _lo, t2.density_mask(_f(np.asarray(_fd['density'], float)), a.density_min))

    models, masks, prof, mohos = [], {}, {}, {}
    for spec in a.fvs:
        path, label = parse_model(spec)
        d = dict(np.load(path, allow_pickle=True))
        z = np.asarray(d['z'], float)
        zm = z <= a.d_max if a.d_max else np.ones(len(z), bool)
        F = {k: np.asarray(v, float)[:, zm] for k, v in model_fields(d).items()}
        models.append((path, label, d, z[zm], F))
        print(f"{label}: {os.path.basename(path)} -> {', '.join(F)}")
        mh = moho_at(a.moho, d['Lon'], d['Lat']) if a.moho else None
        for k in keys:
            m = D.node_mask(d, k, a.min_ndat, fp)
            if cov is not None:
                import tomo2d as t2
                ii, jj = t2.nearest_cell(cov[0], cov[1], d['Lat'], d['Lon'])
                m &= cov[2][ii, jj]
            masks[(label, k)] = m
            print(f"   {D[k]['label']:<24s} {m.sum():5d} nodes")
            for f, A in F.items():
                if m.sum():
                    prof[(label, k, f)] = stats(A[m], a.spread)
            if mh is not None and m.sum():
                mohos[(label, k)] = (np.nanmean(mh[m]), np.nanstd(mh[m]))

    fields = [f for f in (a.fields or FIELD_ORDER) if any(f in mod[4] for mod in models)]
    if not fields:
        sys.exit('no fields to plot')

    # ---- figure: [map] + one panel per field ----
    nmap = 0 if a.no_map else 1
    npan = len(fields) + (1 if a.rel_to else 0)
    W = 3.3 * npan + (4.6 if nmap else 0) + 0.6
    fig = plt.figure(figsize=(W, 6.8))
    x0 = 0.3
    tr = None
    if nmap:
        try:
            import cartopy.crs as ccrs
            tr = ccrs.PlateCarree()
            axm = fig.add_axes([x0 / W, 0.10, 4.2 / W, 0.78], projection=tr)
        except Exception:
            axm = fig.add_axes([x0 / W, 0.10, 4.2 / W, 0.78])
        sites = None
        if a.sites:
            import tomo2d as t2
            sites = t2.load_sites(a.sites)
        stations = None
        if a.stations:
            import tomo2d as t2
            _, slon, slat, _ = t2.load_stations(a.stations)
            stations = (slon, slat)
            print(f"  map: {len(slon)} stations from {a.stations}")
        draw_map(axm, D, keys, models, masks, tr, cov=cov if not a.no_veil else None, sites=sites,
                 veil_alpha=a.veil_alpha, veil_colour=a.veil_colour, stations=stations,
                 show_nodes=a.show_nodes or not a.stations)
        x0 += 4.6
    axes = []
    for j, f in enumerate(fields):
        ax = fig.add_axes([(x0 + 0.45 + 3.3 * j) / W, 0.10, 2.7 / W, 0.78])
        axes.append(ax)
        for mi, (path, label, d, z, F) in enumerate(models):
            if f not in F:
                continue
            for k in keys:
                p = prof.get((label, k, f))
                if p is None:
                    continue
                col = D[k]['colour']
                if mi == 0 or sum(f in m[4] for m in models) == 1:
                    ax.fill_betweenx(z, p['lo'], p['hi'], color=col, alpha=0.13, lw=0)
                ax.plot(p['mean'], z, STYLES[mi % len(STYLES)], color=col, lw=1.6)
        for k in keys:                                # Moho: domain mean on the right edge
            mo = mohos.get((models[0][1], k))
            if mo is not None:
                ax.plot([0.90, 1.0], [mo[0], mo[0]], color=D[k]['colour'], lw=3,
                        transform=ax.get_yaxis_transform(), solid_capstyle='butt')
        ax.set_ylim(models[0][3][-1], 0)
        ax.set_xlabel(FIELD_LABEL[f], fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
        if f == 'xi':
            ax.axvline(1.0, color='0.4', lw=0.8, ls='--')
        if j == 0:
            ax.set_ylabel('Depth (km)', fontsize=10)
    if a.rel_to:                                      # d(field) vs the reference domain mean
        f = fields[0]
        ax = fig.add_axes([(x0 + 0.45 + 3.3 * len(fields)) / W, 0.10, 2.7 / W, 0.78])
        for mi, (path, label, d, z, F) in enumerate(models):
            ref = prof.get((label, a.rel_to, f))
            if ref is None:
                print(f"  (--rel-to {a.rel_to}: no nodes for {label})")
                continue
            for k in keys:
                p = prof.get((label, k, f))
                if p is not None:
                    ax.plot(100 * (p['mean'] - ref['mean']) / ref['mean'], z, STYLES[mi % len(STYLES)],
                            color=D[k]['colour'], lw=1.6)
        ax.axvline(0, color='0.4', lw=0.8)
        ax.set_ylim(models[0][3][-1], 0)
        ax.set_xlabel(f"d{FIELD_LABEL[f].split(' (')[0]} vs {D[a.rel_to]['label']} (%)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5); ax.tick_params(labelsize=8)
    # legend: domains (colour) + models (line style) + band meaning
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color=D[k]['colour'], lw=2,
                label=f"{D[k]['label']} ({int(masks[(models[0][1], k)].sum())})") for k in keys
         if masks[(models[0][1], k)].sum()]
    if len(models) > 1:
        h += [Line2D([], [], color='k', ls=STYLES[i % len(STYLES)], lw=1.4, label=m[1])
              for i, m in enumerate(models)]
    h.append(Patch(color='0.6', alpha=0.3, label='+/-1 sigma lateral' if a.spread == 'sigma'
                   else '16-84 % lateral'))
    if mohos:
        h.append(Line2D([], [], color='0.3', lw=3, label='AR23 Moho (domain mean)'))
    fig.legend(handles=h, loc='upper center', ncol=min(len(h), 6), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, -0.01, 'Domain means over model nodes (count in brackets) with Ndat >= %d%s%s; '
             'GSWA 1:10M tectonic units (CC BY 4.0).' % (a.min_ndat, ', inside the array footprint' if fp else '',
                                                         ', inside the v3b coverage mask (density >= %g)'
                                                         % a.density_min if cov is not None else ''),
             ha='center', va='top', fontsize=7.5, color='0.3')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches='tight')
    fig.savefig(os.path.splitext(a.out)[0] + '.pdf', bbox_inches='tight')
    print(f"Saved: {a.out} (+ .pdf)")

    # ---- tables ----
    stem = os.path.splitext(a.out)[0]
    for path, label, d, z, F in models:
        arrays = dict(z=z, domains=np.array(keys))
        with open(f'{stem}.{label}.txt', 'w') as fh:
            fh.write(f'# domain 1-D profiles of {os.path.basename(path)} (Ndat >= {a.min_ndat}'
                     f"{', footprint' if fp else ''})\n")
            for k in keys:
                n = int(masks[(label, k)].sum())
                for f in F:
                    p = prof.get((label, k, f))
                    if p is None:
                        continue
                    fh.write(f'# {k} {f} n={n}\n# depth mean std p16 p50 p84\n')
                    for i, zz in enumerate(z):
                        fh.write(f"{zz:7.2f} {p['mean'][i]:.4f} {p['std'][i]:.4f} {p['p16'][i]:.4f} "
                                 f"{p['p50'][i]:.4f} {p['p84'][i]:.4f}\n")
                    for s in ('mean', 'std', 'p16', 'p50', 'p84'):
                        arrays[f'{k}.{f}.{s}'] = p[s]
                    arrays[f'{k}.n'] = n
        np.savez(f'{stem}.{label}.npz', **arrays)
        print(f"Tables: {stem}.{label}.txt / .npz")


if __name__ == '__main__':
    main()
