#!/usr/bin/env python3
"""
plot_site_depth_maps.py - Vsv depth slices (default 5, 20, 35, 45 km) of one or more TransD
models with the study sites marked (triangles in the terrane colours of the site map,
numbered), on the GSWA-style Albers frame with the 10M tectonic unit outlines.

  source env_stage6.sh
  python3 plotting/plot_site_depth_maps.py --fvs $FVS2:iter2 $FVS10:iter10 \\
      --sites examples/yilgarn_sites_19.txt --out $FIG/sites/site_depth_maps.iter2_iter10.png

One row per model, one column per depth; each depth column shares one colour scale
(Spectral, warm = slow; 2-98th percentile of the nodes of all models at that depth, or
--vlim per depth). Nodes with Ndat >= --min-ndat are interpolated (thin-plate RBF, 0.05 deg
grid) and cells further than --max-dist deg (0.5) from a node are blank, as in the other
maps. The area outside the v3b coverage mask (density >= 10) is faded with a white veil
and its edge dashed (--coverage blank: no colour there; none: ignore).
--rel shows dVsv (%) relative to each model's own mean at that depth (RdBu, white centre).
"""
import argparse, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', 'basemap'))
import tomo2d as t2                                           # noqa: E402
from domains import model_fields, Domains                     # noqa: E402

LABEL = {'vsv': 'Vsv', 'viso': 'Viso', 'xi': 'xi', 'vpvs': 'Vp/Vs'}


def model_label(spec):
    path, _, label = spec.partition(':')
    if not label:
        m = re.search(r'iter\.?(\d+)', os.path.basename(path))
        label = f'iter{m.group(1)}' if m else os.path.basename(path).replace('.npz', '')
    return path, label


def at_depth(A, z, zt):
    return np.array([np.interp(zt, z, row) for row in A])


def grid_fields(lon, lat, vals, glon, glat, max_dist):
    """Thin-plate RBF of several fields at once (vals [nodes, nfield]) on the glat x glon grid:
    one global solve shared by all depths (2-4x faster than a local fit per depth).
    -> (grids [nfield, nlat, nlon], ok [nlat, nlon] = within max_dist deg of a node)"""
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import cKDTree
    c = np.cos(np.radians(np.mean(lat)))
    ok = np.all(np.isfinite(vals), axis=1)
    X = np.column_stack([lon[ok] * c, lat[ok]])
    GLO, GLA = np.meshgrid(glon, glat)
    Q = np.column_stack([GLO.ravel() * c, GLA.ravel()])
    nb = None if ok.sum() <= 4000 else 60
    G = RBFInterpolator(X, vals[ok], kernel='thin_plate_spline', neighbors=nb)(Q)
    dist, _ = cKDTree(np.column_stack([lon[ok], lat[ok]])).query(np.column_stack([GLO.ravel(), GLA.ravel()]))
    return G.T.reshape(vals.shape[1], *GLO.shape), (dist <= max_dist).reshape(GLO.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs', nargs='+', required=True, metavar='PATH[:LABEL]')
    ap.add_argument('--depths', nargs='+', type=float, default=[5, 20, 35, 45])
    ap.add_argument('--field', default='vsv', choices=list(LABEL))
    ap.add_argument('--rel', action='store_true', help="d(field) %% vs each model's mean at that depth")
    ap.add_argument('--sites', default=os.path.join(_HERE, '..', 'examples', 'yilgarn_sites_19.txt'))
    ap.add_argument('--min-ndat', type=int, default=35)
    ap.add_argument('--max-dist', type=float, default=0.5, help='blank cells further than this (deg) from a node')
    ap.add_argument('--coverage', default='veil', choices=['veil', 'blank', 'none'],
                    help='v3b coverage mask (density >= 10): veil = fade the area outside it (default), '
                         'blank = no colour outside it, none = ignore')
    ap.add_argument('--veil-alpha', type=float, default=0.6)
    ap.add_argument('--veil-colour', default='white')
    ap.add_argument('--footprint', default=None)
    ap.add_argument('--vlim', nargs='+', type=float, default=None,
                    help='colour limits per depth: lo1 hi1 lo2 hi2 ... (abs) or L1 L2 ... (--rel, +-L)')
    ap.add_argument('--extent', default='nodes', help="nodes (model nodes +0.5 deg) | wa | 'lon0,lon1,lat0,lat1'")
    ap.add_argument('--grat', type=float, default=None)
    ap.add_argument('--dgrid', type=float, default=0.05, help='map grid step (deg); 0.1 = 4x fewer cells')
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--no-units', action='store_true', help='no 10M tectonic unit outlines')
    ap.add_argument('--no-labels', action='store_true', help='no site numbers')
    ap.add_argument('--out', default='figures/sites/site_depth_maps.png')
    a = ap.parse_args()

    sites = t2.load_sites(a.sites)
    slon = np.array([s['lon'] for s in sites]); slat = np.array([s['lat'] for s in sites])
    models = []
    for spec in a.fvs:
        path, label = model_label(spec)
        d = dict(np.load(path, allow_pickle=True))
        F = model_fields(d)
        if a.field not in F:
            sys.exit(f"{os.path.basename(path)} has no varying '{a.field}' (has: {', '.join(F)})")
        keep = np.asarray(d['Ndat'], float) >= a.min_ndat if 'Ndat' in d else np.ones(len(d['Lon']), bool)
        models.append(dict(label=label, lon=np.asarray(d['Lon'], float)[keep], lat=np.asarray(d['Lat'], float)[keep],
                           z=np.asarray(d['z'], float), A=np.asarray(F[a.field], float)[keep]))
        print(f"{label}: {os.path.basename(path)}, {keep.sum()} nodes with Ndat >= {a.min_ndat}")

    if a.extent == 'nodes':
        lo = np.concatenate([m['lon'] for m in models]); la = np.concatenate([m['lat'] for m in models])
        ext = [min(lo.min(), slon.min()) - 0.5, max(lo.max(), slon.max()) + 0.5,
               min(la.min(), slat.min()) - 0.5, max(la.max(), slat.max()) + 0.5]
    elif a.extent == 'wa':
        ext = [112.0, 129.6, -35.8, -13.2]
    else:
        ext = [float(x) for x in a.extent.split(',')]
    glon = np.arange(ext[0], ext[1] + 1e-6, a.dgrid); glat = np.arange(ext[2], ext[3] + 1e-6, a.dgrid)
    cov = None
    if a.coverage != 'none':
        try:
            cov, _ = t2.load_coverage(glat, glon, a.footprint)
        except Exception as e:
            print(f"  (no coverage mask: {e})")

    # values at depth + colour limits per depth
    V = {}
    for m in models:
        for zt in a.depths:
            v = at_depth(m['A'], m['z'], zt)
            if a.rel:
                v = 100.0 * (v - np.nanmean(v)) / np.nanmean(v)
            V[(m['label'], zt)] = v
    lims = []
    for j, zt in enumerate(a.depths):
        allv = np.concatenate([V[(m['label'], zt)] for m in models])
        if a.rel:
            L = a.vlim[j] if a.vlim and len(a.vlim) > j else max(0.5, 0.5 * np.ceil(np.nanpercentile(np.abs(allv), 98) / 0.5))
            lims.append((-L, L))
        elif a.vlim and len(a.vlim) >= 2 * (j + 1):
            lims.append((a.vlim[2 * j], a.vlim[2 * j + 1]))
        elif a.field == 'xi':                          # diverging around 1.0
            d = max(0.01, np.ceil(np.nanpercentile(np.abs(allv - 1.0), 98) * 100) / 100)
            lims.append((1.0 - d, 1.0 + d))
        else:
            lo_, hi_ = np.nanpercentile(allv, [2, 98])
            q = 100 if a.field == 'vpvs' else 20
            lims.append((np.floor(lo_ * q) / q, np.ceil(hi_ * q) / q))

    rings = []
    if not a.no_units:
        try:
            rings = [r for r in Domains(a.tect_npz).rings if len(r) >= 3]
        except Exception as e:
            print(f"  (no tectonic unit outlines: {e})")

    P = t2.Albers()
    bx, by = P(np.r_[ext[0], ext[1], ext[1], ext[0], 0.5 * (ext[0] + ext[1])],
               np.r_[ext[2], ext[2], ext[3], ext[3], ext[2]])
    asp = (by.max() - by.min()) / (bx.max() - bx.min())
    nr, nc = len(models), len(a.depths)
    w = 3.4
    W = nc * (w + 0.25) + 0.6
    nterr = len({x['terrane'] for x in sites})
    lcol = min(nterr, max(3, int(W / 1.2)))
    extra = 0.22 * (int(np.ceil(nterr / lcol)) - 1)       # legend rows below the colour bars
    H = nr * (w * asp + 0.6) + 1.6 + extra
    fig = plt.figure(figsize=(W, H))
    # house rules: absolute velocity Spectral (warm = slow); xi and relative RdBu with a white
    # centre; Vp/Vs viridis
    cmap = {'vpvs': matplotlib.colormaps['viridis']}.get(a.field, matplotlib.colormaps['Spectral'])
    if a.rel or a.field == 'xi':
        try:
            import plot_depth_slice as pds
            cmap = pds._cmap('RdBu', 16)
        except Exception:
            pass
    import time
    for r, m in enumerate(models):
        t0 = time.time()
        Gall, ok_all = grid_fields(m['lon'], m['lat'],
                                   np.column_stack([V[(m['label'], zt)] for zt in a.depths]),
                                   glon, glat, a.max_dist)
        print(f"  {m['label']}: interpolated {len(a.depths)} depths on {len(glat)}x{len(glon)} cells "
              f"in {time.time() - t0:.1f} s")
        for j, zt in enumerate(a.depths):
            x0 = 0.45 + j * (w + 0.25)
            y0 = H - 0.55 - (r + 1) * (w * asp + 0.6) + 0.3
            ax = fig.add_axes([x0 / W, y0 / H, w / W, w * asp / H])
            F = t2.MapFrame(ax, ext, grat=a.grat)
            G, ok = Gall[j], ok_all.copy()
            if cov is not None and a.coverage == 'blank':
                ok &= cov
            norm = Normalize(*lims[j])
            pm = F.raster(glat, glon, G, ok, cmap, norm)
            if cov is not None and a.coverage == 'veil':
                F.veil(glat, glon, cov, colour=a.veil_colour, alpha=a.veil_alpha, zorder=3)
                F.outline(glat, glon, cov, zorder=4, colors='0.25', linewidths=0.7, linestyles='--')
            if rings:
                F.lines(rings, colors='0.2', linewidths=0.3, zorder=5)
            x, y = F.xy(slon, slat)
            for k, s in enumerate(sites):
                ax.scatter(x[k], y[k], marker='^', s=46, c=[t2.terrane_colour(s['terrane'], k)],
                           edgecolors='k', linewidths=0.6, zorder=8)
                if not a.no_labels:
                    ax.annotate(str(s['no']), (x[k], y[k]), xytext=(3.5, 2.5), textcoords='offset points',
                                fontsize=6.3, fontweight='bold', zorder=9,
                                path_effects=[pe.withStroke(linewidth=1.6, foreground='w')])
            ax.set_title(f"{m['label']}, {zt:g} km", fontsize=9.5, fontweight='bold', loc='left', pad=3)
            if r == nr - 1:
                cax = fig.add_axes([(x0 + 0.12 * w) / W, (y0 - 0.62) / H, 0.76 * w / W, 0.1 / H])
                cb = plt.colorbar(pm, cax=cax, orientation='horizontal', extend='both')
                cb.ax.tick_params(labelsize=7)
                unit = ' (km/s)' if a.field in ('vsv', 'viso') else ''
                cb.set_label(f"d{LABEL[a.field]} (%) vs model mean at {zt:g} km" if a.rel
                             else f"{LABEL[a.field]}{unit} at {zt:g} km",
                             fontsize=7.5)
    terr = []
    for k, s in enumerate(sites):
        if s['terrane'] not in terr:
            terr.append(s['terrane'])
    h = [Line2D([], [], ls='', marker='^', ms=7, mfc=t2.terrane_colour(t, i), mec='k', label=t)
         for i, t in enumerate(terr)]
    fig.legend(handles=h, loc='lower center', ncol=lcol, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{LABEL[a.field]}" + (" (% vs each model's mean at depth)" if a.rel else '')
                 + f" at {', '.join(f'{d:g}' for d in a.depths)} km - {len(sites)} study sites (numbers as in "
                 f"{os.path.basename(a.sites)})", fontsize=10.5, y=1 - 0.15 / H)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=220); fig.savefig(os.path.splitext(a.out)[0] + '.pdf', dpi=220)
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")


if __name__ == '__main__':
    main()
