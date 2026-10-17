#!/usr/bin/env python3
"""
plot_site_period_maps.py - dispersion maps at a few periods, laid out like
plot_site_depth_maps.py: one row per data set (observed Data_SWD.in, or a model's best-fit
synthetics data_bestd.out), one column per period, the 19 study sites marked.

  A=/workspace/WA.Array/Year.3/noise.processing/ASDF.processing
  python3 plotting/plot_site_period_maps.py --comp cR --periods 10 20 30 50 \\
      --data "observed (v3 IN)"  "$A/v3/IN/Data_SWD.in.*" \\
      --syn  "v3 iter2.sm"       "$A/v3/OUT.iter.2.sm.Z/data_bestd.out*" \\
      --syn  "v3c iter3"         "$A/v3c/OUT.iter.3.Z/data_bestd.out*" \\
      --resid --out $FIG/v3c_iter3/vs_v3/period_maps.cR.png

--data LABEL GLOB  observed files (period first: p cR uR cL uL, -1 = missing); repeatable
--syn  LABEL GLOB  best-fit synthetics (period last: cR uR cL uL p); repeatable
                   (--data-order / --syn-order to change the column orders)
Quote the globs (** recurses). Node positions come from the 'x.<lon>.y.<lat>' in the paths.
--comp  cR (Rayleigh phase, default) | uR (Rayleigh group) | cL | uL
--resid adds one row per --syn: synthetic - observed (km/s, RdBu), against the --data set
        given in the same position (2nd --syn vs 2nd --data), else against the first --data.
--rel   shows each map as % from the mean of the observed at that period.
Values at a period: linear interpolation in log(period) within each node's own period range
(no extrapolation; missing values skipped). Each period column shares one colour scale
(Spectral, warm = slow, 2-98th percentile of all rows, or --vlim LO HI per period).
Maps: thin-plate RBF on a --dgrid grid, blank beyond --max-dist deg of a node, white veil
outside the v3b coverage mask, 10M tectonic unit outlines - as in plot_site_depth_maps.py.
"""
import argparse, os, sys, time
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
from plot_site_fits import index, read_table                  # noqa: E402
from plot_site_depth_maps import grid_fields                  # noqa: E402

NAME = {'cR': 'Rayleigh phase', 'uR': 'Rayleigh group', 'cL': 'Love phase', 'uL': 'Love group'}


def load_set(label, pattern, order, comp, periods, kind):
    files = index(pattern)
    if not files:
        sys.exit(f"{label}: no file with 'x.<lon>.y.<lat>' in its path matches {pattern}")
    keys = sorted(files)
    V = np.full((len(keys), len(periods)), np.nan)
    lp = np.log(np.asarray(periods, float))
    nbad = 0
    for i, k in enumerate(keys):
        T = read_table(files[k], order)
        if T is None:
            nbad += 1
            continue
        ok = np.isfinite(T[comp]) & (T['p'] > 0)
        if ok.sum() < 2:
            continue
        x, y = np.log(T['p'][ok]), T[comp][ok]
        V[i] = np.interp(lp, x, y, left=np.nan, right=np.nan)
    lon = np.array([k[0] for k in keys]); lat = np.array([k[1] for k in keys])
    n = np.isfinite(V).sum(0)
    print(f"{label} ({kind}): {len(keys)} nodes from {pattern}"
          + (f", {nbad} unreadable" if nbad else '') + "; nodes with a value: "
          + ', '.join(f"{p:g}s {c}" for p, c in zip(periods, n)))
    return dict(label=label, kind=kind, lon=lon, lat=lat, V=V, key={k: i for i, k in enumerate(keys)})


def rms(v):
    v = v[np.isfinite(v)]
    return np.sqrt(np.mean(v ** 2)) if len(v) else np.nan


def on_nodes(src, dst):
    """src values at dst's nodes (same node keys); NaN where src has no such node."""
    out = np.full_like(dst['V'], np.nan)
    for k, i in dst['key'].items():
        j = src['key'].get(k)
        if j is not None:
            out[i] = src['V'][j]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs=2, action='append', default=[], metavar=('LABEL', 'GLOB'))
    ap.add_argument('--syn', nargs=2, action='append', default=[], metavar=('LABEL', 'GLOB'))
    ap.add_argument('--comp', default='cR', choices=list(NAME))
    ap.add_argument('--periods', nargs='+', type=float, default=[10, 20, 30, 50])
    ap.add_argument('--resid', action='store_true', help='add synthetic - observed rows')
    ap.add_argument('--rel', action='store_true', help='%% from the mean of the observed at each period')
    ap.add_argument('--data-order', default='p cR uR cL uL')
    ap.add_argument('--syn-order', default='cR uR cL uL p')
    ap.add_argument('--sites', default=os.path.join(_HERE, '..', 'examples', 'yilgarn_sites_19.txt'))
    ap.add_argument('--max-dist', type=float, default=0.5)
    ap.add_argument('--coverage', default='veil', choices=['veil', 'blank', 'none'])
    ap.add_argument('--veil-alpha', type=float, default=0.6)
    ap.add_argument('--footprint', default=None)
    ap.add_argument('--vlim', nargs='+', type=float, default=None, help='LO HI per period (absolute maps)')
    ap.add_argument('--rlim', type=float, default=None, help='+- limit of the residual rows (km/s)')
    ap.add_argument('--extent', default='nodes', help="nodes | wa | 'lon0,lon1,lat0,lat1'")
    ap.add_argument('--dgrid', type=float, default=0.05)
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--no-units', action='store_true')
    ap.add_argument('--no-labels', action='store_true')
    ap.add_argument('--out', default='figures/sites/site_period_maps.png')
    a = ap.parse_args()
    if not a.data and not a.syn:
        sys.exit('give at least one --data or --syn set')
    per = a.periods
    D = [load_set(l, g, a.data_order.split(), a.comp, per, 'observed') for l, g in a.data]
    S = [load_set(l, g, a.syn_order.split(), a.comp, per, 'synthetic') for l, g in a.syn]
    rows = D + S
    R = []
    if a.resid and D:
        for i, s in enumerate(S):
            d = D[i] if i < len(D) else D[0]
            R.append(dict(label=f"{s['label']} - {d['label']}", kind='resid', lon=s['lon'], lat=s['lat'],
                          V=s['V'] - on_nodes(d, s)))
            r = R[-1]['V']
            print(f"  residual {R[-1]['label']}: RMS per period "
                  + ', '.join(f"{p:g}s {rms(r[:, j]):.3f}" for j, p in enumerate(per)))
    if a.rel:
        ref = D[0] if D else S[0]
        mu = np.nanmean(ref['V'], axis=0)
        for r in rows:
            r['V'] = 100.0 * (r['V'] - mu) / mu
    rows = rows + R

    sites = t2.load_sites(a.sites)
    slon = np.array([s['lon'] for s in sites]); slat = np.array([s['lat'] for s in sites])
    if a.extent == 'nodes':
        lo = np.concatenate([r['lon'] for r in rows]); la = np.concatenate([r['lat'] for r in rows])
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

    # colour limits: one per period for the data/synthetic rows, one for all residual rows
    lims = []
    for j, p in enumerate(per):
        allv = np.concatenate([r['V'][:, j] for r in rows if r['kind'] != 'resid'])
        allv = allv[np.isfinite(allv)]
        if a.rel:
            L = max(0.5, 0.5 * np.ceil(np.percentile(np.abs(allv), 98) / 0.5)) if len(allv) else 1
            lims.append((-L, L))
        elif a.vlim and len(a.vlim) >= 2 * (j + 1):
            lims.append((a.vlim[2 * j], a.vlim[2 * j + 1]))
        elif len(allv):
            lo_, hi_ = np.percentile(allv, [2, 98])
            lims.append((np.floor(lo_ * 50) / 50, np.ceil(hi_ * 50) / 50))
        else:
            lims.append((3, 4.5))
    if R:
        rv = np.concatenate([r['V'][np.isfinite(r['V'])] for r in R])
        RL = a.rlim or (max(0.02, 0.01 * np.ceil(np.percentile(np.abs(rv), 98) / 0.01)) if len(rv) else 0.1)

    rings = []
    if not a.no_units:
        try:
            from domains import Domains
            rings = [r for r in Domains(a.tect_npz).rings if len(r) >= 3]
        except Exception as e:
            print(f"  (no tectonic unit outlines: {e})")

    try:
        import plot_depth_slice as pds
        div = pds._cmap('RdBu', 16)
    except Exception:
        div = matplotlib.colormaps['RdBu']
    spec = matplotlib.colormaps['Spectral']
    P = t2.Albers()
    bx, by = P(np.r_[ext[0], ext[1], ext[1], ext[0], 0.5 * (ext[0] + ext[1])],
               np.r_[ext[2], ext[2], ext[3], ext[3], ext[2]])
    asp = (by.max() - by.min()) / (bx.max() - bx.min())
    nr, nc = len(rows), len(per)
    w = 3.4
    W = nc * (w + 0.25) + 0.6 + (1.0 if R else 0)
    nterr = len({x['terrane'] for x in sites})
    lcol = min(nterr, max(3, int(W / 1.2)))
    nd = sum(r['kind'] != 'resid' for r in rows)
    gap = 0.55 if R else 0.0                           # room for the colour bars above the residual rows
    H = nr * (w * asp + 0.6) + 1.6 + gap + 0.22 * (int(np.ceil(nterr / lcol)) - 1)
    fig = plt.figure(figsize=(W, H))
    for ri, r in enumerate(rows):
        t0 = time.time()
        res = r['kind'] == 'resid'
        V = r['V']
        G, okg = np.full((nc, len(glat), len(glon)), np.nan), np.zeros((nc, len(glat), len(glon)), bool)
        for j in range(nc):                            # node sets differ per period (missing values)
            f = np.isfinite(V[:, j])
            if f.sum() >= 4:
                Gj, okj = grid_fields(r['lon'][f], r['lat'][f], V[f, j:j + 1], glon, glat, a.max_dist)
                G[j], okg[j] = Gj[0], okj
        print(f"  {r['label']}: gridded {nc} periods in {time.time() - t0:.1f} s")
        for j, p in enumerate(per):
            x0 = 0.45 + j * (w + 0.25)
            y0 = H - 0.55 - (ri + 1) * (w * asp + 0.6) + 0.3 - (gap if ri >= nd else 0)
            ax = fig.add_axes([x0 / W, y0 / H, w / W, w * asp / H])
            F = t2.MapFrame(ax, ext)
            ok = okg[j].copy()
            if cov is not None and a.coverage == 'blank':
                ok &= cov
            norm = Normalize(-RL, RL) if res else Normalize(*lims[j])
            pm = F.raster(glat, glon, G[j], ok, div if (res or a.rel) else spec, norm)
            if cov is not None and a.coverage == 'veil':
                F.veil(glat, glon, cov, colour='white', alpha=a.veil_alpha, zorder=3)
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
            tag = 'syn - obs' if res else r['kind']
            ax.set_title(f"{r['label']}, {p:g} s", fontsize=9, fontweight='bold', loc='left', pad=3)
            ax.text(0.98, 0.02, tag, transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color='0.35')
            if res:
                ax.text(0.02, 0.02, f"RMS {rms(V[:, j]):.3f}", transform=ax.transAxes,
                        fontsize=7, color='0.2', bbox=dict(fc='w', ec='none', alpha=0.8, pad=1))
            if ri == nd - 1:                               # colour bar under the last absolute row
                cax = fig.add_axes([(x0 + 0.12 * w) / W, (y0 - 0.42) / H, 0.76 * w / W, 0.09 / H])
                cb = plt.colorbar(pm, cax=cax, orientation='horizontal', extend='both')
                cb.ax.tick_params(labelsize=7)
                cb.set_label(f"{NAME[a.comp]} " + (f"(% vs observed mean) at {p:g} s" if a.rel
                                                   else f"(km/s) at {p:g} s"), fontsize=7.5)
            if res and j == nc - 1 and ri == nr - 1:
                cax = fig.add_axes([(x0 + w + 0.3) / W, (y0 + 0.1 * w * asp) / H, 0.1 / W, 0.8 * w * asp / H])
                cb = plt.colorbar(pm, cax=cax, extend='both')
                cb.ax.tick_params(labelsize=7)
                cb.set_label('synthetic - observed (km/s)', fontsize=7.5)
    terr = []
    for s in sites:
        if s['terrane'] not in terr:
            terr.append(s['terrane'])
    h = [Line2D([], [], ls='', marker='^', ms=7, mfc=t2.terrane_colour(t, i), mec='k', label=t)
         for i, t in enumerate(terr)]
    fig.legend(handles=h, loc='lower center', ncol=lcol, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{NAME[a.comp]} at {', '.join(f'{p:g}' for p in per)} s - observed and TransD best-fit "
                 f"synthetics, {len(sites)} study sites", fontsize=10.5, y=1 - 0.15 / H)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    short = lambda g: os.path.join(*os.path.normpath(os.path.dirname(g.replace('**/', ''))).split(os.sep)[-2:],
                                   os.path.basename(g))
    fig.text(0.995, 0.003, 'files: ' + '; '.join(short(g) for _, g in a.data + a.syn), ha='right', va='bottom',
             fontsize=5.5, color='0.45')
    fig.savefig(a.out, dpi=220); fig.savefig(os.path.splitext(a.out)[0] + '.pdf', dpi=220)
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")


if __name__ == '__main__':
    main()
