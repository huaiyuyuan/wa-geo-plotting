#!/usr/bin/env python3
"""
plot_map_compare.py - two models side by side in map view, plus their difference.

  rows    : depths (default 5 15 30 45 km)
  columns : model A | model B | A - B
  abs mode: Vs (km/s), A and B on one shared scale per depth (Spectral, warm = slow);
            difference = (A/B - 1) * 100 %
  rel mode: dVs (%) vs each model's own mean at that depth (RdBu);
            difference = dlnA - dlnB (% points)
  The difference is taken at model A's nodes (B interpolated there), so the two
  models need not share a node set. Crustal boundaries, ocean mask and the array
  footprint focus as on the other maps.

  python3 plot_map_compare.py \\
      --models $OUT/Fvs.iter.10.Z.npz:"iter10 (no Moho)" $OUT/Fvs.iter.8.ZT.npz:"iter8 (Moho)" \\
      --mode abs --out-dir $OUT/figures/compare_iter10_iter8

Model spec PATH[:LABEL[:VEL]]: VEL auto (default) converts an xi (ZT) model to true
Vsv = Viso/sqrt((2+Xi^2)/3) for --field vsv; 'model' keeps Viso.
Output: <out-dir>/compare.<field>.<mode>.<fp|wa>.png (+ .pdf)
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
import plot_depth_slice as pds                       # noqa: E402
from plot_depth_slice import (_make_ax, _add_states_ocean, _mask_wa_points,  # noqa: E402
                              _plot_smooth, _plot_scatter, _cmap, _at_depth, _HAS_CARTOPY)
from wa_basemap import add_crustal_boundaries, boundaries_legend, add_tectonic_outlines  # noqa: E402
import footprint as fpmod                            # noqa: E402
try:
    import cartopy.crs as ccrs
except ImportError:
    ccrs = None


def load_model(spec, field, min_ndat):
    parts = spec.split(':')
    path = parts[0]
    label = parts[1] if len(parts) > 1 and parts[1] else None
    vel = parts[2] if len(parts) > 2 and parts[2] else 'auto'
    raw = np.load(path, allow_pickle=True)
    d = {k: raw[k] for k in raw.files}
    xi_model = 'Xi' in d and np.nanstd(d['Xi']) > 1e-4
    if label is None:
        m = re.search(r'iter\.?(\d+)', os.path.basename(path))
        label = f'iter{m.group(1)}' if m else os.path.basename(path).replace('.npz', '')
    note = ''
    if field == 'xi':
        if not xi_model:
            sys.exit(f"{os.path.basename(path)}: xi is fixed - not an xi model")
        V, name = d['Xi'], 'Xi'
    else:
        if vel == 'auto':
            vel = 'vsv_true' if xi_model else 'model'
        V = d['Vsv']
        if vel == 'vsv_true':
            if not xi_model:
                sys.exit(f"{os.path.basename(path)}: vsv_true needs an xi model")
            V = V / np.sqrt((2.0 + d['Xi'] ** 2) / 3.0)
            note = ' (true Vsv from Viso, xi)'
        name = 'Viso' if (xi_model and vel == 'model') else 'Vsv'
    lon, lat = np.asarray(d['Lon'], float), np.asarray(d['Lat'], float)
    keep = _mask_wa_points(lon, lat)
    if min_ndat and 'Ndat' in d:
        keep &= np.asarray(d['Ndat'], float) >= min_ndat
    print(f"  {label}: {os.path.basename(path)} -> {name}{note}; {keep.sum()} of {len(lon)} nodes")
    return dict(label=label, note=note, name=name, lon=lon, lat=lat, z=np.asarray(d['z'], float),
                V=np.asarray(V, float), keep=keep,
                ndat=np.asarray(d['Ndat'], float) if 'Ndat' in d else None)


def to_nodes(lon_s, lat_s, val, lon_t, lat_t):
    """Values at target nodes: exact coordinate match if the grids coincide, else linear."""
    key = lambda lo, la: np.round(lo, 4) * 1e5 + np.round(la, 4)
    ks, kt = key(lon_s, lat_s), key(lon_t, lat_t)
    if np.all(np.isin(kt, ks)):
        order = {k: i for i, k in enumerate(ks)}
        return val[[order[k] for k in kt]]
    from scipy.interpolate import LinearNDInterpolator
    fin = np.isfinite(val)
    return LinearNDInterpolator(np.column_stack([lon_s[fin], lat_s[fin]]), val[fin])(lon_t, lat_t)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs=2, required=True, metavar='PATH[:LABEL[:VEL]]')
    ap.add_argument('--field', default='vsv', choices=['vsv', 'xi'])
    ap.add_argument('--mode', default='abs', choices=['abs', 'rel'])
    ap.add_argument('--depths', nargs='+', type=float, default=[5, 15, 30, 45])
    ap.add_argument('--sigma', type=float, default=1.5, help='abs limits = median +/- sigma*std')
    ap.add_argument('--clim-rel', type=float, default=6.0)
    ap.add_argument('--clim-diff', type=float, default=4.0, help='difference limit (%% or %% points)')
    ap.add_argument('--ncolors', type=int, default=32)
    ap.add_argument('--min-ndat', type=int, default=35)
    ap.add_argument('--no-smooth', action='store_true', help='plot nodes as dots, no interpolation')
    ap.add_argument('--overlay', default='boundaries', choices=['boundaries', 'outlines', 'both', 'none'])
    ap.add_argument('--bnd-npz', default=None)
    ap.add_argument('--no-footprint', action='store_true', help='full WA view')
    ap.add_argument('--footprint-npz', default=None)
    ap.add_argument('--footprint-outline', default=None)
    ap.add_argument('--footprint-veil', type=float, default=0.75)
    ap.add_argument('--div-gap', type=float, default=0.07)
    ap.add_argument('--div-white', type=int, default=2)
    ap.add_argument('--out-dir', default='figures/compare')
    a = ap.parse_args()
    pds.DIV_GAP, pds.DIV_WHITE = a.div_gap, a.div_white

    print('Models:')
    A, B = (load_model(s, a.field, a.min_ndat) for s in a.models)
    fp = None
    if not a.no_footprint:
        fp = fpmod.Footprint(a.footprint_npz, a.footprint_outline)
        for M in (A, B):
            M['keep'] &= fp.contains(M['lon'], M['lat'])
        print('  ' + fp.describe(A['keep']))

    tr = ccrs.PlateCarree() if (_HAS_CARTOPY and ccrs) else None
    unit = '' if a.field == 'xi' else ' (km/s)'
    nm = A['name']
    nr, ncol = len(a.depths), 3
    pw = 4.4
    pitch = pw * 1.06                       # row pitch: room for tick labels + next title
    W, H = ncol * pw + 0.4, nr * pitch + 1.0
    fig = plt.figure(figsize=(W, H))
    what = (f"{nm}{unit}" if a.mode == 'abs' else f"dln{nm} (%) vs each model's mean at depth")
    fig.suptitle(f"{A['label']} vs {B['label']} - {what}", fontsize=13, fontweight='bold',
                 y=1 - 0.15 / H, va='top')
    heads = [A['label'] + A['note'], B['label'] + B['note'], f"{A['label']} minus {B['label']}"]
    for c, h in enumerate(heads):
        fig.text((0.2 + c * pw + pw / 2) / W, 1 - 0.62 / H, h, ha='center', va='top',
                 fontsize=10, fontweight='bold')

    for r, zt in enumerate(a.depths):
        vals = {}
        for key, M in (('A', A), ('B', B)):
            v = _at_depth(M['V'], M['z'], zt)
            if a.mode == 'rel':
                v = (v / np.nanmean(v[M['keep']]) - 1.0) * 100.0
            vals[key] = v
        k = A['keep']
        vB_at_A = to_nodes(B['lon'][B['keep']], B['lat'][B['keep']], vals['B'][B['keep']],
                           A['lon'][k], A['lat'][k])
        if a.mode == 'abs':
            dif = (vals['A'][k] / vB_at_A - 1.0) * 100.0
            dlab = f"{nm}: A/B - 1 (%)"
            if a.field == 'xi':
                lims = (0.90, 1.10)
            else:
                both = np.r_[vals['A'][A['keep']], vals['B'][B['keep']]]
                med, sd = np.nanmedian(both), np.nanstd(both)
                lims = (med - a.sigma * sd, med + a.sigma * sd)
            cm = _cmap('RdBu' if a.field == 'xi' else 'Spectral', a.ncolors)
            clab = nm + unit
        else:
            dif = vals['A'][k] - vB_at_A
            dlab = f"dln{nm}: A - B (% pts)"
            lims, cm, clab = (-a.clim_rel, a.clim_rel), _cmap('RdBu', a.ncolors), f"dln{nm} (%)"
        cells = [(A, vals['A'][A['keep']], A['keep'], cm, lims, clab),
                 (B, vals['B'][B['keep']], B['keep'], cm, lims, clab),
                 (A, dif, k, _cmap('RdBu', a.ncolors), (-a.clim_diff, a.clim_diff), dlab)]
        for c, (M, v, kk, cmap, (vmin, vmax), lab) in enumerate(cells):
            x0 = 0.2 + c * pw
            y0 = H - 0.9 - (r + 1) * pitch + 0.3
            ax = _make_ax(fig, [x0 / W, y0 / H, (pw - 0.15) / W, (pw * 0.86) / H])
            lon, lat = M['lon'][kk], M['lat'][kk]
            if a.no_smooth:
                im = _plot_scatter(ax, lon, lat, v, cmap, vmin, vmax, s=40)
            else:
                im, *_ = _plot_smooth(ax, lon, lat, v, cmap, vmin, vmax, method='rbf', npts=200,
                                      all_lon=lon, all_lat=lat, max_dist=0.5)
            _add_states_ocean(ax)
            if a.overlay in ('outlines', 'both'):
                add_tectonic_outlines(ax, lw=0.4, alpha=0.4, zorder=6, transform=tr, major_only=True)
            if a.overlay in ('boundaries', 'both'):
                add_crustal_boundaries(ax, a.bnd_npz, transform=tr)
                if r == 0 and c == 0:
                    boundaries_legend(ax, loc='lower left', fontsize=6)
            if fp is not None:
                fp.focus(ax, tr, veil_alpha=a.footprint_veil, zoom=True)
                fpmod.inset_colorbar(ax, im, lab, extend='both')
            else:
                plt.colorbar(im, ax=ax, shrink=0.75, label=lab, extend='both')
            ax.set_title(f'z = {zt:g} km', fontsize=9)

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"compare.{a.field}.{a.mode}.{'wa' if fp is None else 'fp'}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {out} (+ .pdf)")


if __name__ == '__main__':
    main()
