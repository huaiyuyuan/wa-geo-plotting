#!/usr/bin/env python3
"""
plot_depth_panels.py — Multi-depth 2x2 panels, ONE field per figure.
Produces 3 figures (PNG+PDF): absolute, relative, model-error — each a 2x2
grid of depths (default 10/20/30/40 km). Coastline + WA state + structural
overlays. First panel can carry extra structural lines.

Usage:
  python3 plot_depth_panels.py --fvs Fvs.iter.2.Z.npz --depths 10 20 30 40 \
      --smooth --out-dir figures/panels
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# repo layout: basemap/ (wa_basemap) and repo root (config) beside plotting/
for _p in (os.path.join(_HERE, '..', 'basemap'), os.path.join(_HERE, '..')):
    if os.path.abspath(_p) not in map(os.path.abspath, sys.path):
        sys.path.insert(1, os.path.abspath(_p))
from plot_depth_slice import (_make_ax, _add_states_ocean, _mask_wa_points,
                              _plot_smooth, _plot_scatter, _cmap, _at_depth,
                              _HAS_CARTOPY)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    pass

def _add_structural(ax):
    """Add structural geology overlays to a panel (rivers as proxy + placeholder).
    Extend here with WA terrane-boundary shapefiles when available."""
    if not _HAS_CARTOPY: return
    try:
        ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor='0.3', alpha=0.4, zorder=9)
    except Exception:
        pass
    # TODO: overlay GSWA terrane boundaries shapefile here:
    #   import cartopy.io.shapereader as shpreader
    #   reader = shpreader.Reader('/path/to/wa_terranes.shp')
    #   ax.add_geometries(reader.geometries(), ccrs.PlateCarree(),
    #                     facecolor='none', edgecolor='k', linewidth=0.6, zorder=9)

def _panel(fig, rect, d, zt, field, mode, args, wa, Lon_wa, Lat_wa, Lon_sm, Lat_sm,
           ndat_wa, sm, first=False):
    ax = _make_ax(fig, rect)
    z = d['z']
    vname = 'Viso' if _is_xi_model(d) else 'Vsv'
    if field == 'vsv':
        arr = d['Vsv']; label=f'{vname} (km/s)'; cmapname='Spectral'   # warm = slow
    elif field == 'xi':
        arr = d['Xi']; label='Xi'; cmapname='RdBu'
    elif field == 'vpvs':
        arr = d['Vpvs']; label='Vp/Vs'; cmapname='viridis'
    else:
        arr = d['Vsv']; label='Vsv'; cmapname='Spectral'
    val = _at_depth(arr, z, zt)
    val_wa = val[wa]; val_sm = val_wa[sm]

    if mode == 'abs':
        med, std = np.nanmedian(val_sm), np.nanstd(val_sm)
        vmin, vmax = med - args.sigma_abs*std, med + args.sigma_abs*std
        if field == 'xi':
            vmin, vmax = 0.90, 1.10        # same fixed xi scale as sections/slices
        cmap = _cmap(cmapname, args.ncolors)
        data_sm, data_wa = val_sm, val_wa
        cbl = label
    elif mode == 'rel':
        mean = np.nanmean(val_sm)
        rel_sm = (val_sm-mean)/mean*100
        rel_wa = (val_wa-mean)/mean*100
        vmin, vmax = -args.clim_rel, args.clim_rel
        cmap = _cmap('RdBu', args.ncolors)
        data_sm, data_wa = rel_sm, rel_wa
        cbl = f'dln{label.split()[0]} (%)'
    elif mode == 'error':  # DATA misfit (fit quality) — one value per node
        if 'Misfit' not in d:
            ax.set_visible(False); return
        mis = d['Misfit'][wa].astype(float)
        fin = np.isfinite(mis)
        if _FP_ACTIVE: fin &= sm             # footprint: only nodes inside
        vmax = np.nanpercentile(mis[fin], 90) if fin.any() else 1
        cmap = _cmap('YlOrRd', args.ncolors)
        im = _plot_scatter(ax, Lon_wa[fin], Lat_wa[fin], mis[fin], cmap, 0, vmax, s=args.markersize)
        _cbar(im, ax, 'RMS misfit (km/s)', 'max')
        ax.set_title(f'z={zt:.0f} km', fontsize=9)
        _add_states_ocean(ax)
        if first: _add_structural(ax)
        return
    else:  # uncert — MODEL uncertainty (posterior IQR) at this depth
        errkey = {'vsv':'Vsv_err','xi':'Xi_err','vpvs':'Vpvs_err'}.get(field,'Vsv_err')
        if errkey not in d:
            ax.set_visible(False); return
        err = _at_depth(d[errkey], z, zt)[wa].astype(float)
        fin = np.isfinite(err)
        if _FP_ACTIVE: fin &= sm             # footprint: only nodes inside
        vmax = np.nanpercentile(err[fin], 90) if fin.any() else 1
        cmap = _cmap('YlOrRd', args.ncolors)
        im = _plot_scatter(ax, Lon_wa[fin], Lat_wa[fin], err[fin], cmap, 0, vmax, s=args.markersize)
        _cbar(im, ax, f'{field} IQR/2', 'max')
        ax.set_title(f'z={zt:.0f} km', fontsize=9)
        _add_states_ocean(ax)
        if first: _add_structural(ax)
        return

    if args.smooth:
        im, *_ = _plot_smooth(ax, Lon_sm, Lat_sm, data_sm, cmap, vmin, vmax,
                              method=args.interp, npts=args.npts,
                              all_lon=Lon_wa, all_lat=Lat_wa,
                              ndat=ndat_wa, ndat_min=args.min_ndat, max_dist=args.max_dist)
    else:
        im = _plot_scatter(ax, Lon_wa, Lat_wa, data_wa, cmap, vmin, vmax, s=args.markersize)
    _cbar(im, ax, cbl, 'both')
    ax.set_title(f'z={zt:.0f} km', fontsize=9)
    _add_states_ocean(ax)
    if first: _add_structural(ax)

_MAP_AXES = []                     # map axes made by _make_ax (for the footprint overlay)
_FP_ACTIVE = False


def _is_xi_model(d):
    return 'Xi' in d.files and np.nanstd(d['Xi']) > 1e-4


def _check_field_varies(d, field):
    key = {'vsv': 'Vsv', 'xi': 'Xi', 'vpvs': 'Vpvs'}[field]
    if key not in d.files:
        sys.exit(f"no '{key}' array in this Fvs file")
    if np.nanstd(d[key]) <= 1e-4:
        sys.exit(f"{key} is fixed at {np.nanmean(d[key]):.4f} in this model - nothing to map"
                 + (" (use the ZT/xi Fvs for xi)" if field == 'xi' else ""))


def _cbar(im, ax, label, extend):
    """Inset colourbar at the NE corner when focusing on the footprint, else as before."""
    if _FP_ACTIVE:
        import footprint
        return footprint.inset_colorbar(ax, im, label, extend=extend)
    return plt.colorbar(im, ax=ax, shrink=0.75, label=label, extend=extend)
_make_ax_orig = _make_ax


def _make_ax(fig, rect):
    ax = _make_ax_orig(fig, rect)
    _MAP_AXES.append(ax)
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs', required=True)
    ap.add_argument('--depths', nargs='+', type=float, default=[10,20,30,45],
                    help='four depths (km); 45 not 40, which sits on the Moho artefact')
    ap.add_argument('--field', default='vsv', choices=['vsv','xi','vpvs'])
    ap.add_argument('--modes', nargs='+', default=['abs','rel','error'],
                    choices=['abs','rel','error','uncert'],
                    help='abs|rel|error(data misfit)|uncert(posterior IQR)')
    ap.add_argument('--smooth', action='store_true')
    ap.add_argument('--interp', default='rbf')
    ap.add_argument('--npts', type=int, default=200)
    ap.add_argument('--markersize', type=float, default=120)
    ap.add_argument('--ncolors', type=int, default=32)
    ap.add_argument('--sigma-abs', type=float, default=1.25)
    ap.add_argument('--clim-rel', type=float, default=6.0)
    ap.add_argument('--min-ndat', type=int, default=35)
    ap.add_argument('--max-dist', type=float, default=0.5)
    ap.add_argument('--out-dir', default='figures/panels')
    import footprint as _fpmod
    _fpmod.add_args(ap)
    ap.add_argument('--div-gap', type=float, default=0.15,
                    help='diverging colour maps skip +/- this band around white (0 = classic)')
    args = ap.parse_args()
    import plot_depth_slice as _pds
    _pds.DIV_GAP = args.div_gap

    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.fvs, allow_pickle=True)
    _check_field_varies(d, args.field)
    Lon, Lat = d['Lon'], d['Lat']
    wa = _mask_wa_points(Lon, Lat)
    Lon_wa, Lat_wa = Lon[wa], Lat[wa]
    ndat_wa = d['Ndat'][wa].astype(float) if 'Ndat' in d else None
    if args.min_ndat>0 and ndat_wa is not None:
        sm = ndat_wa >= args.min_ndat
    else:
        sm = np.ones(wa.sum(), dtype=bool)
    global _FP_ACTIVE
    fp = _fpmod.from_args(args)
    _FP_ACTIVE = fp is not None
    if fp is not None:                       # selection: nodes inside the footprint mask
        inside = fp.contains(Lon_wa, Lat_wa)
        sm &= inside
        print('  ' + fp.describe(inside))
    Lon_sm, Lat_sm = Lon_wa[sm], Lat_wa[sm]

    depths = args.depths[:4]  # 2x2
    rects = [[0.04,0.53,0.40,0.42],[0.53,0.53,0.40,0.42],
             [0.04,0.05,0.40,0.42],[0.53,0.05,0.40,0.42]]
    base = os.path.basename(args.fvs).replace('.npz','')
    titles = {'abs':'Absolute','rel':'Relative',
              'error':'Data misfit (RMS)','uncert':'Model uncertainty (posterior IQR)'}

    for mode in args.modes:
        fig = plt.figure(figsize=(13,13))
        fig.suptitle(f'{base} — {args.field.upper()} {titles[mode]}', fontsize=13, fontweight='bold')
        _MAP_AXES.clear()
        for k,zt in enumerate(depths):
            _panel(fig, rects[k], d, zt, args.field, mode, args, wa,
                   Lon_wa, Lat_wa, Lon_sm, Lat_sm, ndat_wa, sm, first=(k==0))
        if fp is not None:
            tr = ccrs.PlateCarree() if _HAS_CARTOPY else None
            for ax in _MAP_AXES:
                fp.focus(ax, tr, veil_alpha=args.footprint_veil,
                         zoom=not args.no_footprint_zoom)
        stem = os.path.join(args.out_dir, f'{args.field}.{mode}.panels')
        fig.savefig(stem+'.png', dpi=150, bbox_inches='tight')
        fig.savefig(stem+'.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {stem}.png / .pdf')

if __name__ == '__main__':
    main()
