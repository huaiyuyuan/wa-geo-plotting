#!/usr/bin/env python3
"""
plot_vsv_tectonic.py — Vsv/xi depth slices with ocean mask, coastline, AND
GSWA tectonic overlay. Reuses the masking/ocean helpers from plot_depth_slice.py
so the grey ocean + coastline match the other figures.

  python3 plot_vsv_tectonic.py --fvs Fvs.iter.2.Z.npz --depths 10 20 30 40 \
      --field vsv --overlay outlines --major --smooth --out fig.png
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# repo layout: basemap/ (wa_basemap) and repo root (config) beside plotting/
for _p in (os.path.join(_HERE, '..', 'basemap'), os.path.join(_HERE, '..')):
    if os.path.abspath(_p) not in map(os.path.abspath, sys.path):
        sys.path.insert(1, os.path.abspath(_p))
# Reuse the proven helpers (ocean/coastline/WA-mask/smoothing)
from plot_depth_slice import (_make_ax, _add_states_ocean, _mask_wa_points,
                              _plot_smooth, _plot_scatter, _cmap, _at_depth,
                              _HAS_CARTOPY)
from wa_basemap import add_tectonic_outlines, add_tectonic_background
try:
    import cartopy.crs as ccrs
except ImportError:
    pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fvs', required=True)
    ap.add_argument('--depths', nargs='+', type=float, default=[10,20,30,40])
    ap.add_argument('--field', default='vsv', choices=['vsv','xi','vpvs'])
    ap.add_argument('--overlay', default='outlines', choices=['outlines','filled','none'])
    ap.add_argument('--major', action='store_true')
    ap.add_argument('--smooth', action='store_true')
    ap.add_argument('--interp', default='rbf')
    ap.add_argument('--npts', type=int, default=200)
    ap.add_argument('--sigma', type=float, default=1.25)
    ap.add_argument('--ncolors', type=int, default=32)
    ap.add_argument('--min-ndat', type=int, default=35)
    ap.add_argument('--max-dist', type=float, default=0.5)
    ap.add_argument('--out', default='vsv_tectonic.png')
    import footprint as _fpmod
    _fpmod.add_args(ap)
    args = ap.parse_args()

    d = np.load(args.fvs, allow_pickle=True)
    Lon, Lat, z = d['Lon'], d['Lat'], d['z']
    arr = {'vsv':'Vsv','xi':'Xi','vpvs':'Vpvs'}[args.field]
    data = d[arr]
    cmapname = 'viridis' if args.field=='vpvs' else 'RdBu'

    wa = _mask_wa_points(Lon, Lat)
    Lon_wa, Lat_wa = Lon[wa], Lat[wa]
    ndat_wa = d['Ndat'][wa].astype(float) if 'Ndat' in d else None
    if args.min_ndat>0 and ndat_wa is not None:
        sm = ndat_wa >= args.min_ndat
    else:
        sm = np.ones(wa.sum(), dtype=bool)
    fp = _fpmod.from_args(args)
    if fp is not None:                       # selection: nodes inside the footprint mask
        inside = fp.contains(Lon_wa, Lat_wa)
        sm &= inside
        print('  ' + fp.describe(inside))
    Lon_sm, Lat_sm = Lon_wa[sm], Lat_wa[sm]

    tr = ccrs.PlateCarree() if _HAS_CARTOPY else None
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(f"{os.path.basename(args.fvs).replace('.npz','')} — {arr}",
                 fontsize=13, fontweight='bold')
    rects = [[0.04,0.52,0.42,0.42],[0.52,0.52,0.42,0.42],
             [0.04,0.04,0.42,0.42],[0.52,0.04,0.42,0.42]]

    for rect, zt in zip(rects, args.depths[:4]):
        ax = _make_ax(fig, rect)   # cartopy GeoAxes with WA extent
        v = _at_depth(data, z, zt)
        v_wa = v[wa]; v_sm = v_wa[sm]
        if args.field=='xi':
            vmin,vmax = 0.90, 1.10
        else:
            med,std = np.nanmedian(v_sm), np.nanstd(v_sm)
            vmin,vmax = med-args.sigma*std, med+args.sigma*std
        cmap = _cmap(cmapname, args.ncolors)

        # Optional filled tectonic UNDER the velocity
        if args.overlay=='filled':
            add_tectonic_background(ax, alpha=0.30, zorder=1,
                                    transform=tr, major_only=args.major)

        if args.smooth:
            im, *_ = _plot_smooth(ax, Lon_sm, Lat_sm, v_sm, cmap, vmin, vmax,
                                  method=args.interp, npts=args.npts,
                                  all_lon=Lon_wa, all_lat=Lat_wa,
                                  ndat=ndat_wa, ndat_min=args.min_ndat,
                                  max_dist=args.max_dist)
        else:
            Ls, As, Vs = (Lon_sm, Lat_sm, v_sm) if fp is not None else (Lon_wa, Lat_wa, v_wa)
            im = _plot_scatter(ax, Ls, As, Vs, cmap, vmin, vmax, s=120)

        # Ocean mask + coastline ON TOP of velocity (grey ocean hides offshore)
        _add_states_ocean(ax)
        # Tectonic OUTLINES on top (between velocity and ocean-edge)
        if args.overlay=='outlines':
            add_tectonic_outlines(ax, lw=0.4, alpha=0.65, zorder=6,
                                  transform=tr, major_only=args.major)

        if fp is not None:
            fp.focus(ax, tr, veil_alpha=args.footprint_veil, zoom=not args.no_footprint_zoom)
        ax.set_title(f'z={zt:.0f} km', fontsize=10)
        cblab = arr+(' (km/s)' if args.field!='xi' else '')
        if fp is not None:
            _fpmod.inset_colorbar(ax, im, cblab, extend='both')
        else:
            plt.colorbar(im, ax=ax, shrink=0.78, label=cblab, extend='both')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print('Saved:', args.out)

if __name__ == '__main__':
    main()
