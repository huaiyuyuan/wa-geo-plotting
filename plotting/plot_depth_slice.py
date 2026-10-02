#!/usr/bin/env python3
"""
plot_depth_slice.py — 2×2 depth slice maps from TransD 1D Vs results.
Panels: absolute Vsv | relative dlnVsv | data misfit | data coverage

Usage:
  python3 plot_depth_slice.py --fvs Fvs.iter.1.Z.npz --depths 10 20 30 --smooth
  python3 plot_depth_slice.py --fvs Fvs.iter.1.Z.npz --depths 30 --field vpvs --smooth
  python3 plot_depth_slice.py --fvs Fvs.iter.1.Z.npz --depths 30 --smooth --contour 8

H. Yuan / Claude, Sep 2026
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Cartopy data dir
for _c in ['/data/25/yuan/mfiles/cartopy_data',
           '/mnt/data64/yuan/mfiles/cartopy_data',
           os.path.expanduser('~/.local/share/cartopy')]:
    if os.path.isdir(_c) and not os.environ.get('CARTOPY_DATA_DIR'):
        os.environ['CARTOPY_DATA_DIR'] = _c; break

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

# ── WA polygon (cached) ─────────────────────────────────────────────────────
_WA_POLY = None
_WA_PATH = None   # matplotlib.path.Path for fast grid masking

def _load_wa():
    global _WA_POLY, _WA_PATH
    if _WA_POLY is not None:
        return
    if not _HAS_CARTOPY:
        return
    try:
        import cartopy.io.shapereader as shpreader
        import matplotlib.path as mpath
        shp = shpreader.natural_earth('50m', 'cultural', 'admin_1_states_provinces')
        for rec in shpreader.Reader(shp).records():
            if rec.attributes.get('name') == 'Western Australia':
                _WA_POLY = rec.geometry
                if hasattr(_WA_POLY, 'geoms'):
                    ext = max(_WA_POLY.geoms, key=lambda g: g.area).exterior
                else:
                    ext = _WA_POLY.exterior
                _WA_PATH = mpath.Path(np.array(ext.coords))
                break
    except Exception as e:
        print(f"WARNING: WA polygon load failed: {e}")

def _mask_grid_to_wa(GLON, GLAT, GVAL):
    """NaN grid cells outside WA polygon."""
    _load_wa()
    if _WA_PATH is None:
        return GVAL
    pts = np.column_stack([GLON.ravel(), GLAT.ravel()])
    inside = _WA_PATH.contains_points(pts)
    GVAL[~inside.reshape(GLON.shape)] = np.nan
    return GVAL

def _mask_grid_ndat(GLON, GLAT, GVAL, all_lon, all_lat, ndat, ndat_min):
    """NaN grid cells whose nearest WA node has ndat < ndat_min."""
    if ndat is None or ndat_min <= 0 or len(all_lon) != len(ndat):
        return GVAL
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([all_lon, all_lat]))
    _, idx = tree.query(np.column_stack([GLON.ravel(), GLAT.ravel()]))
    bad = ndat.astype(float)[idx].reshape(GLON.shape) < ndat_min
    GVAL[bad] = np.nan
    return GVAL

def _mask_grid_maxdist(GLON, GLAT, GVAL, lon, lat, max_dist):
    """NaN grid cells further than max_dist degrees from any data node."""
    if max_dist <= 0 or len(lon) == 0:
        return GVAL
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([lon, lat]))
    dists, _ = tree.query(np.column_stack([GLON.ravel(), GLAT.ravel()]))
    GVAL[dists.reshape(GLON.shape) > max_dist] = np.nan
    return GVAL

def _mask_wa_points(lon, lat):
    """Boolean mask: True where point is inside WA (for scatter)."""
    _load_wa()
    if _WA_POLY is None:
        return np.ones(len(lon), dtype=bool)
    try:
        from shapely.geometry import Point
        return np.array([_WA_POLY.contains(Point(x, y)) for x, y in zip(lon, lat)])
    except ImportError:
        return np.ones(len(lon), dtype=bool)

# ── Map axes ─────────────────────────────────────────────────────────────────
def _make_ax(fig, rect):
    if _HAS_CARTOPY:
        ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
        ax.set_facecolor('lightgrey')
        ax.coastlines('50m', linewidth=0.8, color='k', zorder=10)
    else:
        ax = fig.add_axes(rect)
        ax.set_facecolor('lightgrey')
        ax.set_aspect('equal')
    return ax

def _add_states_ocean(ax):
    """Add grey ocean + non-WA states overlay (for scatter panels)."""
    if not _HAS_CARTOPY: return
    ax.add_feature(cfeature.OCEAN, facecolor='lightgrey', zorder=5)
    try:
        import cartopy.io.shapereader as shpreader
        shp = shpreader.natural_earth('50m', 'cultural', 'admin_1_states_provinces')
        for rec in shpreader.Reader(shp).records():
            if (rec.attributes.get('admin','') == 'Australia' and
                    rec.attributes.get('name','') != 'Western Australia'):
                ax.add_geometries([rec.geometry], ccrs.PlateCarree(),
                                  facecolor='lightgrey', edgecolor='0.5',
                                  linewidth=0.4, zorder=5)
    except Exception: pass
    ax.coastlines('50m', linewidth=0.8, color='k', zorder=10)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=10)
    _load_wa()
    if _WA_POLY is not None:
        ax.add_geometries([_WA_POLY], ccrs.PlateCarree(),
                          facecolor='none', edgecolor='k',
                          linewidth=0.8, zorder=11)

# ── Interpolation ─────────────────────────────────────────────────────────────
def _interp_to_grid(lon, lat, val, npts=200,
                    lon_range=None, lat_range=None, method='rbf'):
    lo0 = lon_range[0] if lon_range else lon.min() - 0.3
    lo1 = lon_range[1] if lon_range else lon.max() + 0.3
    la0 = lat_range[0] if lat_range else lat.min() - 0.3
    la1 = lat_range[1] if lat_range else lat.max() + 0.3
    glo = np.linspace(lo0, lo1, npts)
    gla = np.linspace(la0, la1, npts)
    GLON, GLAT = np.meshgrid(glo, gla)
    pts = np.column_stack([lon, lat])
    qpts = np.column_stack([GLON.ravel(), GLAT.ravel()])

    if method == 'rbf':
        try:
            from scipy.interpolate import RBFInterpolator
            GVAL = RBFInterpolator(pts, val, kernel='thin_plate_spline',
                                   smoothing=0.01)(qpts).reshape(GLON.shape)
        except ImportError:
            method = 'cubic'
    if method == 'cubic':
        from scipy.interpolate import griddata
        GVAL = griddata(pts, val, (GLON, GLAT), method='cubic')
        nan_m = np.isnan(GVAL)
        if nan_m.any():
            GVAL[nan_m] = griddata(pts, val, (GLON[nan_m], GLAT[nan_m]), method='linear')
    if method == 'linear':
        from scipy.interpolate import griddata
        GVAL = griddata(pts, val, (GLON, GLAT), method='linear')
    if method == 'kriging':
        try:
            from pykrige.ok import OrdinaryKriging
            ok = OrdinaryKriging(lon, lat, val, variogram_model='spherical',
                                 verbose=False, enable_plotting=False)
            GVAL, _ = ok.execute('grid', glo, gla)
        except ImportError:
            return _interp_to_grid(lon, lat, val, npts, lon_range, lat_range, 'rbf')
    return GLON, GLAT, GVAL

def _plot_smooth(ax, lon, lat, val, cmap, vmin, vmax,
                 method='rbf', npts=200,
                 all_lon=None, all_lat=None, ndat=None, ndat_min=0,
                 max_dist=0.5):
    GLON, GLAT, GVAL = _interp_to_grid(lon, lat, val, npts, method=method)
    # Apply masks in order
    GVAL = _mask_grid_maxdist(GLON, GLAT, GVAL, lon, lat, max_dist)
    GVAL = _mask_grid_ndat(GLON, GLAT, GVAL,
                           all_lon if all_lon is not None else lon,
                           all_lat if all_lat is not None else lat,
                           ndat, ndat_min)
    GVAL = _mask_grid_to_wa(GLON, GLAT, GVAL)
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, zorder=2, shading='auto')
    if _HAS_CARTOPY:
        im = ax.pcolormesh(GLON, GLAT, GVAL, transform=ccrs.PlateCarree(), **kw)
    else:
        im = ax.pcolormesh(GLON, GLAT, GVAL, **kw)
    return im, GLON, GLAT, GVAL

def _plot_scatter(ax, lon, lat, val, cmap, vmin, vmax, s=120):
    fin = np.isfinite(val)
    kw = dict(c=val[fin], cmap=cmap, vmin=vmin, vmax=vmax,
              s=s, linewidths=0, zorder=2)
    if _HAS_CARTOPY:
        return ax.scatter(lon[fin], lat[fin], transform=ccrs.PlateCarree(), **kw)
    return ax.scatter(lon[fin], lat[fin], **kw)

# ── Colormap helper ───────────────────────────────────────────────────────────
def _cmap(name, n):
    try:
        return matplotlib.colormaps[name].resampled(n)
    except Exception:
        return plt.cm.get_cmap(name, n)

# ── Depth interpolation ───────────────────────────────────────────────────────
def _at_depth(data, z, zt):
    iz = np.searchsorted(z, zt)
    if iz == 0: return data[:, 0]
    if iz >= len(z): return data[:, -1]
    f = (zt - z[iz-1]) / (z[iz] - z[iz-1])
    return data[:, iz-1] * (1-f) + data[:, iz] * f

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs',     required=True)
    ap.add_argument('--depths',  nargs='+', type=float, default=[30])
    ap.add_argument('--field',   default='vsv',
                    choices=['vsv','vpvs','xi','pani'])
    ap.add_argument('--smooth',  action='store_true')
    ap.add_argument('--interp',  default='rbf',
                    choices=['rbf','cubic','linear','kriging'])
    ap.add_argument('--npts',    type=int,   default=200)
    ap.add_argument('--contour',     type=int, default=0,
                    help='N contour lines on absolute panel (0=none)')
    ap.add_argument('--contour-rel', type=int, default=0,
                    help='N contour lines on relative panel (0=none, default)')
    ap.add_argument('--markersize', type=float, default=120)
    ap.add_argument('--ncolors',    type=int,   default=32)
    ap.add_argument('--sigma-abs',  type=float, default=1.5,
                    help='Auto colorlim = median ± sigma*std (default 1.5)')
    ap.add_argument('--vmin-abs',   type=float, default=None)
    ap.add_argument('--vmax-abs',   type=float, default=None)
    ap.add_argument('--clim-rel',   type=float, default=6.0)
    ap.add_argument('--vmin-rel',   type=float, default=None)
    ap.add_argument('--vmax-rel',   type=float, default=None)
    ap.add_argument('--min-ndat',   type=int,   default=35,
                    help='Grey out nodes with <N measurements (default 35)')
    ap.add_argument('--max-dist',   type=float, default=0.5,
                    help='Max deg from nearest node for smooth panels (default 0.5)')
    ap.add_argument('--no-rel',  action='store_true')
    ap.add_argument('--out-dir', default='figures/depth_slices')
    ap.add_argument('--cmap-abs', default='RdBu')
    ap.add_argument('--cmap-rel', default='RdBu')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.fvs, allow_pickle=True)
    Lon = d['Lon']; Lat = d['Lat']; z = d['z']

    # Field selection
    field_cfg = {
        'vsv':  ('Vsv',  'Vsv (km/s)',     'dlnVsv',  'dlnVsv (%)', 'RdBu',   'RdBu'),
        'vpvs': ('Vpvs', 'Vp/Vs',          'dlnVpvs', 'dlnVpvs (%)','RdBu',   'RdBu'),
        'xi':   ('Xi',   'xi = Vsh/Vsv',   'dlnXi',   'dlnXi (%)',  'RdBu_r', 'RdBu'),
        'pani': ('Pani', 'P(aniso) %',      None,       None,        'YlOrRd', None),
    }
    fkey, flabel, dkey, dlabel, cabs_def, crel_def = field_cfg[args.field]
    DATA  = d[fkey]
    dlnD  = d[dkey] if (dkey and dkey in d) else None
    cmap_abs = _cmap(args.cmap_abs if args.cmap_abs != 'RdBu' else cabs_def, args.ncolors)
    cmap_rel = _cmap(args.cmap_rel if args.cmap_rel != 'RdBu' else (crel_def or 'RdBu'), args.ncolors)

    # WA mask for all points
    print("Computing WA mask ...")
    wa_mask = _mask_wa_points(Lon, Lat)
    Lon_wa = Lon[wa_mask]; Lat_wa = Lat[wa_mask]

    # Ndat
    ndat_wa = d['Ndat'][wa_mask].astype(float) if 'Ndat' in d else None
    if ndat_wa is not None:
        print(f"  Ndat: mean={ndat_wa.mean():.1f} range={ndat_wa.min():.0f}-{ndat_wa.max():.0f}")

    # Smooth mask (ndat filtered)
    if args.min_ndat > 0 and ndat_wa is not None:
        sm = ndat_wa >= args.min_ndat
        n_grey = (~sm).sum()
        if n_grey: print(f"  Ndat mask: {n_grey} nodes greyed (Ndat<{args.min_ndat})")
        Lon_sm = Lon_wa[sm]; Lat_sm = Lat_wa[sm]
    else:
        Lon_sm = Lon_wa; Lat_sm = Lat_wa

    for zt in args.depths:
        if zt < z[0] or zt > z[-1]:
            print(f"WARNING: {zt} km outside range"); continue

        val_abs = _at_depth(DATA, z, zt)
        val_wa  = val_abs[wa_mask]
        val_sm  = val_abs[wa_mask][ndat_wa >= args.min_ndat] if \
                  (args.min_ndat > 0 and ndat_wa is not None) else val_wa

        # Colorlimits
        if args.vmin_abs is not None and args.vmax_abs is not None:
            vmin_a, vmax_a = args.vmin_abs, args.vmax_abs
        else:
            med = np.nanmedian(val_sm); std = np.nanstd(val_sm)
            vmin_a = med - args.sigma_abs * std
            vmax_a = med + args.sigma_abs * std
        print(f"  z={zt:.0f} km  {flabel}: {vmin_a:.3f} – {vmax_a:.3f}")

        # Relative
        val_mean = np.nanmean(val_sm)
        rel_wa = (val_wa - val_mean) / val_mean * 100 if val_mean else None
        rel_sm = (val_sm - val_mean) / val_mean * 100 if val_mean else None
        if args.vmin_rel is not None and args.vmax_rel is not None:
            vmin_r, vmax_r = args.vmin_rel, args.vmax_rel
        else:
            vmin_r, vmax_r = -args.clim_rel, args.clim_rel

        # Figure
        fig = plt.figure(figsize=(13, 13))
        title = (f"{os.path.basename(args.fvs).replace('.npz','')}  —  "
                 f"{flabel}  z={zt:.0f} km  (2×2)")
        fig.suptitle(title, fontsize=11, fontweight='bold')

        # ── Panel 1: Absolute ────────────────────────────────────────────────
        ax1 = _make_ax(fig, [0.04, 0.53, 0.40, 0.42])
        if args.smooth:
            im1, g1lo, g1la, g1v = _plot_smooth(
                ax1, Lon_sm, Lat_sm, val_sm, cmap_abs, vmin_a, vmax_a,
                method=args.interp, npts=args.npts,
                all_lon=Lon_wa, all_lat=Lat_wa,
                ndat=ndat_wa, ndat_min=args.min_ndat,
                max_dist=args.max_dist)
            if args.contour > 0:
                ck = dict(colors='k', linewidths=0.4, alpha=0.5)
                if _HAS_CARTOPY: ck['transform'] = ccrs.PlateCarree()
                ax1.contour(g1lo, g1la, g1v, args.contour, **ck)
        else:
            val_sc = val_wa.copy()
            if args.min_ndat > 0 and ndat_wa is not None:
                val_sc[ndat_wa < args.min_ndat] = np.nan
            im1 = _plot_scatter(ax1, Lon_wa, Lat_wa, val_sc,
                                cmap_abs, vmin_a, vmax_a, s=args.markersize)
        plt.colorbar(im1, ax=ax1, shrink=0.8, label=flabel, extend='both')
        ax1.set_title(f'Absolute  (mean={val_mean:.3f})', fontsize=9)
        _add_states_ocean(ax1)

        # ── Panel 2: Relative ────────────────────────────────────────────────
        ax2 = _make_ax(fig, [0.53, 0.53, 0.40, 0.42])
        if rel_sm is not None and not args.no_rel and dlnD is not None:
            if args.smooth:
                im2, g2lo, g2la, g2v = _plot_smooth(
                    ax2, Lon_sm, Lat_sm, rel_sm, cmap_rel, vmin_r, vmax_r,
                    method=args.interp, npts=args.npts,
                    all_lon=Lon_wa, all_lat=Lat_wa,
                    ndat=ndat_wa, ndat_min=args.min_ndat,
                    max_dist=args.max_dist)
                if args.contour_rel > 0:
                    ck = dict(colors='k', linewidths=0.4, alpha=0.5)
                    if _HAS_CARTOPY: ck['transform'] = ccrs.PlateCarree()
                    ax2.contour(g2lo, g2la, g2v, args.contour_rel, **ck)
            else:
                rel_sc = rel_wa.copy()
                if args.min_ndat > 0 and ndat_wa is not None:
                    rel_sc[ndat_wa < args.min_ndat] = np.nan
                im2 = _plot_scatter(ax2, Lon_wa, Lat_wa, rel_sc,
                                    cmap_rel, vmin_r, vmax_r, s=args.markersize)
            lbl = f'Relative ({vmin_r:.0f} to +{vmax_r:.0f}%)' \
                  if args.vmin_rel else f'Relative (±{args.clim_rel:.0f}%)'
            plt.colorbar(im2, ax=ax2, shrink=0.8, label=dlabel or 'dlnV (%)', extend='both')
            ax2.set_title(lbl, fontsize=9)
        else:
            ax2.set_visible(False)
        _add_states_ocean(ax2)

        # ── Panel 3: Misfit ──────────────────────────────────────────────────
        ax3 = _make_ax(fig, [0.04, 0.05, 0.40, 0.42])
        if 'Misfit' in d:
            mis = d['Misfit'][wa_mask].copy().astype(float)
            if args.min_ndat > 0 and ndat_wa is not None:
                mis[ndat_wa < args.min_ndat] = np.nan
            fin = np.isfinite(mis)
            if fin.sum() > 10:
                vmax_mis = np.nanpercentile(mis[fin], 90)
                im3 = _plot_scatter(ax3, Lon_wa[fin], Lat_wa[fin], mis[fin],
                                    _cmap('YlOrRd', args.ncolors), 0, vmax_mis,
                                    s=args.markersize)
                plt.colorbar(im3, ax=ax3, shrink=0.8,
                             label='RMS misfit (km/s)', extend='max')
                ax3.set_title('Data misfit (RMS) — per node', fontsize=9)
        _add_states_ocean(ax3)

        # ── Panel 4: Coverage ────────────────────────────────────────────────
        ax4 = _make_ax(fig, [0.53, 0.05, 0.40, 0.42])
        if 'Ndat' in d:
            nd = d['Ndat'][wa_mask].astype(float)
            im4 = _plot_scatter(ax4, Lon_wa, Lat_wa, nd,
                                _cmap('Blues', args.ncolors), 0, nd.max(),
                                s=args.markersize)
            plt.colorbar(im4, ax=ax4, shrink=0.8, label='N measurements')
            ax4.set_title('Data coverage (N measurements)', fontsize=9)
        _add_states_ocean(ax4)

        out = os.path.join(args.out_dir,
                           f"depth_slice.{args.field}.{zt:.0f}km.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")

if __name__ == '__main__':
    main()
