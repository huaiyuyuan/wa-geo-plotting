#!/usr/bin/env python3
"""
plot_moho_map.py - a Moho-depth map on the GSWA-style Albers frame, reusing the
depth-slice map engine (tomo2d.MapFrame + the thin-plate RBF from
plot_site_depth_maps.grid_fields). A Moho map is just a single scalar field over
WA, so this is the same machinery as the Vs depth slices with the field being
Moho depth (km) from the CCP picker instead of Vs at a depth.

  source env_stage6.sh
  python3 plotting/plot_moho_map.py \
      --moho /workspace/CCP/CCP.WA.Array.2027.phase.3/Moho.picked/moho_ccp_refined_AR23grid.txt \
      --stations $ST/stations.phases.1.2.3.txt \
                 "$HOME/Downloads/New.WA.Seismic.2026- Perm-phase-0.csv" \
                 "$HOME/Downloads/New.WA.Seismic.2026- Phase 4.csv" \
      --out $FIG/moho/moho_ccp_refined.png
  # whole WA by default; --extent focus for the SW core (--extent nodes), or a box

Smooth interpolated surface (thin-plate RBF, cells > --max-dist deg from a node
blanked, clipped to WA), tectonic UNIT OUTLINES (not filled) + major crustal
boundaries, and all stations as one layer. Writes PNG + PDF.

The Moho grid is a text table (lon lat moho_km; the CCP picker's
moho_ccp_refined_AR23grid.txt has 11 header lines -> --skip-header 11, the
default). Station files may be the 'code lat lon' phase lists or the GSWA
'New.WA.Seismic' CSVs (Latitude/Longitude columns) - lon/lat are auto-detected
by WA range, so mixed formats merge into one layer.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', 'basemap'))
import tomo2d as t2                                              # noqa: E402
from plot_site_depth_maps import grid_fields                    # noqa: E402

WA_EXTENT = [112.0, 129.6, -35.8, -13.2]        # whole-state frame (as plot_site_depth_maps)
LON_RANGE = (110.0, 135.0)
LAT_RANGE = (-45.0, -10.0)


def _pick_lonlat(med):
    """Column indices (lon, lat) from per-column medians, by WA range."""
    loncol = [j for j, v in enumerate(med) if LON_RANGE[0] <= v <= LON_RANGE[1]]
    latcol = [j for j, v in enumerate(med) if LAT_RANGE[0] <= v <= LAT_RANGE[1]]
    return (loncol[0] if loncol else None), (latcol[0] if latcol else None)


def read_moho(path, skip):
    """lon, lat, moho from a text table; lon/lat columns auto-detected by range,
    the first remaining column is the Moho value."""
    A = np.loadtxt(path, skiprows=skip)
    if A.ndim != 2 or A.shape[1] < 3:
        sys.exit(f"{path}: expected >=3 numeric columns (lon lat moho) after "
                 f"{skip} header lines; got shape {A.shape}")
    med = np.nanmedian(A, axis=0)
    jlo, jla = _pick_lonlat(med)
    if jlo is None or jla is None:
        sys.exit(f"{path}: couldn't find lon/lat columns by WA range (medians {med})")
    jval = next(j for j in range(A.shape[1]) if j not in (jlo, jla))
    lon, lat, moho = A[:, jlo], A[:, jla], A[:, jval]
    good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(moho)
    return lon[good], lat[good], moho[good]


def load_any_stations(path):
    """lon, lat from a station file: GSWA 'New.WA.Seismic' CSV (Latitude/Longitude
    columns) or a whitespace 'code lat lon' list. lon/lat auto-detected by range."""
    first = ''
    with open(path) as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith('#'):
                first = line
                break
    if (',' in first) and any(k in first.lower() for k in ('lat', 'lon')):
        import plot_wa_map as pwm
        lo, la, _ = pwm._load_station_csv(path)
        return np.asarray(lo, float), np.asarray(la, float)
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            toks = s.replace(',', ' ').split()
            vals = []
            for tk in toks:
                try:
                    vals.append(float(tk))
                except ValueError:
                    vals.append(np.nan)
            if vals:
                rows.append(vals)
    if not rows:
        return np.array([]), np.array([])
    ncol = max(len(r) for r in rows)
    A = np.full((len(rows), ncol), np.nan)
    for i, r in enumerate(rows):
        A[i, :len(r)] = r
    import warnings
    with warnings.catch_warnings():                 # string columns -> all-NaN, ignore
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(A, axis=0)
    jlo, jla = _pick_lonlat(med)
    if jlo is None or jla is None:
        sys.exit(f"{path}: couldn't find lon/lat columns by WA range (medians {med})")
    lo, la = A[:, jlo], A[:, jla]
    good = np.isfinite(lo) & np.isfinite(la)
    return lo[good], la[good]


def wa_mask(glon, glat):
    """inside-WA boolean grid [nlat, nlon] from the frame's WA polygon (the same
    Natural Earth data MapFrame drew), with plot_depth_slice's cartopy WA polygon
    as a fallback. None if no WA polygon is available."""
    import matplotlib.path as mpath
    GLO, GLA = np.meshgrid(glon, glat)
    pts = np.column_stack([GLO.ravel(), GLA.ravel()])
    rings = None
    try:
        for name, rr in (getattr(t2, '_STATES', None) or {}).get('states', []):
            if name == 'Western Australia':
                rings = rr
                break
    except Exception:
        pass
    if rings:
        inside = np.zeros(len(pts), bool)
        for r in rings:
            inside |= mpath.Path(np.asarray(r)[:, :2]).contains_points(pts)
        return inside.reshape(GLO.shape)
    try:                                            # fallback: cartopy admin_1 WA
        import plot_depth_slice as pds
        pds._load_wa()
        if pds._WA_PATH is not None:
            return pds._WA_PATH.contains_points(pts).reshape(GLO.shape)
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--moho', required=True, help='Moho grid text table (lon lat moho_km)')
    ap.add_argument('--skip-header', type=int, default=11, help='header lines to skip (default 11)')
    ap.add_argument('--extent', default='wa',
                    help="wa (whole state, default) | nodes (data bbox +0.5 deg) | 'lon0,lon1,lat0,lat1'")
    ap.add_argument('--dgrid', type=float, default=0.05, help='map grid step (deg)')
    ap.add_argument('--max-dist', type=float, default=0.5, help='blank cells > this (deg) from a node')
    ap.add_argument('--clim', nargs=2, type=float, default=None, metavar=('LO', 'HI'),
                    help='colour limits (km); default 2-98th percentile of the data')
    ap.add_argument('--cmap', default='Spectral',
                    help='colormap (default Spectral: shallow warm/red, deep cool/blue)')
    ap.add_argument('--ncolors', type=int, default=0, help='discrete colour bands (0 = continuous)')
    ap.add_argument('--alpha', type=float, default=0.85, help='field opacity (lower = softer; 1 = solid)')
    ap.add_argument('--clip', default='wa', choices=['wa', 'none'],
                    help='wa (default): blank cells outside WA so offshore + other states show '
                         'as sea/grey; none: paint the field everywhere it is defined')
    ap.add_argument('--stations', nargs='+', default=None,
                    help='one or more station files (CSV or code/lat/lon), all drawn as one layer')
    ap.add_argument('--no-stations', action='store_true')
    ap.add_argument('--station-marker', default='^')
    ap.add_argument('--station-size', type=float, default=16)
    ap.add_argument('--station-colour', default='k')
    ap.add_argument('--no-units', action='store_true', help='no tectonic unit outlines')
    ap.add_argument('--no-boundaries', action='store_true', help='no major crustal boundaries')
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--bnd-npz', default=None)
    ap.add_argument('--grat', type=float, default=None, help='graticule step (deg)')
    ap.add_argument('--title', default='WA Array refined CCP Moho')
    ap.add_argument('--out', default='moho_map.png')
    a = ap.parse_args()

    lon, lat, moho = read_moho(a.moho, a.skip_header)
    print(f"Moho: {len(lon)} nodes <- {os.path.basename(a.moho)} "
          f"({moho.min():.1f}-{moho.max():.1f} km)")

    if a.extent == 'wa':
        ext = list(WA_EXTENT)
    elif a.extent == 'nodes':
        ext = [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]
    else:
        ext = [float(x) for x in a.extent.split(',')]
    glon = np.arange(ext[0], ext[1] + 1e-6, a.dgrid)
    glat = np.arange(ext[2], ext[3] + 1e-6, a.dgrid)

    # smooth field (thin-plate RBF), cells > max_dist from a node blanked
    G, ok = grid_fields(lon, lat, moho[:, None], glon, glat, a.max_dist)
    Z = G[0]

    if a.clim:
        vmin, vmax = a.clim
    else:
        vmin = float(np.floor(np.nanpercentile(moho, 2)))
        vmax = float(np.ceil(np.nanpercentile(moho, 98)))
    cmap = matplotlib.colormaps[a.cmap]
    if a.ncolors > 0:
        cmap = cmap.resampled(a.ncolors)
    norm = Normalize(vmin, vmax)

    # figure sized to the projected extent aspect
    P = t2.Albers()
    bx, by = P(np.r_[ext[0], ext[1], ext[1], ext[0]], np.r_[ext[2], ext[2], ext[3], ext[3]])
    asp = (by.max() - by.min()) / (bx.max() - bx.min())
    w = 8.0
    fig = plt.figure(figsize=(w + 1.4, w * asp + 1.0))
    ax = fig.add_axes([0.04, 0.06, 0.80, 0.88])
    F = t2.MapFrame(ax, ext, grat=a.grat)

    # clip to WA so offshore + other states are not painted (shown as sea/grey)
    if a.clip == 'wa':
        wam = wa_mask(glon, glat)
        if wam is not None:
            ok = ok & wam
            print(f"  clipped to WA: {int(ok.sum())} of {ok.size} cells kept")
        else:
            print("  (WA clip requested but no WA polygon available - not clipped)")

    pm = F.raster(glat, glon, Z, ok, cmap, norm, zorder=2)
    if a.alpha < 1.0:
        pm.set_alpha(a.alpha)

    # tectonic unit outlines (not filled) + major crustal boundaries
    if not a.no_units:
        try:
            import plot_xsection as px
            from domains import Domains
            tect_npz = px._resolve_npz('TECTONICS_NPZ', 'wa_tectonics.npz', a.tect_npz)
            rings = [r for r in Domains(tect_npz).rings if len(r) >= 3]
            n = F.lines(rings, colors='0.35', linewidths=0.3, zorder=4)
            print(f"tectonic unit outlines: {n}")
        except Exception as e:
            print(f"  (no tectonic unit outlines: {e})")
    if not a.no_boundaries:
        try:
            import plot_xsection as px
            bnd_npz = px._resolve_npz('BOUNDARIES_NPZ', 'wa_crustal_boundaries.npz', a.bnd_npz)
            B = np.load(bnd_npz, allow_pickle=True)
            blines = [np.asarray(ln, float)[:, :2] for ln in B['lines'] if len(ln) >= 2]
            n = F.lines(blines, colors='k', linewidths=0.8, zorder=5)
            print(f"crustal boundaries: {n}")
        except Exception as e:
            print(f"  (no crustal boundaries: {e})")

    # all stations, one layer
    if not a.no_stations and a.stations:
        SL, SA = [], []
        for sp in a.stations:
            if not os.path.isfile(sp):
                print(f"  (stations: not found, skipped: {sp})")
                continue
            lo, la = load_any_stations(sp)
            SL.append(lo); SA.append(la)
            print(f"  stations: {len(lo)} <- {os.path.basename(sp)}")
        if SL:
            slon = np.concatenate(SL); slat = np.concatenate(SA)
            x, y = F.xy(slon, slat)
            ax.scatter(x, y, marker=a.station_marker, s=a.station_size,
                       c=a.station_colour, edgecolors='w', linewidths=0.4, zorder=8)

    cax = fig.add_axes([0.86, 0.12, 0.022, 0.72])
    cb = fig.colorbar(pm, cax=cax, extend='both')
    cb.set_label('Moho depth (km)', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(a.title, fontsize=12)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=220)
    fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")


if __name__ == '__main__':
    main()
