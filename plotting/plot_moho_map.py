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


def interp_field(lon, lat, val, glon, glat, method, max_dist):
    """Interpolate val(lon,lat) onto the glat x glon grid. 'rbf' uses the shared
    thin-plate grid_fields; 'kriging' ordinary kriging (pykrige, falls back to rbf
    if unavailable); 'cubic'/'linear' scipy griddata. Returns (Z, ok), ok True
    where a data node is within max_dist deg."""
    if method == 'rbf':
        G, ok = grid_fields(lon, lat, val[:, None], glon, glat, max_dist)
        return G[0], ok
    GLO, GLA = np.meshgrid(glon, glat)
    if method == 'kriging':
        try:
            from pykrige.ok import OrdinaryKriging
            if len(lon) > 3000:
                print(f"  (kriging on {len(lon)} nodes may be slow / memory-heavy)")
            okr = OrdinaryKriging(lon, lat, val, variogram_model='spherical',
                                  verbose=False, enable_plotting=False)
            Zg, _ = okr.execute('grid', glon, glat)
            Z = np.asarray(Zg, float)
        except Exception as e:
            print(f"  (kriging unavailable: {e}; using rbf)")
            G, ok = grid_fields(lon, lat, val[:, None], glon, glat, max_dist)
            return G[0], ok
    else:
        from scipy.interpolate import griddata
        Z = griddata(np.column_stack([lon, lat]), val, (GLO, GLA), method=method)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.column_stack([lon, lat])).query(np.column_stack([GLO.ravel(), GLA.ravel()]))
    ok = (d <= max_dist).reshape(GLO.shape) & np.isfinite(Z)
    return Z, ok


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
                    help="wa (whole state, default) | nodes (data bbox +0.5 deg) | "
                         "coverage (bbox of --coverage stations) | footprint (bbox of the WA "
                         "Array footprint, with --footprint) | 'lon0,lon1,lat0,lat1'")
    ap.add_argument('--dgrid', type=float, default=0.05, help='map grid step (deg)')
    ap.add_argument('--max-dist', type=float, default=0.5, help='blank cells > this (deg) from a node')
    ap.add_argument('--coverage', nargs='+', default=None,
                    help='station file(s) whose neighbourhood defines where to show Moho; cells '
                         'further than --coverage-dist from all of them are masked (e.g. a phases '
                         '1-3 coverage version: --coverage $ST/stations.phases.1.2.3.txt)')
    ap.add_argument('--coverage-dist', type=float, default=1.0,
                    help='coverage radius in deg (default 1.0; measured in projected km so the '
                         'disc is a true circle on the map)')
    ap.add_argument('--interp', default='rbf', choices=['rbf', 'kriging', 'cubic', 'linear'],
                    help='field interpolation (default rbf = thin-plate spline; kriging via pykrige)')
    ap.add_argument('--footprint', action='store_true',
                    help='mask Moho to the WA Array coverage footprint (the array outline); use '
                         'with --extent footprint to frame to it. This is the phases 1-3 area.')
    ap.add_argument('--footprint-npz', default=None, help='override config.FOOTPRINT_NPZ')
    ap.add_argument('--footprint-outline', default=None, help='override config.FOOTPRINT_OUTLINE')
    ap.add_argument('--draw-footprint', action='store_true', help='also draw the footprint outline')
    ap.add_argument('--draw-coverage', action='store_true',
                    help='draw the boundary of the --coverage region (handy to tune --coverage-dist '
                         'so the effective area aligns to a target outline)')
    ap.add_argument('--clip-poly', default=None,
                    help='lon/lat polygon file; the effective area is also restricted to inside it '
                         '(intersected with coverage/footprint), e.g. a hand-drawn N/E boundary')
    ap.add_argument('--draw-clip-poly', action='store_true', help='draw the --clip-poly outline')
    ap.add_argument('--smooth-edge', type=float, default=0.0,
                    help='smooth the effective-area boundary: Gaussian sigma in degrees (e.g. 0.3) '
                         'to round off the scalloped coverage edge; also fills small interior holes')
    ap.add_argument('--clim', nargs=2, type=float, default=[30.0, 50.0], metavar=('LO', 'HI'),
                    help='colour limits (km), default 30 50; values beyond saturate to the end colours')
    ap.add_argument('--cmap', default='Spectral',
                    help='colormap (default Spectral: shallow warm/red, deep cool/blue)')
    ap.add_argument('--ncolors', type=int, default=16,
                    help='discrete colour bands (default 16; 0 = continuous)')
    ap.add_argument('--alpha', type=float, default=0.85, help='field opacity (lower = softer; 1 = solid)')
    ap.add_argument('--clip', default='wa', choices=['wa', 'none'],
                    help='wa (default): blank cells outside WA so offshore + other states show '
                         'as sea/grey; none: paint the field everywhere it is defined')
    ap.add_argument('--mask-colour', default='0.8',
                    help="grey (or any colour) filling the masked area outside WA (default 0.8); "
                         "'none' leaves it as the frame's sea/states")
    ap.add_argument('--stations', nargs='+', default=None,
                    help='one or more station files (CSV or code/lat/lon), all drawn as one layer')
    ap.add_argument('--no-stations', action='store_true')
    ap.add_argument('--station-marker', default='^')
    ap.add_argument('--station-size', type=float, default=16)
    ap.add_argument('--station-colour', default='k')
    ap.add_argument('--station-exclude-box', nargs=4, type=float, action='append', default=[],
                    metavar=('LON0', 'LON1', 'LAT0', 'LAT1'),
                    help='drop stations inside this lon/lat box (repeatable); e.g. drop the '
                         'Phase 4 eastern grid from the full map: --station-exclude-box 124.5 131 -30 -22')
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

    # optional coverage: stations whose neighbourhood defines where Moho is shown
    cov_lon = cov_lat = None
    if a.coverage:
        CL, CA = [], []
        for sp in a.coverage:
            if not os.path.isfile(sp):
                print(f"  (coverage: not found, skipped: {sp})"); continue
            clo, cla = load_any_stations(sp)
            CL.append(clo); CA.append(cla)
        if CL:
            cov_lon = np.concatenate(CL); cov_lat = np.concatenate(CA)
            print(f"  coverage: {len(cov_lon)} stations, Moho shown within {a.coverage_dist} deg")

    fp = None
    if a.footprint:
        try:
            from footprint import Footprint
            fp = Footprint(a.footprint_npz, a.footprint_outline)
            print(f"  footprint: {fp.outline_source}")
        except Exception as e:
            print(f"  (footprint unavailable: {e})")

    if a.extent == 'wa':
        ext = list(WA_EXTENT)
    elif a.extent == 'nodes':
        ext = [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]
    elif a.extent == 'coverage':
        if cov_lon is None:
            sys.exit("--extent coverage needs --coverage STATIONS")
        m = max(1.0, a.coverage_dist)
        ext = [cov_lon.min() - m, cov_lon.max() + m, cov_lat.min() - m, cov_lat.max() + m]
    elif a.extent == 'footprint':
        if fp is None or fp.olon is None:
            sys.exit("--extent footprint needs --footprint with an outline")
        m = 0.5
        ext = [fp.olon.min() - m, fp.olon.max() + m, fp.olat.min() - m, fp.olat.max() + m]
    else:
        ext = [float(x) for x in a.extent.split(',')]
    glon = np.arange(ext[0], ext[1] + 1e-6, a.dgrid)
    glat = np.arange(ext[2], ext[3] + 1e-6, a.dgrid)

    # interpolated field (--interp), cells > max_dist from a node blanked
    Z, ok = interp_field(lon, lat, moho, glon, glat, a.interp, a.max_dist)
    GLO, GLA = np.meshgrid(glon, glat)

    # coverage mask: Moho within --coverage-dist of a coverage station. Distance is
    # in projected (Albers km) space, so a single-station disc is a true circle on
    # the map, not a lon/lat ellipse tilted off the 121E central meridian.
    cov_mask = None
    if cov_lon is not None:
        from scipy.spatial import cKDTree
        Pp = t2.Albers()
        sx, sy = Pp(cov_lon, cov_lat)
        gx, gy = Pp(GLO.ravel(), GLA.ravel())
        d, _ = cKDTree(np.column_stack([sx, sy])).query(np.column_stack([gx, gy]))
        cov_mask = (d <= a.coverage_dist * 111.195).reshape(GLO.shape)
        ok = ok & cov_mask

    # footprint mask: Moho everywhere inside the array OUTLINE (the black margin
    # line), not just the coverage-selection mask (which has interior holes), and
    # relaxed past --max-dist so the whole interior of the line fills in.
    fp_mask = None
    if fp is not None:
        if fp.olon is not None:
            import matplotlib.path as mpath
            poly = mpath.Path(np.column_stack([fp.olon, fp.olat]))
            fp_mask = poly.contains_points(
                np.column_stack([GLO.ravel(), GLA.ravel()])).reshape(GLO.shape)
        else:
            fp_mask = np.asarray(fp.contains(GLO.ravel(), GLA.ravel()), bool).reshape(GLO.shape)
        ok = np.isfinite(Z) & fp_mask
        if cov_mask is not None:
            ok = ok & cov_mask

    # extra polygon clip (e.g. a hand-drawn N/E boundary), intersected with the above
    clip_xy = None
    clip_mask = None
    if a.clip_poly:
        import matplotlib.path as mpath
        clip_xy = np.loadtxt(a.clip_poly, comments='#', usecols=(0, 1))
        clip_mask = mpath.Path(clip_xy).contains_points(
            np.column_stack([GLO.ravel(), GLA.ravel()])).reshape(GLO.shape)
        ok = ok & clip_mask
        print(f"  clip-poly: {len(clip_xy)} vertices <- {os.path.basename(a.clip_poly)}")

    vmin, vmax = a.clim
    if a.ncolors > 0:                               # discrete bands (crisp, not continuous)
        from matplotlib.colors import BoundaryNorm
        cmap = matplotlib.colormaps[a.cmap].resampled(a.ncolors)
        cmap = cmap.with_extremes(under=cmap(0), over=cmap(cmap.N - 1))
        norm = BoundaryNorm(np.linspace(vmin, vmax, a.ncolors + 1), a.ncolors)
    else:
        cmap = matplotlib.colormaps[a.cmap]
        norm = Normalize(vmin, vmax)

    # figure sized to the projected extent aspect
    P = t2.Albers()
    bx, by = P(np.r_[ext[0], ext[1], ext[1], ext[0]], np.r_[ext[2], ext[2], ext[3], ext[3]])
    asp = (by.max() - by.min()) / (bx.max() - bx.min())
    w = 8.0
    fig = plt.figure(figsize=(w + 1.4, w * asp + 1.0))
    ax = fig.add_axes([0.04, 0.06, 0.80, 0.88])
    F = t2.MapFrame(ax, ext, grat=a.grat)

    # clip to WA so offshore + other states are not painted. With --footprint the
    # outline is the sole boundary, so skip the WA clip (it would otherwise carve
    # grey slivers inside the line where the smooth outline sits just seaward of
    # the jagged coast).
    wam = None
    if a.clip == 'wa' and fp is None:
        wam = wa_mask(glon, glat)
        if wam is not None:
            ok = ok & wam
            print(f"  clipped to WA: {int(ok.sum())} of {ok.size} cells kept")
        else:
            print("  (WA clip requested but no WA polygon available - not clipped)")

    # round off the scalloped coverage edge and fill small interior holes
    smoothed = False
    if a.smooth_edge > 0:
        from scipy.ndimage import gaussian_filter, binary_fill_holes
        okf = gaussian_filter(ok.astype(float), a.smooth_edge / a.dgrid) > 0.5
        ok = binary_fill_holes(okf) & np.isfinite(Z)
        smoothed = True

    pm = F.raster(glat, glon, Z, ok, cmap, norm, zorder=2)
    if a.alpha < 1.0:
        pm.set_alpha(a.alpha)

    # cover the masked area with solid grey (outside WA, and outside coverage when
    # given), above the field and the line work there, below the stations
    if smoothed:                                    # grey matches the smoothed edge
        veil_inside = ok
    else:
        veil_inside = None
        for mk in (wam, cov_mask, fp_mask, clip_mask):
            if mk is not None:
                veil_inside = mk if veil_inside is None else (veil_inside & mk)
    if veil_inside is not None and a.mask_colour.lower() != 'none':
        F.veil(glat, glon, veil_inside, colour=a.mask_colour, alpha=1.0, zorder=7)

    if fp is not None and a.draw_footprint and fp.olon is not None:
        fx, fy = F.xy(fp.olon, fp.olat)
        ax.plot(fx, fy, color='0.15', lw=1.3, zorder=7.5)
    if cov_mask is not None and a.draw_coverage:
        F.outline(glat, glon, cov_mask, zorder=7.5, colors='0.15', linewidths=1.2)
    if clip_xy is not None and a.draw_clip_poly:
        cx, cy = F.xy(np.r_[clip_xy[:, 0], clip_xy[0, 0]], np.r_[clip_xy[:, 1], clip_xy[0, 1]])
        ax.plot(cx, cy, color='orangered', lw=1.4, zorder=7.6)

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
            for x0, x1, y0, y1 in a.station_exclude_box:   # e.g. drop Phase 4 eastern grid
                keep = ~((slon >= x0) & (slon <= x1) & (slat >= y0) & (slat <= y1))
                n_drop = int((~keep).sum())
                if n_drop:
                    print(f"  excluded {n_drop} stations in box "
                          f"[{x0:g},{x1:g},{y0:g},{y1:g}]")
                slon, slat = slon[keep], slat[keep]
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
