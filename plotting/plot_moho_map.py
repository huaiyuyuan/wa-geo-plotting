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


def wa_buffer_mask(glon, glat, km, dgrid):
    """inside-WA boolean grid dilated by ~km (so it extends ~km seaward of the
    coast). Uses the WA polygon via wa_mask + a morphological dilation - no
    shapely needed. None if no WA polygon is available."""
    base = wa_mask(glon, glat)
    if base is None:
        return None
    from scipy.ndimage import (binary_dilation, generate_binary_structure,
                               iterate_structure)
    ncell = max(1, int(round(km / (dgrid * 111.0))))
    st = iterate_structure(generate_binary_structure(2, 1), ncell)
    return binary_dilation(base, structure=st)


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
    ap.add_argument('--moho', default=None, help='Moho grid text table (lon lat moho_km)')
    ap.add_argument('--skip-header', type=int, default=11, help='header lines to skip (default 11)')
    ap.add_argument('--fvs', default=None,
                    help='Vs model npz (alternative to --moho): sample --field at --depth for an '
                         'absolute-Vs map in the same style')
    ap.add_argument('--depth', type=float, default=30.0, help='depth (km) to sample --fvs at')
    ap.add_argument('--field', default='vsv', help='--fvs field: vsv (default), viso, xi, vpvs')
    ap.add_argument('--cbar-label', default=None, help='colorbar label (auto from source if unset)')
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
    ap.add_argument('--add-poly', action='append', default=[],
                    help='lon/lat polygon file whose interior is ADDED (union) to the effective '
                         'area, e.g. a patch that fills the eastern or northern gap out to a hand '
                         'line while keeping the working mask elsewhere. Repeatable.')
    ap.add_argument('--add-poly-minlon', type=float, default=None,
                    help='only add the part of every --add-poly east of this longitude')
    ap.add_argument('--draw-add-poly', action='store_true', help='draw the --add-poly outlines')
    ap.add_argument('--save-outline', default=None,
                    help='write the final effective-area boundary (lon lat) to this file, to reuse '
                         'as a single mask later (e.g. via --clip-poly or --footprint-outline)')
    ap.add_argument('--clim', nargs=2, type=float, default=None, metavar=('LO', 'HI'),
                    help='colour limits; default 30 50 for --moho, 2-98th percentile for --fvs; '
                         'values beyond saturate to the end colours')
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
    ap.add_argument('--coast-buffer', type=float, default=0.0,
                    help='keep the mask within ~this many km of the WA coast (dilates the WA '
                         'landmass), so the W/S edges sit ~KM offshore instead of being cut; 0 = off')
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

    if a.fvs:                                        # absolute-Vs (or other field) at a depth
        from domains import model_fields
        d = dict(np.load(a.fvs, allow_pickle=True))
        F = model_fields(d)
        if a.field not in F:
            sys.exit(f"{os.path.basename(a.fvs)} has no '{a.field}' (has: {', '.join(F)})")
        lon = np.asarray(d['Lon'], float); lat = np.asarray(d['Lat'], float)
        zc = np.asarray(d['z'], float)
        moho = np.array([np.interp(a.depth, zc, row) for row in np.asarray(F[a.field], float)])
        good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(moho)
        lon, lat, moho = lon[good], lat[good], moho[good]
        field_label = a.cbar_label or ('Vsv (km/s)' if a.field in ('vsv', 'viso') else a.field)
        print(f"Field: {a.field} at {a.depth:g} km, {len(lon)} nodes <- "
              f"{os.path.basename(a.fvs)} ({moho.min():.2f}-{moho.max():.2f})")
    elif a.moho:
        lon, lat, moho = read_moho(a.moho, a.skip_header)
        field_label = a.cbar_label or 'Moho depth (km)'
        print(f"Moho: {len(lon)} nodes <- {os.path.basename(a.moho)} "
              f"({moho.min():.1f}-{moho.max():.1f} km)")
    else:
        sys.exit("give --moho (Moho grid) or --fvs (Vs model + --depth)")

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

    if a.clim:
        vmin, vmax = a.clim
    elif a.moho and not a.fvs:
        vmin, vmax = 30.0, 50.0
    else:
        vmin = float(np.floor(np.nanpercentile(moho, 2) * 10) / 10)
        vmax = float(np.ceil(np.nanpercentile(moho, 98) * 10) / 10)
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
    if a.clip == 'wa' and fp is None and a.coast_buffer <= 0:   # coast-buffer replaces it
        wam = wa_mask(glon, glat)
        if wam is not None:
            ok = ok & wam
            print(f"  clipped to WA: {int(ok.sum())} of {ok.size} cells kept")
        else:
            print("  (WA clip requested but no WA polygon available - not clipped)")

    # add (union) polygon interiors, e.g. fill the eastern / northern gaps out to a
    # hand line while keeping the working mask on the other sides
    add_xys = []
    for apath in a.add_poly:
        import matplotlib.path as mpath
        axy = np.loadtxt(apath, comments='#', usecols=(0, 1))
        amask = mpath.Path(axy).contains_points(
            np.column_stack([GLO.ravel(), GLA.ravel()])).reshape(GLO.shape)
        if a.add_poly_minlon is not None:
            amask = amask & (GLO >= a.add_poly_minlon)
        ok = ok | (amask & np.isfinite(Z))
        add_xys.append(axy)
        print(f"  add-poly: {len(axy)} vertices <- {os.path.basename(apath)}")

    # cap the mask ~coast-buffer km seaward of the WA coast (keeps the W/S edges
    # a clean fixed distance offshore instead of cut inside the coast)
    if a.coast_buffer > 0:
        wab = wa_buffer_mask(glon, glat, a.coast_buffer, a.dgrid)
        if wab is not None:
            ok = ok & wab
            print(f"  coast-buffer: within ~{a.coast_buffer:g} km of the WA coast")
        else:
            print("  (coast-buffer: no WA polygon available)")

    # round off the scalloped coverage edge and fill small interior holes
    smoothed = False
    if a.smooth_edge > 0:
        from scipy.ndimage import gaussian_filter, binary_fill_holes
        okf = gaussian_filter(ok.astype(float), a.smooth_edge / a.dgrid) > 0.5
        ok = binary_fill_holes(okf) & np.isfinite(Z)
        smoothed = True

    # bake the final effective-area boundary to a reusable mask file
    if a.save_outline:
        from scipy.ndimage import binary_closing, binary_fill_holes
        # bridge thin necks + fill holes so the region is one solid blob and its
        # outer boundary traces as a single closed loop (no NE lobe pinched off)
        okc = binary_fill_holes(binary_closing(ok, iterations=3, border_value=0))
        _f = plt.figure(); _axc = _f.add_subplot()
        _cs = _axc.contour(glon, glat, okc.astype(float), levels=[0.5])
        _segs = _cs.allsegs[0] if _cs.allsegs else []
        plt.close(_f)
        if _segs:
            def _area(s):
                x, y = s[:, 0], s[:, 1]
                return abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
            seg = max(_segs, key=_area)         # outer boundary = largest enclosed area
            np.savetxt(a.save_outline, seg, fmt='%.4f', header='lon lat  effective-area outline')
            print(f"  saved outline: {len(seg)} vertices ({len(_segs)} segs) -> {a.save_outline}")

    pm = F.raster(glat, glon, Z, ok, cmap, norm, zorder=2)
    if a.alpha < 1.0:
        pm.set_alpha(a.alpha)

    # cover the masked area with solid grey (outside WA, and outside coverage when
    # given), above the field and the line work there, below the stations
    # grey fills everywhere not painted, matching the final effective area
    veil_inside = ok if (a.clip == 'wa' or cov_mask is not None or fp is not None
                         or clip_mask is not None or add_xys or a.coast_buffer > 0) else None
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
    if add_xys and a.draw_add_poly:
        for axy in add_xys:
            px, py = F.xy(np.r_[axy[:, 0], axy[0, 0]], np.r_[axy[:, 1], axy[0, 1]])
            ax.plot(px, py, color='orangered', lw=1.4, zorder=7.6)

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
    cb.set_label(field_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(a.title, fontsize=12)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=220)
    fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")


if __name__ == '__main__':
    main()
