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


def compute_extent(a, lon, lat, cov_lon, cov_lat, fp):
    """lon/lat box for the frame, from --extent."""
    if a.extent == 'wa':
        return list(WA_EXTENT)
    if a.extent == 'nodes':
        return [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]
    if a.extent == 'coverage':
        if cov_lon is None:
            sys.exit("--extent coverage needs --coverage STATIONS")
        m = max(1.0, a.coverage_dist)
        return [cov_lon.min() - m, cov_lon.max() + m, cov_lat.min() - m, cov_lat.max() + m]
    if a.extent == 'footprint':
        if fp is None or fp.olon is None:
            sys.exit("--extent footprint needs --footprint with an outline")
        m = 0.5
        return [fp.olon.min() - m, fp.olon.max() + m, fp.olat.min() - m, fp.olat.max() + m]
    return [float(x) for x in a.extent.split(',')]


def make_norm(a, values):
    """(cmap, norm, vmin, vmax) for a field; discrete bands when --ncolors>0."""
    if a.clim:
        vmin, vmax = a.clim
    elif a.moho and not a.fvs:
        vmin, vmax = 30.0, 50.0
    else:
        vmin = float(np.floor(np.nanpercentile(values, 2) * 10) / 10)
        vmax = float(np.ceil(np.nanpercentile(values, 98) * 10) / 10)
    if a.ncolors > 0:
        from matplotlib.colors import BoundaryNorm
        cmap = matplotlib.colormaps[a.cmap].resampled(a.ncolors)
        cmap = cmap.with_extremes(under=cmap(0), over=cmap(cmap.N - 1))
        norm = BoundaryNorm(np.linspace(vmin, vmax, a.ncolors + 1), a.ncolors)
    else:
        cmap = matplotlib.colormaps[a.cmap]
        norm = Normalize(vmin, vmax)
    return cmap, norm, vmin, vmax


def build_mask(a, ext, lon, lat, moho, cov_lon, cov_lat, fp, quiet=False):
    """Grid + effective-area mask (depth-independent geometry). Returns
    glon, glat, GLO, GLA, Z, ok, cov_mask, fp_mask, clip_xy, clip_mask, add_xys."""
    def say(*m):
        if not quiet:
            print(*m)
    glon = np.arange(ext[0], ext[1] + 1e-6, a.dgrid)
    glat = np.arange(ext[2], ext[3] + 1e-6, a.dgrid)
    Z, ok = interp_field(lon, lat, moho, glon, glat, a.interp, a.max_dist)
    GLO, GLA = np.meshgrid(glon, glat)

    cov_mask = None
    if cov_lon is not None:
        from scipy.spatial import cKDTree
        Pp = t2.Albers()
        sx, sy = Pp(cov_lon, cov_lat)
        gx, gy = Pp(GLO.ravel(), GLA.ravel())
        dd, _ = cKDTree(np.column_stack([sx, sy])).query(np.column_stack([gx, gy]))
        cov_mask = (dd <= a.coverage_dist * 111.195).reshape(GLO.shape)
        ok = ok & cov_mask

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

    clip_xy = clip_mask = None
    if a.clip_poly:
        import matplotlib.path as mpath
        clip_xy = np.loadtxt(a.clip_poly, comments='#', usecols=(0, 1))
        clip_mask = mpath.Path(clip_xy).contains_points(
            np.column_stack([GLO.ravel(), GLA.ravel()])).reshape(GLO.shape)
        ok = ok & clip_mask
        say(f"  clip-poly: {len(clip_xy)} vertices <- {os.path.basename(a.clip_poly)}")

    if a.clip == 'wa' and fp is None and a.coast_buffer <= 0:
        wam = wa_mask(glon, glat)
        if wam is not None:
            ok = ok & wam
            say(f"  clipped to WA: {int(ok.sum())} of {ok.size} cells kept")
        else:
            say("  (WA clip requested but no WA polygon available - not clipped)")

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
        say(f"  add-poly: {len(axy)} vertices <- {os.path.basename(apath)}")

    if a.coast_buffer > 0:
        inwa = wa_mask(glon, glat)
        if inwa is not None:
            from scipy.ndimage import (binary_dilation, generate_binary_structure,
                                       iterate_structure)
            ncell = max(1, int(round(a.coast_buffer / (a.dgrid * 111.0))))
            grown = binary_dilation(ok, structure=iterate_structure(
                generate_binary_structure(2, 1), ncell))
            ok = ok | (grown & ~inwa & np.isfinite(Z))
            say(f"  coast-buffer: extended ~{a.coast_buffer:g} km offshore")
        else:
            say("  (coast-buffer: no WA polygon available)")

    if a.smooth_edge > 0:
        from scipy.ndimage import gaussian_filter, binary_fill_holes
        okf = gaussian_filter(ok.astype(float), a.smooth_edge / a.dgrid) > 0.5
        ok = binary_fill_holes(okf) & np.isfinite(Z)

    return glon, glat, GLO, GLA, Z, ok, cov_mask, fp_mask, clip_xy, clip_mask, add_xys


def make_frame(a, ext, figsize=None, rect=(0.04, 0.06, 0.80, 0.88)):
    """Figure + Albers MapFrame sized to the projected extent aspect."""
    P = t2.Albers()
    bx, by = P(np.r_[ext[0], ext[1], ext[1], ext[0]], np.r_[ext[2], ext[2], ext[3], ext[3]])
    asp = (by.max() - by.min()) / (bx.max() - bx.min())
    if figsize is None:
        w = 8.0
        figsize = (w + 1.4, w * asp + 1.0)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    F = t2.MapFrame(ax, ext, grat=a.grat)
    return fig, ax, F, asp


def _graticule_over(ax, F, ext, a, colour='0.55', lw=0.3, zorder=7.2):
    """Redraw the lon/lat graticule ON TOP of the grey veil so the grid reads
    across the masked area too (lat/lon ticks follow the whole frame)."""
    lo0, lo1, la0, la1 = ext
    step = a.grat or (2.0 if (lo1 - lo0) < 14 else 5.0)
    glon = np.arange(np.ceil(lo0 / step) * step, lo1 + 1e-9, step)
    glat = np.arange(np.ceil(la0 / step) * step, la1 + 1e-9, step)
    tl = np.linspace(la0 - 2, la1 + 2, 200)
    tL = np.linspace(lo0 - 2, lo1 + 2, 200)
    for g in glon:
        x, y = F.xy(np.full_like(tl, g), tl)
        ax.plot(x, y, color=colour, lw=lw, zorder=zorder, clip_on=True)
    for g in glat:
        x, y = F.xy(tL, np.full_like(tL, g))
        ax.plot(x, y, color=colour, lw=lw, zorder=zorder, clip_on=True)


def draw_field_veil(a, ax, F, glon, glat, ok, Z, cmap, norm):
    """Raster the field, then grey-veil the WHOLE frame outside `ok` (padded well
    past the lon/lat box so the rotated axes corners are covered too), then redraw
    the graticule on top. Returns the pcolormesh handle for the colour bar."""
    pm = F.raster(glat, glon, Z, ok, cmap, norm, zorder=2)
    if a.alpha < 1.0:
        pm.set_alpha(a.alpha)
    if a.mask_colour.lower() != 'none':
        PAD = 4.0
        gv_lon = np.arange(glon[0] - PAD, glon[-1] + PAD + 1e-6, a.dgrid)
        gv_lat = np.arange(glat[0] - PAD, glat[-1] + PAD + 1e-6, a.dgrid)
        inside_v = np.zeros((gv_lat.size, gv_lon.size), bool)
        j0 = int(round((glon[0] - gv_lon[0]) / a.dgrid))
        i0 = int(round((glat[0] - gv_lat[0]) / a.dgrid))
        inside_v[i0:i0 + ok.shape[0], j0:j0 + ok.shape[1]] = ok
        F.veil(gv_lat, gv_lon, inside_v, colour=a.mask_colour, alpha=1.0, zorder=7)
        _graticule_over(ax, F, F.extent, a)
    return pm


def draw_linework(a, ax, F, glon, glat, ok, cov_mask, fp, clip_xy, add_xys):
    """Tectonic domains + crustal boundaries + optional outlines, coastline overlay."""
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
    if not a.no_units:
        try:
            import plot_xsection as px
            from domains import Domains
            tect_npz = px._resolve_npz('TECTONICS_NPZ', 'wa_tectonics.npz', a.tect_npz)
            rings = [r for r in Domains(tect_npz).rings if len(r) >= 3]
            F.lines(rings, colors=a.tect_colour, linewidths=a.tect_lw, zorder=5.5)
        except Exception as e:
            print(f"  (no tectonic unit outlines: {e})")
    if not a.no_boundaries:
        try:
            import plot_xsection as px
            bnd_npz = px._resolve_npz('BOUNDARIES_NPZ', 'wa_crustal_boundaries.npz', a.bnd_npz)
            B = np.load(bnd_npz, allow_pickle=True)
            blines = [np.asarray(ln, float)[:, :2] for ln in B['lines'] if len(ln) >= 2]
            F.lines(blines, colors=a.bnd_colour, linewidths=a.bnd_lw, zorder=5)
        except Exception as e:
            print(f"  (no crustal boundaries: {e})")
    if a.coast_colour.lower() != 'none':
        for r in (getattr(t2, '_STATES', None) or {}).get('coast', []):
            x, y = F.xy(r[:, 0], r[:, 1])
            ax.plot(x, y, color=a.coast_colour, lw=a.coast_lw,
                    solid_capstyle='round', zorder=7.7)


def draw_stations(a, ax, F):
    if a.no_stations or not a.stations:
        return
    SL, SA = [], []
    for sp in a.stations:
        if not os.path.isfile(sp):
            print(f"  (stations: not found, skipped: {sp})")
            continue
        lo, la = load_any_stations(sp)
        SL.append(lo); SA.append(la)
        print(f"  stations: {len(lo)} <- {os.path.basename(sp)}")
    if not SL:
        return
    slon = np.concatenate(SL); slat = np.concatenate(SA)
    for x0, x1, y0, y1 in a.station_exclude_box:
        keep = ~((slon >= x0) & (slon <= x1) & (slat >= y0) & (slat <= y1))
        slon, slat = slon[keep], slat[keep]
    x, y = F.xy(slon, slat)
    ax.scatter(x, y, marker=a.station_marker, s=a.station_size,
               c=a.station_colour, edgecolors='w', linewidths=0.4, zorder=8)


def panels(a, ext, field_at, depths, field_label, cov_lon, cov_lat, fp):
    """2xN panel figure: one field map per depth, shared extent/mask/style."""
    n = len(depths)
    ncols = a.panel_cols or (2 if n <= 4 else int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))

    # geometry/mask from the first depth (node positions are depth-independent)
    lon0, lat0, v0 = field_at(depths[0])
    glon, glat, GLO, GLA, _Z0, ok, cov_mask, fp_mask, clip_xy, clip_mask, add_xys = \
        build_mask(a, ext, lon0, lat0, v0, cov_lon, cov_lat, fp)
    print(f"  panels: {n} depths, {nrows}x{ncols}, {int(ok.sum())} cells in effective area")

    P = t2.Albers()
    bx, by = P(np.r_[ext[0], ext[1], ext[1], ext[0]], np.r_[ext[2], ext[2], ext[3], ext[3]])
    asp = (by.max() - by.min()) / (bx.max() - bx.min())
    pw = 4.2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * pw + 0.6, nrows * pw * asp + 1.0))
    axs = np.atleast_1d(axs).ravel()

    for k, dep in enumerate(depths):
        ax = axs[k]
        F = t2.MapFrame(ax, ext, grat=a.grat)
        lon, lat, v = field_at(dep)
        Z, _okd = interp_field(lon, lat, v, glon, glat, a.interp, a.max_dist)
        cmap, norm, vmin, vmax = make_norm(a, v)
        pm = draw_field_veil(a, ax, F, glon, glat, ok, Z, cmap, norm)
        draw_linework(a, ax, F, glon, glat, ok, cov_mask, fp, clip_xy, add_xys)
        draw_stations(a, ax, F)
        ax.set_title(f"{dep:g} km", fontsize=11)
        cb = fig.colorbar(pm, ax=ax, extend='both', fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        print(f"  panel {dep:g} km: {v.min():.2f}-{v.max():.2f}, clim {vmin:.2f}-{vmax:.2f}")

    for ax in axs[n:]:
        ax.axis('off')
    fig.suptitle(a.title, fontsize=13, y=0.995)
    fig.text(0.995, 0.5, field_label, rotation=90, va='center', ha='right', fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.99, 0.98))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=220)
    fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--moho', default=None, help='Moho grid text table (lon lat moho_km)')
    ap.add_argument('--skip-header', type=int, default=11, help='header lines to skip (default 11)')
    ap.add_argument('--fvs', default=None,
                    help='Vs model npz (alternative to --moho): sample --field at --depth for an '
                         'absolute-Vs map in the same style')
    ap.add_argument('--depth', type=float, default=30.0, help='depth (km) to sample --fvs at')
    ap.add_argument('--depths', default=None,
                    help='comma-separated depths (km) for a multi-panel figure, e.g. '
                         '"5,15,30,45" (needs --fvs). Shares the extent/mask/style; each '
                         'panel gets its own colour bar unless --clim is given.')
    ap.add_argument('--panel-cols', type=int, default=0,
                    help='columns in the --depths panel grid (0 = auto: 2 cols for <=4 panels)')
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
    ap.add_argument('--tect-lw', type=float, default=1.1,
                    help='tectonic-domain edge linewidth (unfilled, thick by default)')
    ap.add_argument('--tect-colour', default='0.25', help='tectonic-domain edge colour')
    ap.add_argument('--bnd-lw', type=float, default=0.4,
                    help='crustal-boundary linewidth (thin by default)')
    ap.add_argument('--bnd-colour', default='k', help='crustal-boundary colour')
    ap.add_argument('--coast-colour', default='none',
                    help="overlay the WA coastline in this colour (e.g. 'steelblue'); "
                         "'none' keeps only the base thin black coast")
    ap.add_argument('--coast-lw', type=float, default=1.8,
                    help='overlaid coastline linewidth')
    ap.add_argument('--grat', type=float, default=None, help='graticule step (deg)')
    ap.add_argument('--title', default='WA Array refined CCP Moho')
    ap.add_argument('--out', default='moho_map.png')
    a = ap.parse_args()
    depths = [float(x) for x in a.depths.split(',')] if a.depths else None

    field_at = None
    if a.fvs:                                        # absolute-Vs (or other field) at a depth
        from domains import model_fields
        d = dict(np.load(a.fvs, allow_pickle=True))
        Fm = model_fields(d)
        if a.field not in Fm:
            sys.exit(f"{os.path.basename(a.fvs)} has no '{a.field}' (has: {', '.join(Fm)})")
        lon_all = np.asarray(d['Lon'], float); lat_all = np.asarray(d['Lat'], float)
        zc = np.asarray(d['z'], float)
        fld = np.asarray(Fm[a.field], float)

        def field_at(depth):
            v = np.array([np.interp(depth, zc, row) for row in fld])
            g = np.isfinite(lon_all) & np.isfinite(lat_all) & np.isfinite(v)
            return lon_all[g], lat_all[g], v[g]

        lon, lat, moho = field_at(a.depth)
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

    ext = compute_extent(a, lon, lat, cov_lon, cov_lat, fp)

    # multi-panel figure (one field map per depth), shared extent/mask/style
    if depths:
        if field_at is None:
            sys.exit("--depths needs --fvs (a depth-varying model)")
        panels(a, ext, field_at, depths, field_label, cov_lon, cov_lat, fp)
        return

    # single map
    glon, glat, GLO, GLA, Z, ok, cov_mask, fp_mask, clip_xy, clip_mask, add_xys = \
        build_mask(a, ext, lon, lat, moho, cov_lon, cov_lat, fp)
    cmap, norm, vmin, vmax = make_norm(a, moho)
    fig, ax, F, asp = make_frame(a, ext)

    # bake the final effective-area boundary to a reusable mask file
    if a.save_outline:
        from scipy.ndimage import binary_closing, binary_fill_holes
        okc = binary_fill_holes(binary_closing(ok, iterations=3, border_value=0))
        _f = plt.figure(); _axc = _f.add_subplot()
        _cs = _axc.contour(glon, glat, okc.astype(float), levels=[0.5])
        _segs = _cs.allsegs[0] if _cs.allsegs else []
        plt.close(_f)
        if _segs:
            def _area(s):
                x, y = s[:, 0], s[:, 1]
                return abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
            seg = max(_segs, key=_area)
            np.savetxt(a.save_outline, seg, fmt='%.4f', header='lon lat  effective-area outline')
            print(f"  saved outline: {len(seg)} vertices ({len(_segs)} segs) -> {a.save_outline}")

    pm = draw_field_veil(a, ax, F, glon, glat, ok, Z, cmap, norm)
    draw_linework(a, ax, F, glon, glat, ok, cov_mask, fp, clip_xy, add_xys)
    draw_stations(a, ax, F)

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
