#!/usr/bin/env python3
"""
plot_wa_map.py - standalone WA map plotter (profile locator / station map).

The map figure that was embedded in plot_xsection.py (_plot_index_map) lives here now,
as plot_map(). It draws the GSWA 2022 tectonic basemap, major crustal boundaries, unit
labels/legend, profile lines (straight or along picked polylines with distance ticks),
station layers (WA Array + arbitrary CSV overlays), and the array footprint.

It can be driven two ways:
  - called by plot_xsection.py --index-map (passing the Vs model d for the extent), or
  - standalone (its own CLI below): a WA map with stations / domains / boundaries /
    profile lines and an explicit --extent, with NO Vs model required.

Shared geometry/geology helpers (_great_circle_path, _polyline_path, _load_geology,
_resolve_npz, _load_stations, _index_footprint) stay in plot_xsection and are imported
lazily to avoid a circular import (plot_xsection imports this module only inside its
--index-map branch).
"""
import os, sys
import numpy as np
import matplotlib


def _load_station_csv(path):
    """Station lon/lat (and code, if present) from a CSV with named Latitude/Longitude
    columns (any order; e.g. the GSWA 'New.WA.Seismic' exports). Returns (lon, lat, codes):
    codes is a list of per-station code strings from the first of
    Code/Station Code/Station/Sta/Site_ID/Site/Name found, or None if no such column."""
    import csv
    lon, lat, codes = [], [], []
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = next(r, [])
        hl = [h.strip().lower() for h in header]
        def find(*names):
            for n in names:
                if n in hl:
                    return hl.index(n)
            return None
        ila = find('latitude', 'lat')
        ilo = find('longitude', 'lon', 'long')
        if ila is None or ilo is None:
            raise SystemExit(f"{path}: need Latitude/Longitude columns; header was {header}")
        icode = find('code', 'station code', 'station', 'sta', 'site_id', 'site', 'name')
        for row in r:
            if len(row) <= max(ila, ilo):
                continue
            try:
                lat.append(float(row[ila])); lon.append(float(row[ilo]))
            except ValueError:
                continue
            codes.append(row[icode].strip() if icode is not None and len(row) > icode else '')
    return np.array(lon), np.array(lat), (codes if icode is not None else None)



# MATLAB 'parula' (anchors) so the Moho depth map matches flipud(parula) exactly.
_PARULA = [(0.2422, 0.1504, 0.6603), (0.2810, 0.3228, 0.9579), (0.1786, 0.5289, 0.9682),
           (0.0689, 0.6948, 0.8394), (0.2161, 0.7843, 0.5923), (0.6720, 0.7793, 0.2227),
           (0.9970, 0.9043, 0.1407), (0.9769, 0.9839, 0.0805)]


def _resolve_cmap(spec):
    """matplotlib name, or 'parula'/'parula_r', or 'rwb' (red-white-blue, like the xi map),
    or a path to a MATLAB .mat colormap (first Nx3 array, e.g. cmapx.mat)."""
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    if not spec:
        return 'Spectral'
    if spec.endswith('.mat') and os.path.isfile(spec):
        from scipy.io import loadmat
        for k, v in loadmat(spec).items():
            v = np.asarray(v)
            if not k.startswith('__') and v.ndim == 2 and v.shape[1] == 3:
                return ListedColormap(v.astype(float))
        return 'Spectral'
    rev = spec.endswith('_r'); base = spec[:-2] if rev else spec
    if base == 'parula':
        cm = LinearSegmentedColormap.from_list('parula', _PARULA)
    elif base == 'rwb':
        cm = LinearSegmentedColormap.from_list('rwb', [(0.78, 0, 0), (1, 1, 1), (0, 0, 0.78)])
    elif base == 'rgb':                      # red (low) -> green (mid) -> blue (high), smooth
        cm = LinearSegmentedColormap.from_list('rgb', [
            (0.63, 0.09, 0.16), (0.88, 0.40, 0.22), (0.62, 0.70, 0.35),
            (0.38, 0.70, 0.40), (0.26, 0.62, 0.62), (0.26, 0.42, 0.72), (0.30, 0.22, 0.60)])
    else:
        return spec                      # a plain matplotlib name (handles its own _r)
    return cm.reversed() if rev else cm


def _read_moho_mat(path, field='z_moho', apply_mask=True):
    """Moho pick struct moho_ccp_refined.final.mat -> 2-D grids (LON, LAT, V). With
    apply_mask (default) V is NaN where stn_mask is false (the MATLAB masking); with
    apply_mask=False the full grid is returned (only the data's own NaNs are holes)."""
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=True, struct_as_record=False)['moho_refined']
    LON = np.asarray(m.lon2, float); LAT = np.asarray(m.lat2, float)
    V = np.asarray(getattr(m, field), float)
    if apply_mask and hasattr(m, 'stn_mask'):
        V = np.where(np.asarray(m.stn_mask).astype(bool), V, np.nan)
    return LON, LAT, V


# HK summary table columns (after dropping the '|' separators):
# STA lat lon elev H_old k_old H_new dH k_new dk moho_ref k_ref Vp Vs nRF st band project
_HK_COLS = {'h_old': 4, 'k_old': 5, 'h_new': 6, 'dh': 7, 'k_new': 8, 'dk': 9,
            'moho_ref': 10, 'k_ref': 11, 'vp': 12, 'vs': 13, 'nrf': 14}


def _read_hk(path, field='k_new'):
    """lon, lat, value from an HK summary table (per-station H-K). '|' separators and
    '%'/'#' comments handled; field = k_new (Vp/Vs), h_new, vp, vs, ..."""
    jc = _HK_COLS[field.lower()]
    lon, lat, val = [], [], []
    for line in open(path):
        s = line.strip()
        if not s or s[0] in '%#':
            continue
        t = [x for x in s.split() if x != '|']
        if len(t) <= jc:
            continue
        try:
            la, lo, v = float(t[1]), float(t[2]), float(t[jc])
        except ValueError:
            continue
        if np.isfinite(la) and np.isfinite(lo) and np.isfinite(v):
            lat.append(la); lon.append(lo); val.append(v)
    return np.array(lon), np.array(lat), np.array(val)


def _read_ccp_grid(path):
    """LON2, LAT2, stn_mask from moho_ccp_refined.final.mat (for CCP-footprint masking)."""
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=True, struct_as_record=False)['moho_refined']
    LON = np.asarray(m.lon2, float); LAT = np.asarray(m.lat2, float)
    mask = np.asarray(m.stn_mask).astype(bool) if hasattr(m, 'stn_mask') else np.ones(LON.shape, bool)
    return LON, LAT, mask


def _amp_at_moho(pmat, LON2, LAT2, ZC, win=3.0, amp_var='cmp'):
    """Peak CCP amplitude within +/-win km of the final Moho depth ZC, on the CCP grid
    (same nodes as moho_refined). Mirrors samp_amp() in plot_moho_final_amp.m."""
    from plot_ccp_profiles import load_ccp
    ccp = load_ccp(pmat, amp_var=amp_var)
    cube = np.asarray(ccp['cmp'], float); zg = np.asarray(ccp['zg'], float)
    cube[~np.isfinite(cube)] = 0.0
    if cube.shape[:2] != LON2.shape:                 # align cube [nx,ny] with the moho grid
        if cube.shape[:2] == LON2.shape[::-1]:
            cube = np.transpose(cube, (1, 0, 2))
        else:
            raise ValueError(f"cube {cube.shape[:2]} != moho grid {LON2.shape}")
    sel = np.abs(zg[None, None, :] - ZC[:, :, None]) <= win     # [nx,ny,nz]
    masked = np.where(sel, cube, -np.inf)
    A = masked.max(axis=2)
    A[(~np.isfinite(ZC)) | (~sel.any(axis=2))] = np.nan
    amax = np.nanmax(A) if np.isfinite(A).any() else np.nan
    print(f"  amp cube {cube.shape}: peak within +/-{win:g} km of Moho, "
          f"{int(np.isfinite(A).sum())} nodes, max {amax:.4g}")
    return A


def _read_cut_poly(path):
    """Ordered (lon,lat) vertices of a NE cut line, from a 2-col text file
    ('#'/'%' comments ok). Returns an (N,2) array NW->SE."""
    pts = []
    for line in open(path):
        s = line.strip()
        if not s or s[0] in '#%':
            continue
        t = s.split()
        if len(t) >= 2:
            try:
                pts.append((float(t[0]), float(t[1])))
            except ValueError:
                continue
    return np.asarray(pts, float)


def _yilgarn_units_draw(ax, units_npz, transform=None, zorder=1.2,
                        fill_types=('greenstone',), colour_override=None, alpha=1.0):
    """Fill Yilgarn rock-type polygons (yilgarn_units.npz). Self-contained here so the
    overlay needs no wa_basemap change (only _compound, which is stable there)."""
    from matplotlib.collections import PathCollection
    from collections import defaultdict
    try:
        from wa_basemap import _compound
    except Exception:
        from matplotlib.path import Path
        def _compound(rings):
            return Path.make_compound_path(*[Path(np.asarray(r, float)) for r in rings])
    d = np.load(units_npz, allow_pickle=True)
    rings = d['rings']; rec = d['rec_idx']; rock = d['rocktype'].astype(str)
    rc = {str(k): str(v) for k, v in d['rock_colours']} if 'rock_colours' in d.files else {}
    want = set(fill_types)
    proj = getattr(ax, 'projection', None)
    pre = proj is not None and hasattr(transform, 'transform_points')
    P = (lambda r: proj.transform_points(transform, r[:, 0], r[:, 1])[:, :2]) if pre else (lambda r: np.asarray(r))
    groups = defaultdict(list); gtype = {}
    for i in range(len(rings)):
        if rock[i] in want:
            groups[int(rec[i])].append(np.asarray(rings[i], float)); gtype[int(rec[i])] = rock[i]
    paths = [_compound([P(r) for r in rr]) for rr in groups.values()]
    faces = [colour_override or rc.get(gtype[k], '#2f8f5b') for k in groups]
    kw = {'transform': ax.transData if pre else transform} if transform is not None else {}
    ax.add_collection(PathCollection(paths, facecolors=faces, edgecolors='none',
                                     linewidths=0.0, alpha=alpha, zorder=zorder, **kw))
    print(f"  yilgarn units: filled {len(paths)} polygons ({', '.join(sorted(want))})")


def _yilgarn_bounds_draw(ax, bnd_npz, transform=None, zorder=4.6, kinds=('craton', 'domain'),
                         craton_color='red', domain_color='0.4', lw_craton=1.4, lw_domain=0.6, alpha=0.9):
    """Yilgarn craton outline (red) / terrane-domain boundaries (grey). Self-contained."""
    from matplotlib.collections import LineCollection
    d = np.load(bnd_npz, allow_pickle=True)
    lines = d['lines']; kind = d['kind'].astype(str)
    proj = getattr(ax, 'projection', None)
    pre = proj is not None and hasattr(transform, 'transform_points')
    P = (lambda r: proj.transform_points(transform, r[:, 0], r[:, 1])[:, :2]) if pre else (lambda r: np.asarray(r))
    kw = {'transform': ax.transData if pre else transform} if transform is not None else {}
    drawn = {}
    for ksel, col, lw in (('domain', domain_color, lw_domain), ('craton', craton_color, lw_craton)):
        if ksel not in kinds:
            continue
        segs = [P(np.asarray(lines[i], float)) for i in range(len(lines)) if kind[i] == ksel]
        if segs:
            ax.add_collection(LineCollection(segs, colors=col, linewidths=lw, alpha=alpha, zorder=zorder, **kw))
            drawn[ksel] = len(segs)
    print("  yilgarn boundaries: " + ", ".join(f"{v} {k}" for k, v in drawn.items()))


def _keep_poly(cut_pts):
    """Build a closed matplotlib Path enclosing the KEEP region (SW of the cut line).
    The open NE cut polyline (NW->SE) is closed with a big box far to the S and W so
    contains_points() is True for everything on the south-west side of the line."""
    from matplotlib.path import Path
    cp = np.asarray(cut_pts, float)
    lo_w, la_s = cp[:, 0].min() - 20.0, cp[:, 1].min() - 20.0   # far SW corner
    verts = np.vstack([cp,
                       [cp[-1, 0], la_s],     # straight down from the SE end
                       [lo_w,       la_s],     # across to the far west
                       [lo_w,       cp[0, 1]], # up the west side to the start latitude
                       cp[0]])                 # close back to the NW start
    return Path(verts)


def _krige_points(lon, lat, val, qlon, qlat, method='kriging'):
    """Interpolate scattered (lon,lat,val) to query points. Ordinary kriging (pykrige)
    if available, else RBF thin-plate spline."""
    if method == 'kriging':
        try:
            from pykrige.ok import OrdinaryKriging
            ok = OrdinaryKriging(lon, lat, val, variogram_model='spherical',
                                 verbose=False, enable_plotting=False)
            z, _ = ok.execute('points', np.asarray(qlon), np.asarray(qlat))
            return np.asarray(z, float)
        except Exception as e:
            print(f"  (kriging unavailable: {e}; using bounded linear interpolation)")
    from scipy.interpolate import griddata
    pts = np.column_stack([lon, lat]); q = np.column_stack([np.asarray(qlon), np.asarray(qlat)])
    z = griddata(pts, val, q, method='linear')          # bounded, no overshoot (NaN outside hull)
    nan = ~np.isfinite(z)
    if nan.any():
        z[nan] = griddata(pts, val, q[nan], method='nearest')   # fill the hull exterior
    return z


def _nan_smooth(Z, sig):
    """NaN-aware Gaussian smooth (smooth over finite values, keep NaNs), like the MATLAB."""
    if sig <= 0:
        return Z
    from scipy.ndimage import gaussian_filter
    m = np.isfinite(Z); z0 = np.where(m, Z, 0.0)
    out = gaussian_filter(z0, sig) / np.maximum(gaussian_filter(m.astype(float), sig), 1e-6)
    out[~m] = np.nan
    return out


def _depth_desc(a, b):
    """(tag, human) for a depth window; inf = that node's Moho (matches MATLAB depth_tag)."""
    if np.isfinite(a) and np.isfinite(b) and a == b: return (f'z{a:g}',       f'{a:g} km slice')
    if np.isinf(b) and np.isfinite(a):               return (f'{a:g}-moho',   f'{a:g} km to Moho')
    if np.isinf(a) and np.isfinite(b):               return (f'moho-{b:g}',   f'Moho to {b:g} km')
    if np.isfinite(a) and np.isfinite(b):            return (f'{a:g}-{b:g}',   f'{a:g}-{b:g} km avg')
    return ('moho', 'Moho')


def _avg_model_window(npz_path, field, LON2, LAT2, ZC, window=(0.0, np.inf), smooth=0.8):
    """Depth-window average of a TransD field (Vsv/Xi) on the native CCP grid, matching
    plot_moho_final_*avg window_reduce: lateral linear interpolation to each node, then
    mean over z in [a,b] per node. An infinite bound = that node's Moho, so (20,inf)=
    20->Moho, (inf,60)=Moho->60 (sub-Moho), (a,a)=single nearest-depth slice above Moho."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay
    a, b = float(window[0]), float(window[1])
    d = np.load(npz_path, allow_pickle=True)
    mlon = np.asarray(d['Lon'], float).ravel(); mlat = np.asarray(d['Lat'], float).ravel()
    zz = np.asarray(d['z'], float).ravel()
    X = np.asarray(d[field], float)
    if X.shape[0] != mlon.size:
        X = X.T                                          # -> [nnode, nz]
    good = np.isfinite(mlon) & np.isfinite(mlat)
    mlon, mlat, X = mlon[good], mlat[good], X[good]
    qlon = LON2.ravel(); qlat = LAT2.ravel(); zc = ZC.ravel()
    zneed = np.nanmax([np.nanmax(zc)] + [x for x in (a, b) if np.isfinite(x)])
    kmax = max(int(np.searchsorted(zz, zneed, side='right')), 1)
    tri = Delaunay(np.column_stack([mlon, mlat]))        # built once, reused per depth
    qpts = np.column_stack([qlon, qlat]); zk = zz[:kmax]
    prof = np.full((qlon.size, kmax), np.nan)
    for k in range(kmax):
        prof[:, k] = LinearNDInterpolator(tri, X[:, k])(qpts)
    if np.isfinite(a) and np.isfinite(b) and a == b:     # single fixed-depth slice
        ks = int(np.argmin(np.abs(zk - a)))
        avg = prof[:, ks].copy()
        avg[(a > zc) | ~np.isfinite(avg) | ~np.isfinite(zc)] = np.nan
    else:                                                # window mean (inf bound -> Moho)
        lo = zc.copy() if np.isinf(a) else np.full(qlon.size, a)
        hi = zc.copy() if np.isinf(b) else np.full(qlon.size, b)
        los = np.minimum(lo, hi); his = np.maximum(lo, hi)
        M = (zk[None, :] >= los[:, None]) & (zk[None, :] <= his[:, None])
        fin = np.isfinite(prof); cnt = (M & fin).sum(1)
        avg = np.where(cnt > 0, np.sum(np.where(M & fin, prof, 0.0), 1) / np.maximum(cnt, 1), np.nan)
        avg[~np.isfinite(zc)] = np.nan
    V = _nan_smooth(avg.reshape(LON2.shape), smooth)
    _, desc = _depth_desc(a, b)
    print(f"  {field} {desc}: {int(np.isfinite(V).sum())} nodes, mean {np.nanmean(V):.4f}, "
          f"range {np.nanmin(V):.4f}-{np.nanmax(V):.4f}")
    return V


def _nice_ticks(vmin, vmax, n=8):
    """Round 'nice' tick values spanning [vmin,vmax] (steps of 1/2/2.5/5 x10^k),
    plus the decimal count to label them cleanly."""
    import math
    span = float(vmax) - float(vmin)
    if not np.isfinite(span) or span <= 0:
        return None, 2
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 5, 10) if raw <= m * mag * (1 + 1e-9)), 10 * mag)
    t0 = math.ceil(vmin / step - 1e-9) * step
    ticks = []
    t = t0
    while t <= vmax + step * 1e-6:
        ticks.append(round(t, 10)); t += step
    dec = max(0, -int(math.floor(math.log10(step))))
    return ticks, dec


def _draw_field(ax, field, extent, tr):
    """Fill a scalar field over the map. A 'grid2d' field (LON,LAT,V 2-D) is drawn
    on its native grid like the MATLAB pcolor; otherwise node lon/lat/val are RBF-
    gridded and masked to WA land + within max_dist of a node."""
    cmap = _resolve_cmap(field.get('cmap', 'Spectral'))
    nc = int(field.get('ncolors', 0) or 0)
    if nc > 0:                                        # discrete colour levels (like MATLAB cmap(N))
        import matplotlib
        cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        cmap = cm.resampled(nc)
    kw = {'transform': tr} if tr is not None else {}
    style = field.get('style', 'pcolor')
    if 'grid2d' in field:                         # native CCP grid (matches MATLAB)
        LON, LAT, V = field['grid2d']
        Zm = np.ma.array(V, mask=~np.isfinite(V))
        good = np.isfinite(V)
        vmin, vmax = field['clim'] if field.get('clim') else \
            (np.nanpercentile(V[good], [2, 98]) if good.any() else (None, None))
        if style == 'contourf' and vmin is not None:   # smooth filled contours (edge='none'), MATLAB-like
            lv = np.linspace(vmin, vmax, (nc or 16) + 1)
            im = ax.contourf(LON, LAT, Zm, levels=lv, cmap=cmap, extend='both',
                             antialiased=True, alpha=field.get('alpha', 0.9), zorder=1.5, **kw)
            try:                                       # hide thin lines between bands
                im.set_edgecolor('face')
            except Exception:
                for c in getattr(im, 'collections', []):
                    c.set_edgecolor('face')
        else:
            im = ax.pcolormesh(LON, LAT, Zm, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading=('gouraud' if style == 'smooth' else 'nearest'),
                               zorder=1.5, alpha=field.get('alpha', 0.9), **kw)
        n = int(good.sum()); dmin, dmax = np.nanmin(V), np.nanmax(V)
    else:                                          # scattered nodes -> RBF grid
        from plot_moho_map import interp_field, wa_mask
        lon = np.asarray(field['lon'], float); lat = np.asarray(field['lat'], float)
        val = np.asarray(field['val'], float)
        dgrid = field.get('dgrid', 0.05); max_dist = field.get('max_dist', 0.6)
        lo0, lo1, la0, la1 = extent
        glon = np.arange(lo0, lo1 + dgrid, dgrid); glat = np.arange(la0, la1 + dgrid, dgrid)
        Z, ok = interp_field(lon, lat, val, glon, glat, field.get('interp', 'rbf'), max_dist)
        try:
            m = wa_mask(glon, glat)
            if m is not None:
                ok = ok & m
        except Exception as e:
            print(f"  (WA land mask unavailable: {e})")
        Zm = np.ma.array(Z, mask=~ok)
        vmin, vmax = field['clim'] if field.get('clim') else \
            (np.nanpercentile(Z[ok], [2, 98]) if ok.any() else (None, None))
        im = ax.pcolormesh(glon, glat, Zm, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto', zorder=1.5, alpha=field.get('alpha', 0.9), **kw)
        n = len(lon); dmin, dmax = np.nanmin(val), np.nanmax(val)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.55, pad=0.02, extend='both')
    cb.set_label(field.get('label', ''), fontsize=10, fontweight='bold')
    cb.ax.tick_params(labelsize=8)
    if vmin is not None:                           # snap colorbar ticks to round numbers
        ticks, dec = _nice_ticks(vmin, vmax)
        if ticks:
            cb.set_ticks(ticks)
            cb.set_ticklabels([f'{t:.{dec}f}' for t in ticks])
    print(f"  field: {n} nodes, data {dmin:.2f}-{dmax:.2f}"
          + (f", shown {vmin:.2f}-{vmax:.2f}" if vmin is not None else ""))
    return im


def plot_map(sections, out_dir, d=None, extent=None, label='sections', moho=None,
             litho_npz=None, tick_km=100, stations=None, style='map2022',
             footprint=None, footprint_veil=0.5, tect_npz=None, bnd_npz=None,
             projection='albers', legend='upper left', labels='major',
             label_scale=1.4, polylines=None, overlays=None, geo=None,
             stations_label='Available', stations_size=22,
             box=None, box_color=(0, 0, 0), box_lw=2.2,
             station_marker='^', station_color='k', station_open=False,
             out_name='xsection_index_map', legend_ages=False, bnd_loc='lower left',
             field=None, yilgarn=None, cut_line=None, deposit_args=None):
    """Profile lines / station map on the WA tectonic basemap, distance ticks + labels
    every tick_km. style: map2022 = 10M units in the GSWA 2022 simplified tectonic map
    colours + major crustal boundaries; gswa = 500k units in TECTCOLOUR; craton =
    domains + Yilgarn/Pilbara granite-greenstone from litho_npz; plain = domains.

    extent: (lon0, lon1, lat0, lat1) sets the frame. If None it is taken from the Vs
    model d (if given) else from the sections/stations/overlays. d is used only for the
    extent and, in the no-tectonic fallback, to show the model nodes; it is optional, so
    a map can be drawn with no Vs model at all. geo: a loaded geology dict (as from
    plot_xsection._load_geology) for the non-map2022 boundary overlay; default None."""
    import matplotlib.pyplot as plt
    import plot_xsection as px            # lazy: reuse shared geometry/geology helpers
    # keep text editable in the .pdf (Type-42 fonts, not outlines) for Illustrator/Inkscape
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if geo is None:
        geo = getattr(px, '_GEO', None)
    # points used for the extent (and the fallback node scatter when d is given)
    if d is not None:
        Lon = np.asarray(d['Lon'], float); Lat = np.asarray(d['Lat'], float)
    else:
        xs, ys = [], []
        for (la1, lo1, la2, lo2, _l) in sections:
            xs += [lo1, lo2]; ys += [la1, la2]
        if stations is not None:
            xs += list(stations[0]); ys += list(stations[1])
        for _ov in (overlays or []):
            xs += list(_ov[0]); ys += list(_ov[1])
        if not xs:
            if extent is None:
                sys.exit("plot_map: nothing to set the extent from (give d, extent, "
                         "sections or stations)")
            xs = [extent[0], extent[1]]; ys = [extent[2], extent[3]]
        Lon = np.asarray(xs, float); Lat = np.asarray(ys, float)
    if extent is not None:
        ext_lo0, ext_lo1, ext_la0, ext_la1 = extent
    else:
        ext_lo0, ext_lo1 = Lon.min() - 0.5, Lon.max() + 0.5
        ext_la0, ext_la1 = Lat.min() - 0.5, Lat.max() + 0.5
    try:
        from plot_depth_slice import _add_states_ocean, _HAS_CARTOPY
        import cartopy.crs as ccrs
        if not _HAS_CARTOPY:
            raise ImportError('cartopy')
        fig = plt.figure(figsize=(13, 13))
        # albers = the projection of the GSWA 1:10M simplified tectonic map (2022):
        # Albers equal-area, central meridian 121E, standard parallels 17.5S / 31.5S
        # (fitted to the PDF graticule, 0.1 pt rms) - meridians converge, 129E tilts.
        proj = (ccrs.AlbersEqualArea(central_longitude=121.0, standard_parallels=(-17.5, -31.5))
                if projection == 'albers' else ccrs.PlateCarree())
        ax = fig.add_axes([0.06, 0.05, 0.88, 0.90], projection=proj)
        ax.set_facecolor('white' if style == 'map2022' else 'lightgrey')
        ax.coastlines('50m', linewidth=0.8, color='k', zorder=10)
        tr = ccrs.PlateCarree()
        _geo = True
    except Exception:
        fig, ax = plt.subplots(figsize=(11, 13))
        tr = None; _geo = False

    # Clip tectonic drawing to the WA data bounds (+margin) so the map auto-fits
    # to WA, not the full tectonic polygon extent (which reaches the NT/ocean).
    cb = (Lon.min()-1.0, Lon.max()+1.0, Lat.min()-1.0, Lat.max()+1.0)
    try:
        from wa_basemap import (add_tectonic_background, add_tectonic_outlines,
                                tectonic_legend, add_craton_geology,
                                add_gswa_units, has_gswa_colours)
        if style == 'gswa' and not (litho_npz and has_gswa_colours(litho_npz)):
            print("  (index map: lithology npz has no GSWA colours - re-run "
                  "extract/make_litho.py; using --index-style craton)")
            style = 'craton'
        litho_drawn = None
        if style == 'map2022':
            from wa_basemap import add_map2022_units, add_crustal_boundaries
            if field is not None:          # Moho/Vsv/Xi fill: no colour units, grey terrane outlines
                _draw_field(ax, field, (ext_lo0, ext_lo1, ext_la0, ext_la1), tr)
                add_map2022_units(ax, tect_npz, transform=tr, clip_box=cb, zorder=4.0,
                                  fill=False, edgecolor='0.55', lw=0.3, verbose=False)
            else:
                add_map2022_units(ax, tect_npz, transform=tr, clip_box=cb, zorder=0)
            if labels and labels != 'none':
                from wa_basemap import add_map2022_labels
                lvl  = 2 if labels == 'all' else 1
                excl = {'terrane'} if labels == 'provinces' else None   # craton/orogen/basin only
                # provinces: keep cratons + Kimberley at full size, halve orogens/basins
                oscale = 0.5 if labels == 'provinces' else 1.0
                nl = add_map2022_labels(ax, transform=tr, level=lvl, scale=label_scale,
                                        clip_box=cb, color='k', exclude_class=excl,
                                        other_scale=oscale, full_names=('KIMBERLEY',))
                print(f"  unit labels: {nl} ({labels})")
            try:                           # major crustal boundaries: thin black
                add_crustal_boundaries(ax, bnd_npz, transform=tr, clip_box=cb, zorder=4.2,
                                       lw_litho=1.0, lw_crust=0.5, color='k',
                                       alpha=(0.95 if field is not None else 0.85))
            except Exception as e:
                print(f"  (crustal boundaries unavailable: {e})")
        else:
            add_tectonic_background(ax, alpha=0.5, zorder=0, transform=tr, clip_box=cb)
        if yilgarn:                         # Yilgarn greenstone/rock-type fill + craton contours
            try:
                if yilgarn.get('fill'):
                    # over the GSWA unit fill (z=0) / field (z=1.5), under boundaries & labels
                    _zf = 1.7 if field is not None else 1.2
                    _yilgarn_units_draw(ax, yilgarn['units_npz'], transform=tr, zorder=_zf,
                                        fill_types=yilgarn['fill'], alpha=yilgarn.get('alpha', 1.0),
                                        colour_override=yilgarn.get('colour'))
                _bk = yilgarn.get('bounds', 'all')
                _kinds = {'all': ('craton', 'domain'), 'craton': ('craton',),
                          'domain': ('domain',)}.get(_bk, ())
                if _kinds:
                    _yilgarn_bounds_draw(ax, yilgarn['bnd_npz'], transform=tr, zorder=4.6,
                                         kinds=_kinds)
            except Exception as e:
                print(f"  (Yilgarn overlay unavailable: {e})")
        if cut_line is not None:            # diagnostic: show the cut boundary
            cl = np.asarray(cut_line, float)
            ax.plot(cl[:, 0], cl[:, 1], color='magenta', lw=2.0, ls='--',
                    zorder=9.5, **({'transform': tr} if tr is not None else {}))
        if style == 'gswa':
            add_gswa_units(ax, litho_npz, transform=tr, clip_box=cb, zorder=1)
        elif style == 'craton' and litho_npz:
            litho_drawn = add_craton_geology(ax, litho_npz, zorder=1, transform=tr,
                                             clip_box=cb)
        if style != 'map2022':
            add_tectonic_outlines(ax, lw=0.4, alpha=0.6, zorder=3, major_only=True,
                                  transform=tr, clip_box=cb)
        _tect = True
    except Exception as e:
        print(f"(tectonic bg unavailable: {e})")
        if d is not None:                 # show the model nodes as a stand-in backdrop
            ax.scatter(Lon, Lat, c='0.7', s=10, zorder=1,
                       **({'transform': tr} if tr else {}))
        _tect = False

    if deposit_args is not None:          # mineral-deposit / greenstone overlay (deposits.py)
        try:
            from deposits import overlay_deposits
            overlay_deposits(ax, deposit_args, transform=tr)
        except Exception as e:
            print(f"  (deposits overlay unavailable: {e})")

    lkw = {'transform': tr} if tr else {}
    xyc = tr._as_mpl_transform(ax) if tr else 'data'     # annotate in lon/lat on any projection
    if footprint is not None:          # outline + veil, no zoom (sections may extend past it)
        footprint.focus(ax, tr, veil_alpha=footprint_veil, zoom=False,
                        veil_zorder=4.4)     # over the geology, under stations + sections
    n_sta = 0
    if stations is not None:            # above the ocean mask (5), below section lines
        slon, slat = stations
        n_sta = len(slon)
        if station_open:
            ax.scatter(slon, slat, marker=station_marker, s=stations_size, facecolors='none',
                       edgecolors=station_color, linewidths=0.5, zorder=5.5, **lkw)
        else:
            ax.scatter(slon, slat, marker=station_marker, s=stations_size, c=station_color,
                       edgecolors='k', linewidths=0.3, zorder=5.5, **lkw)
    for ov in (overlays or []):         # extra station layers (e.g. phase 4, permanent)
        olon, olat, ocol, omark, osize = ov[0], ov[1], ov[2], ov[3], ov[4]
        oedge = ov[6] if len(ov) > 6 else 'w'
        ax.scatter(olon, olat, marker=omark, s=osize, c=ocol, edgecolors=oedge,
                   linewidths=0.4, zorder=5.6, **lkw)
        ocodes = ov[7] if len(ov) > 7 else None     # per-station code labels for this layer
        if ocodes is not None:
            import matplotlib.patheffects as _pe
            for _x, _y, _c in zip(olon, olat, ocodes):
                if not _c:
                    continue
                ax.annotate(str(_c), (_x, _y), fontsize=5.5, xycoords=xyc,
                            ha='left', va='center', zorder=8, xytext=(4, 0),
                            textcoords='offset points',
                            path_effects=[_pe.withStroke(linewidth=1.2, foreground='white')])
    if style != 'map2022' and geo is not None and geo['bnd'] is not None:
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(geo['bnd'].lines, colors='firebrick',
                                         linewidths=0.9, alpha=0.8, zorder=4, **lkw))
    for i, (lat1, lon1, lat2, lon2, lab) in enumerate(sections):
        if polylines and lab in polylines:        # draw the true picked path, not a chord
            plo, pla = polylines[lab]
            plat, plon, dist = px._polyline_path(plo, pla)
        else:
            plat, plon, dist = px._great_circle_path(lat1, lon1, lat2, lon2)
        ax.plot(plon, plat, 'w-', lw=3.4, zorder=5, **lkw)   # white halo
        ax.plot(plon, plat, 'k-', lw=2.0, zorder=6, **lkw)
        ax.plot(plon[0], plat[0], 'o', mfc='none', mec='k', mew=0.9, ms=4, zorder=7, **lkw)
        ax.plot(plon[-1], plat[-1], 'o', mfc='none', mec='k', mew=0.9, ms=4, zorder=7, **lkw)
        for dkm in np.arange(0, dist[-1], tick_km):
            j = np.argmin(np.abs(dist - dkm))
            ax.plot(plon[j], plat[j], '|', color='k', ms=7, mew=1.2, zorder=7, **lkw)
            ax.annotate(f'{int(dkm)}', (plon[j], plat[j]), fontsize=5.5, xycoords=xyc,
                        ha='center', va='bottom', zorder=8,
                        xytext=(0, 3), textcoords='offset points',
                        bbox=dict(fc='w', ec='none', alpha=0.6, pad=0.3))
        import matplotlib.patheffects as _pe
        ax.annotate(lab, (plon[0], plat[0]), fontsize=9, fontweight='bold', xycoords=xyc,
                    color='red', zorder=8, xytext=(5,5),
                    textcoords='offset points',
                    path_effects=[_pe.withStroke(linewidth=1.6, foreground='white')])

    # Grey ocean mask + coastline on top (like the depth-slice panels)
    if _geo:
        if style == 'map2022':      # as on the GSWA map: white sea, other states pale grey
            _add_states_ocean(ax, ocean='white', states='#e8e8e8')
        else:
            _add_states_ocean(ax)
        ax.set_extent([ext_lo0, ext_lo1, ext_la0, ext_la1], crs=tr)
        # graticule as on the GSWA map: 5 deg, thin grey, labels '115°' / '20°' bottom + left
        import matplotlib.ticker as mticker
        gcol = '#858c8c'
        glon, glat = np.arange(100, 151, 5), np.arange(-45, 1, 5)
        # lines drawn as our own polylines above the sea and land masks (cartopy's
        # gridliner draws at its own zorder, under the ocean); gridliner only for labels
        lat_d, lon_d = np.linspace(-50, 5, 221), np.linspace(95, 155, 241)
        for lo in glon:
            ax.plot(np.full_like(lat_d, lo), lat_d, color=gcol, lw=0.35, zorder=5.2, transform=tr)
        for la in glat:
            ax.plot(lon_d, np.full_like(lon_d, la), color=gcol, lw=0.35, zorder=5.2, transform=tr)
        gl = ax.gridlines(crs=tr, draw_labels=True, xlocs=glon, ylocs=glat,
                          color='none', linewidth=0)
        gl.top_labels = gl.right_labels = False
        gl.x_inline = gl.y_inline = False
        gl.rotate_labels = False
        gl.xformatter = mticker.FuncFormatter(lambda v, pos=None: f'{v:g}\u00b0')
        gl.yformatter = mticker.FuncFormatter(lambda v, pos=None: f'{abs(v):g}\u00b0')
        gl.xlabel_style = gl.ylabel_style = {'size': 9, 'color': 'k', 'weight': 'heavy'}
    else:
        ax.set_xlim(ext_lo0, ext_lo1)
        ax.set_ylim(ext_la0, ext_la1); ax.set_aspect('equal')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    if box is not None:                       # coverage box of a smaller map (lon0 lon1 lat0 lat1)
        bl0, bl1, bb0, bb1 = box
        tt = np.linspace(0, 1, 100)
        bxlon = np.concatenate([bl0 + tt*(bl1-bl0), np.full(100, bl1),
                                bl1 - tt*(bl1-bl0), np.full(100, bl0)])
        bxlat = np.concatenate([np.full(100, bb0), bb0 + tt*(bb1-bb0),
                                np.full(100, bb1), bb1 - tt*(bb1-bb0)])
        bkw = {'transform': tr} if tr is not None else {}
        ax.plot(bxlon, bxlat, '-', color=box_color, lw=box_lw, zorder=9,
                solid_joinstyle='miter', **bkw)
    if field is not None:
        ax.set_title(field.get('label', 'Field') + (f' ({n_sta} stations)' if n_sta else ''),
                     fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'Cross-section locations ({len(sections)} lines, ticks every {tick_km:g} km'
                     + (f'; {n_sta} stations)' if n_sta else ')'),
                     fontsize=11)
    if _tect and style == 'map2022':
        from wa_basemap import boundaries_legend   # credit now in the map-legend title
        boundaries_legend(ax, loc=bnd_loc, fontsize=7, lw_litho=1.1, lw_crust=0.5)
        if legend and legend != 'none':
            from wa_basemap import add_map2022_legend
            if legend_ages:        # drop rock-type headers + centre sub-eras; keep age numbers; big bold eras
                add_map2022_legend(ax, loc=legend, rock_labels=False, sub_labels=False,
                                   axis_labels=True, era_scale=1.3, era_bold=True,
                                   title='Tectonic units and age')
            else:
                add_map2022_legend(ax, loc=legend)
    elif _tect and style == 'gswa':
        ax.text(0.01, 0.99, 'Geology: GSWA 1:500k tectonic units, GSWA colours\n'
                '(\u00a9 Geological Survey of Western Australia, CC-BY-4.0)',
                transform=ax.transAxes, ha='left', va='top', fontsize=7, zorder=12,
                bbox=dict(fc='w', ec='0.6', alpha=0.9, pad=3))
    elif _tect:
        tectonic_legend(ax, loc='upper left', fontsize=7,
                        clip_box=(ext_lo0, ext_lo1, ext_la0, ext_la1),
                        exclude=('STATE',),
                        litho=litho_drawn)
    # station-type legend (WA Array + any overlay layers), as a separate legend
    if stations is not None or overlays:
        from matplotlib.lines import Line2D
        from matplotlib.legend import Legend
        sh = []
        if stations is not None:
            sh.append(Line2D([], [], marker=station_marker, color='none',
                             markerfacecolor=('none' if station_open else station_color),
                             markeredgecolor=('k' if not station_open else station_color),
                             markersize=8, label=stations_label))
        for ov in (overlays or []):
            sh.append(Line2D([], [], marker=ov[3], color='none', markerfacecolor=ov[2],
                             markeredgecolor=(ov[6] if len(ov) > 6 else 'w'),
                             markersize=8, label=ov[5]))
        _DEP_LEGEND = {'gold': ('o', (1.00, 0.90, 0.00)), 'copper': ('^', (0.85, 0.0, 0.0)),
                       'lithium': ('h', (0.60, 0.20, 0.80)), 'nickel': ('h', (0.20, 0.70, 0.20))}
        dep_names = []
        if deposit_args is not None and getattr(deposit_args, 'deposits', None):
            dep_names = [s.strip().lower() for s in deposit_args.deposits.split(',') if s.strip()]
        for nm in dep_names:
            if nm in _DEP_LEGEND:
                mk, col = _DEP_LEGEND[nm]
                sh.append(Line2D([], [], marker=mk, color='none', markerfacecolor=col,
                                 markeredgecolor='k', markersize=8, label=nm.capitalize()))
        if sh:
            _title = 'Stations & deposits' if dep_names else 'Stations'
            leg_s = Legend(ax, sh, [h.get_label() for h in sh], loc='lower right',
                           fontsize=7, framealpha=0.9, title=_title, title_fontsize=7)
            ax.add_artist(leg_s)
    import os
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, out_name +
                       ('.fp' if footprint is not None else '') +
                       ('_w_station' if (stations is not None or overlays) else '') + '.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.png','.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"Index map: {out}")




# ── standalone CLI ────────────────────────────────────────────────────────────
def main():
    import argparse
    import plot_xsection as px
    ap = argparse.ArgumentParser(
        description="Standalone WA map: profile lines and station layers on the GSWA "
                    "tectonic basemap. No Vs model needed (extent from --extent or the "
                    "sections/stations).")
    ap.add_argument('--sections', default=None, help='sections.txt (lat1 lon1 lat2 lon2 label)')
    ap.add_argument('--load-polylines', default=None, help='polylines npz (draw true picked paths)')
    ap.add_argument('--extent', nargs=4, type=float, default=None,
                    metavar=('LON0', 'LON1', 'LAT0', 'LAT1'), help='map frame')
    ap.add_argument('--stations', default=None, help='WA Array stations file (code lat lon)')
    ap.add_argument('--no-stations', action='store_true')
    ap.add_argument('--station-marker', default='^', help="WA Array station marker (default '^')")
    ap.add_argument('--station-color', default='k', help="WA Array station colour (default black)")
    ap.add_argument('--station-open', action='store_true', help='open (unfilled) station markers')
    ap.add_argument('--phase4', default=None,
                    help='Phase 4 station CSV (Latitude/Longitude cols) - drawn as green triangles')
    ap.add_argument('--map-stations', nargs='+', action='append', default=[],
                    metavar='CSV COLOUR MARKER LABEL [EDGE]',
                    help='extra station layer from a CSV with Latitude/Longitude columns '
                         '(repeatable). EDGE (marker edge colour) is optional, default white. '
                         'e.g. --map-stations phase4.csv lime s "Phase 4"  or  '
                         '--map-stations perm.csv red ^ Permanent black')
    ap.add_argument('--stations-label', default='Available',
                    help='legend label for the --stations layer (red squares)')
    ap.add_argument('--label-stations', nargs='+', default=None, metavar='LABEL',
                    help='annotate stations with their station code, for the named '
                         '--map-stations layer(s) only (matched by their LABEL; quote labels '
                         'with spaces). The CSV must have a code column '
                         '(Code/Station/Sta/Site_ID/Site/Name). e.g. --label-stations Permanent')
    ap.add_argument('--footprint', action='store_true')
    ap.add_argument('--footprint-npz', default=None)
    ap.add_argument('--footprint-outline', default=None)
    ap.add_argument('--footprint-veil', type=float, default=0.5)
    ap.add_argument('--moho', default=None)
    # ---- scalar field fill (Moho depth now; Vsv/Xi later) ----
    ap.add_argument('--moho-grid', default=None,
                    help='Moho depth grid (lon lat moho_km text table) - fills the map')
    ap.add_argument('--moho-skip', type=int, default=11, help='header lines in --moho-grid (default 11)')
    ap.add_argument('--moho-field', default='z_moho',
                    help="field in a .mat Moho pick: z_moho (depth) or amp_moho")
    ap.add_argument('--moho-amp', default=None, metavar='PMAT',
                    help='Moho RF amplitude map: peak of the pvh.6 CCP cube within +/-amp-win '
                         'of the final Moho. PMAT = the pvh.6 *.P.mat; needs --moho-grid .mat')
    ap.add_argument('--amp-win', type=float, default=3.0,
                    help='+/- km window around the Moho for the amplitude peak (default 3)')
    ap.add_argument('--moho-fill-hk', default=None, metavar='HK',
                    help='fill the Moho north/gaps (beyond the CCP grid) with H-K station Moho '
                         '(h_new) kriged onto the extent; keeps continuous CCP where it exists. '
                         'HK = the HK summary table.')
    ap.add_argument('--clip-stations', default=None, metavar='HK',
                    help='mask ANY field (Vsv/xi/amp/Moho) to within --clip-dist of these stations '
                         '(the HK table), so the footprint matches the Vp/Vs map.')
    ap.add_argument('--clip-dist', type=float, default=0.8,
                    help='degrees for --clip-stations (default 0.8, same as Vp/Vs --field-max-dist)')
    ap.add_argument('--no-stn-mask', action='store_true',
                    help='do NOT apply the CCP stn_mask to Moho/Vsv/xi/amp; show the data '
                         'wherever it has a value (holes stay blank), clipped by --cut-poly. '
                         'Reveals the northern Capricorn/Hamersley strip the stn_mask hides.')
    ap.add_argument('--vsavg', default=None,
                    help='crustal-average Vsv map from a Vsv npz (needs --moho-grid .mat)')
    ap.add_argument('--xiavg', default=None,
                    help='crustal-average xi map from a ZT npz (needs --moho-grid .mat)')
    ap.add_argument('--avg-window', nargs=2, default=None, metavar=('A', 'B'),
                    help="depth window for --vsavg/--xiavg: two bounds, each a number (km) or "
                         "'moho'. e.g. 0 5 | 5 20 | 20 moho | moho 60 | 30 30 (single slice). "
                         "Default 0 moho (whole crust).")
    ap.add_argument('--avg-top', type=float, default=None,
                    help='shorthand: average from this depth to the Moho (= --avg-window TOP moho)')
    ap.add_argument('--avg-field', default=None, help='override npz field name (default Vsv / Xi)')
    # ---- H-K station Vp/Vs (scattered -> kriged) ----
    ap.add_argument('--hk', default=None, help='HK summary table (per-station); kriged Vp/Vs map')
    ap.add_argument('--hk-field', default='k_new',
                    help='HK column to map: k_new (Vp/Vs, default), h_new, vp, vs, k_ref')
    ap.add_argument('--hk-interp', default='kriging', choices=['kriging', 'rbf', 'cubic', 'linear'],
                    help='interpolation for the scattered HK stations (default kriging)')
    ap.add_argument('--ccp-mask', default=None,
                    help='moho_ccp_refined.final.mat: krige HK onto the CCP grid + stn_mask '
                         '(small-area masking, same as Vsv/xi/Moho). Omit for the whole-state map.')
    ap.add_argument('--field-clim', nargs=2, type=float, default=None, metavar=('LO', 'HI'))
    ap.add_argument('--field-cmap', default='DEFAULT',
                    help="colormap: a matplotlib name, parula/parula_r, rwb, or a .mat "
                         "colormap path (e.g. cmapx.mat). Default = parula_r for Moho depth")
    ap.add_argument('--field-ncolors', type=int, default=0,
                    help='discrete colour levels (0 = per-map default: Moho 20, xi/Vsv/HK 16)')
    ap.add_argument('--field-style', default=None, choices=['pcolor', 'contourf', 'smooth'],
                    help="field rendering: pcolor (cells), contourf (smooth filled, edge=none), "
                         "smooth (gouraud). Default contourf for HK, pcolor otherwise.")
    ap.add_argument('--hk-markers', dest='hk_markers', action='store_true', default=True,
                    help='show the HK station locations as markers (default on)')
    ap.add_argument('--no-hk-markers', dest='hk_markers', action='store_false')
    ap.add_argument('--field-label', default='Moho depth (km)')
    ap.add_argument('--field-alpha', type=float, default=0.9)
    ap.add_argument('--field-interp', default='rbf', choices=['rbf', 'kriging', 'cubic', 'linear'])
    ap.add_argument('--field-dgrid', type=float, default=0.05, help='field grid step (deg)')
    ap.add_argument('--field-max-dist', type=float, default=0.6,
                    help='blank field cells > this many deg from a node')
    ap.add_argument('--style', dest='index_style', default='map2022',
                    help='map2022 | gswa | craton | plain')
    ap.add_argument('--tick', dest='index_tick', type=float, default=100)
    ap.add_argument('--labels', dest='index_labels', default='major',
                    choices=['provinces', 'major', 'all', 'none'],
                    help="provinces = craton/orogen/basin names only (no sub-terranes); "
                         "major = + Yilgarn terranes; all = every unit; none")
    ap.add_argument('--label-scale', type=float, default=2.8,
                    help='unit-label font multiplier (2.8 = double the old 1.4 default)')
    ap.add_argument('--legend', dest='index_legend', default='upper left')
    ap.add_argument('--legend-ages', action='store_true',
                    help='units legend: drop rock-type headers + left axis, big bold era names only')
    ap.add_argument('--boundaries-loc', dest='bnd_loc', default='lower left',
                    help="corner for the crustal-boundaries legend (default 'lower left')")
    ap.add_argument('--box', nargs=4, type=float, default=None,
                    metavar=('LON0', 'LON1', 'LAT0', 'LAT1'),
                    help='draw a coverage box (e.g. a smaller map extent) on the map')
    ap.add_argument('--box-color', default='k')
    ap.add_argument('--box-lw', type=float, default=1.4)
    ap.add_argument('--proj', dest='index_proj', default='albers', choices=['albers', 'plate'])
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--bnd-npz', default=None)
    ap.add_argument('--litho-npz', default=None)
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--name', dest='out_name', default='xsection_index_map',
                    help="output file base name (default 'xsection_index_map')")
    ap.add_argument('--station-south-of', type=float, default=None, metavar='LAT',
                    help='only plot stations south of this latitude (e.g. -22 to drop '
                         'everything north of the Hamersley Basin)')
    ap.add_argument('--station-exclude-box', nargs=4, type=float, action='append', default=[],
                    metavar=('LON0', 'LON1', 'LAT0', 'LAT1'),
                    help='drop stations inside this lon/lat box (repeatable); applies to the '
                         'WA Array layer and any --map-stations/--phase4 overlay')
    ap.add_argument('--cut-poly', default=None, metavar='FILE',
                    help='lon/lat polyline (NW->SE) defining a NE cut boundary; keep only '
                         'stations (and colour) on its SW side. e.g. data/masks/wa_array_cut.txt')
    ap.add_argument('--draw-cut', action='store_true',
                    help='draw the --cut-poly line on the map (magenta dashed) to see which '
                         'stations fall inside vs outside it')
    ap.add_argument('--yilgarn', action='store_true',
                    help='overlay the Yilgarn craton: fill rock types (--yilgarn-fill) and draw '
                         'the craton outline (red) + terrane/domain boundaries (grey)')
    ap.add_argument('--yilgarn-fill', default='greenstone',
                    help="rock types to fill, comma-separated (greenstone,granite,gneiss,"
                         "sedimentary) or 'all' or 'none' (default greenstone)")
    ap.add_argument('--yilgarn-colour', default=None,
                    help='single fill colour for all --yilgarn-fill types (else per-type colours)')
    ap.add_argument('--yilgarn-alpha', type=float, default=0.9, help='Yilgarn fill alpha (default 0.9)')
    ap.add_argument('--yilgarn-bounds', choices=['all', 'craton', 'domain', 'none'], default='all',
                    help="which Yilgarn boundaries to draw: all (red craton + grey terrane), "
                         "craton (red outline only), domain (grey only), none (default all)")
    ap.add_argument('--yilgarn-units-npz', default='data/geology/yilgarn_units.npz')
    ap.add_argument('--yilgarn-bnd-npz', default='data/geology/yilgarn_boundaries.npz')
    _have_deposits = False
    try:                                   # mineral-deposit / greenstone overlay (deposits.py)
        from deposits import add_deposit_args
        add_deposit_args(ap); _have_deposits = True
    except Exception as e:
        print(f"  (deposits options unavailable: {e})", file=sys.stderr)
    args = ap.parse_args()

    def _repo_rel(p):                       # resolve a path relative to the repo root too
        if p and not os.path.isfile(p):
            alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)
            if os.path.isfile(alt):
                return alt
        return p

    yilgarn = None
    if args.yilgarn:
        _ft = [] if args.yilgarn_fill.lower() == 'none' else (
            ['greenstone', 'granite', 'gneiss', 'sedimentary', 'other']
            if args.yilgarn_fill.lower() == 'all' else
            [s.strip() for s in args.yilgarn_fill.split(',') if s.strip()])
        un, bn = _repo_rel(args.yilgarn_units_npz), _repo_rel(args.yilgarn_bnd_npz)
        for lbl, pth in (('units', un), ('bounds', bn)):
            if not os.path.isfile(pth):
                ap.error(f"--yilgarn {lbl} npz not found: {pth} (cwd={os.getcwd()})")
        yilgarn = dict(fill=_ft, bounds=args.yilgarn_bounds, colour=args.yilgarn_colour,
                       alpha=args.yilgarn_alpha, units_npz=un, bnd_npz=bn)
        if not _ft and args.yilgarn_bounds == 'none':
            ap.error("--yilgarn with nothing to draw (fill none + bounds none)")

    cut_path = None; cut_line = None
    if args.cut_poly:
        cp = _repo_rel(args.cut_poly)
        if not os.path.isfile(cp):
            ap.error(f"--cut-poly file not found: {args.cut_poly} (cwd={os.getcwd()})")
        _cpts = _read_cut_poly(cp)
        cut_path = _keep_poly(_cpts)
        if args.draw_cut:
            cut_line = _cpts
        print(f"  cut-poly: keeping SW side of {cp}")

    sections = px._load_sections(args.sections) if args.sections else []
    poly = None
    if args.load_polylines:
        p = np.load(args.load_polylines, allow_pickle=True)
        poly = {str(p['short'][i]): (np.asarray(p['lon'][i], float),
                                     np.asarray(p['lat'][i], float))
                for i in range(len(p['short']))}
    sta = None
    if not args.no_stations:                 # default to config.STATIONS, as --index-map does
        try:
            import config
            sta_path, cols = args.stations or config.STATIONS, config.STATION_COLS
        except Exception:
            sta_path, cols = args.stations, (2, 1)
        if sta_path and os.path.isfile(sta_path):
            sta = px._load_stations(sta_path, cols)
            print(f"  stations: {len(sta[0])} <- {sta_path}")
        else:
            print(f"  stations: file not found ({sta_path}) - none drawn "
                  f"(pass --stations, or --no-stations to silence)")
    label_set = set(args.label_stations or [])
    overlays = []
    for spec in args.map_stations:
        if len(spec) < 4:
            ap.error(f"--map-stations needs CSV COLOUR MARKER LABEL [EDGE]; got {spec}")
        csv_path, col, mark, lab = spec[0], spec[1], spec[2], spec[3]
        edge = spec[4] if len(spec) > 4 else 'w'
        if not os.path.isfile(csv_path):
            print(f"  map-stations: file not found ({csv_path}) - skipped"); continue
        olon, olat, ocodes = _load_station_csv(csv_path)
        want_codes = lab in label_set
        if want_codes and ocodes is None:
            print(f"  map-stations: '{lab}' has no code column (Code/Station/Sta/Site_ID/"
                  f"Site/Name) in {csv_path} - labels skipped")
        codes = ocodes if want_codes else None
        overlays.append((olon, olat, col, mark, 22, lab, edge, codes))
        print(f"  map-stations: {len(olon)} '{lab}' <- {csv_path} ({col} {mark}, edge {edge}"
              + (", + codes" if codes is not None else "") + ")")
    for want in label_set:
        if want not in [spec[3] for spec in args.map_stations]:
            print(f"  --label-stations: no --map-stations layer labelled '{want}'")
    if args.phase4:                          # Phase 4 as green triangles, black edge
        if os.path.isfile(args.phase4):
            p4lon, p4lat, _ = _load_station_csv(args.phase4)
            overlays.append((p4lon, p4lat, 'limegreen', '^', 26, 'Phase 4', 'k', None))
            print(f"  phase4: {len(p4lon)} green triangles <- {args.phase4}")
        else:
            print(f"  phase4: file not found ({args.phase4}) - skipped")
    field = None
    clim = tuple(args.field_clim) if args.field_clim else None
    _cm = lambda dflt: args.field_cmap if args.field_cmap != 'DEFAULT' else dflt
    _nc = lambda dflt: args.field_ncolors if args.field_ncolors > 0 else dflt
    _HK_LABELS = {'k_new': 'Vp/Vs', 'k_ref': 'Vp/Vs (ref)', 'h_new': 'H (km)',
                  'vp': 'Vp (km/s)', 'vs': 'Vs (km/s)'}
    if args.hk:                                     # per-station H-K Vp/Vs, kriged
        hlon, hlat, hval = _read_hk(args.hk, args.hk_field)
        lbl = _HK_LABELS.get(args.hk_field.lower(), args.hk_field)
        if cut_path is not None:                              # drop stations NE of the cut line
            k = cut_path.contains_points(np.column_stack([hlon, hlat]))
            print(f"  cut-poly: HK stations {len(hlon)} -> {int(k.sum())} kept (SW of cut)")
            hlon, hlat, hval = hlon[k], hlat[k], hval[k]
        print(f"  HK: {len(hlon)} stations, {args.hk_field} {np.nanmin(hval):.3f}-{np.nanmax(hval):.3f}")
        if args.ccp_mask and os.path.isfile(args.ccp_mask):   # small area: krige onto CCP grid + stn_mask
            LON2, LAT2, MASK = _read_ccp_grid(args.ccp_mask)
            V = _krige_points(hlon, hlat, hval, LON2.ravel(), LAT2.ravel(),
                              args.hk_interp).reshape(LON2.shape)
            V = np.where(MASK, V, np.nan)
            print(f"  HK kriged onto CCP grid ({args.hk_interp}): {int(np.isfinite(V).sum())} cells")
        else:                                                 # whole state: krige onto a regular grid
            lo0, lo1, la0, la1 = tuple(args.extent) if args.extent else (112, 129.6, -36, -13.2)
            dg = args.field_dgrid if args.field_dgrid != 0.05 else 0.1
            glon = np.arange(lo0, lo1 + dg, dg); glat = np.arange(la0, la1 + dg, dg)
            LON2, LAT2 = np.meshgrid(glon, glat)
            V = _krige_points(hlon, hlat, hval, LON2.ravel(), LAT2.ravel(),
                              args.hk_interp).reshape(LON2.shape)
            try:                                              # mask to WA land
                from plot_moho_map import wa_mask
                m = wa_mask(glon, glat)
                if m is not None:
                    V = np.where(m, V, np.nan)
            except Exception as e:
                print(f"  (WA land mask unavailable: {e})")
            mx = 1.5 if args.field_max_dist == 0.6 else args.field_max_dist   # blank big station gaps
            from scipy.spatial import cKDTree
            d, _ = cKDTree(np.column_stack([hlon, hlat])).query(np.column_stack([LON2.ravel(), LAT2.ravel()]))
            V[(d > mx).reshape(V.shape)] = np.nan
            print(f"  HK kriged state-wide ({args.hk_interp}, dgrid {dg}, max_dist {mx}): "
                  f"{int(np.isfinite(V).sum())} cells")
        V = _nan_smooth(V, 1.0)                        # light smoothing for a clean contourf
        # per-field colour: Vp/Vs -> rgb (red low -> green mid -> blue high), Moho depth -> parula_r
        _hf = args.hk_field.lower()
        if _hf in ('h_new', 'h_old', 'moho_ref'):      # station Moho depth (fills north like Vp/Vs)
            _hcmap, _hclim, _hnc = 'parula_r', (30, 55), 16
        elif _hf in ('vp',):
            _hcmap, _hclim, _hnc = 'rgb', (5.8, 6.6), 16
        elif _hf in ('vs',):
            _hcmap, _hclim, _hnc = 'rgb', (3.3, 3.8), 16
        else:                                          # k_new / k_old / k_ref = Vp/Vs (rgb)
            _hcmap, _hclim, _hnc = 'rgb', (1.70, 1.86), 16
        field = dict(grid2d=(LON2, LAT2, V), label=lbl, cmap=_cm(_hcmap),
                     clim=(clim if clim else _hclim), ncolors=_nc(_hnc), alpha=args.field_alpha,
                     style=(args.field_style or 'contourf'))
        if args.hk_markers:                            # show the HK stations (incl. COPA) as markers
            sta = (hlon, hlat); args.stations_label = 'HK / COPA'
    elif args.vsavg or args.xiavg:                  # depth-window-average Vsv / xi (needs the Moho .mat)
        if not (args.moho_grid and os.path.isfile(args.moho_grid) and args.moho_grid.endswith('.mat')):
            print("  --vsavg/--xiavg need --moho-grid = moho_ccp_refined.final.mat (for the Moho surface)")
        else:
            _p = lambda s: np.inf if str(s).lower().startswith('moho') else float(s)
            if args.avg_window is not None:
                win = (_p(args.avg_window[0]), _p(args.avg_window[1]))
            elif args.avg_top is not None:
                win = (args.avg_top, np.inf)
            else:
                win = (0.0, np.inf)
            tag, desc = _depth_desc(win[0], win[1])
            LON2, LAT2, ZC = _read_moho_mat(args.moho_grid, 'z_moho', apply_mask=not args.no_stn_mask)
            if args.vsavg:
                V = _avg_model_window(args.vsavg, args.avg_field or 'Vsv', LON2, LAT2, ZC, win)
                field = dict(grid2d=(LON2, LAT2, V), clim=clim, alpha=args.field_alpha,
                             cmap=_cm('parula_r'),       # cmapx.mat via --field-cmap to match exactly
                             ncolors=_nc(16), style=(args.field_style or 'contourf'), label=f'Vsv (km/s), {desc}')
            else:
                V = _avg_model_window(args.xiavg, args.avg_field or 'Xi', LON2, LAT2, ZC, win)
                # xi is diverging about 1.0 (rwb: red<1, white=1, blue>1) -> clim MUST be
                # symmetric about 1. crustal -> 0.96-1.04; sub-Moho -> symmetric auto about 1.
                if clim:
                    xclim = tuple(clim)
                elif not np.isinf(win[0]):
                    xclim = (0.96, 1.04)                 # crustal windows
                else:
                    g = V[np.isfinite(V)]
                    dv = float(np.percentile(np.abs(g - 1.0), 98)) if g.size else 0.04
                    dv = max(dv, 0.02)
                    xclim = (1.0 - dv, 1.0 + dv)         # sub-Moho: white stays at 1.0
                    print(f"  xi sub-Moho symmetric clim about 1.0: [{xclim[0]:.3f} {xclim[1]:.3f}]")
                field = dict(grid2d=(LON2, LAT2, V), alpha=args.field_alpha,
                             clim=xclim, cmap=_cm('rwb'), ncolors=_nc(16), style=(args.field_style or 'contourf'),
                             label=r'$\xi=(V_{SH}/V_{SV})^2$' + f', {desc}')
            args.out_name = f'{args.out_name}.{tag}'     # distinct file per depth window
    elif args.moho_amp:                             # Moho RF amplitude (peak of pvh.6 cube)
        if not (args.moho_grid and os.path.isfile(args.moho_grid) and args.moho_grid.endswith('.mat')):
            print("  --moho-amp needs --moho-grid = moho_ccp_refined.final.mat (for the Moho surface)")
        elif not os.path.isfile(args.moho_amp):
            print(f"  --moho-amp P.mat not found ({args.moho_amp})")
        else:
            LON2, LAT2, ZC = _read_moho_mat(args.moho_grid, 'z_moho', apply_mask=not args.no_stn_mask)
            amask = np.isfinite(ZC)                  # stn_mask footprint (same as other maps)
            V = _amp_at_moho(args.moho_amp, LON2, LAT2, ZC, win=args.amp_win)
            V = _nan_smooth(V, 1.5)                  # amp is noisy (MATLAB SMOOTH_SIG=1.5)
            V[~amask] = np.nan                       # re-apply mask after smoothing (MATLAB does this)
            field = dict(grid2d=(LON2, LAT2, V), alpha=args.field_alpha,
                         cmap=_cm('turbo_r'),         # MATLAB flipud(turbo(10))
                         clim=(clim if clim else (0.0, 0.1)), ncolors=_nc(10),
                         style=(args.field_style or 'contourf'),
                         label='RF amplitude at Moho (pvh.6 cube)')
    elif args.moho_grid:
        if os.path.isfile(args.moho_grid):
            if args.moho_grid.endswith('.mat'):         # moho_ccp_refined.final.mat (native CCP grid)
                LON2, LAT2, V = _read_moho_mat(args.moho_grid, args.moho_field, apply_mask=not args.no_stn_mask)
                if args.moho_field.lower() in ('amp_moho', 'amp'):   # picked RF amplitude at Moho
                    g = V[np.isfinite(V)]
                    aclim = clim if clim else (0.0, float(np.percentile(g, 98)) if g.size else 0.1)
                    field = dict(grid2d=(LON2, LAT2, V), alpha=args.field_alpha,
                                 cmap=_cm('turbo'), clim=aclim, ncolors=_nc(16),
                                 style=(args.field_style or 'contourf'),
                                 label=args.field_label or 'RF amplitude at Moho')
                    print(f"  moho amplitude (picked amp_moho): {int(np.isfinite(V).sum())} nodes, "
                          f"clim [{aclim[0]:.3g} {aclim[1]:.3g}]")
                else:
                    if args.moho_field == 'z_moho' and clim is None:
                        clim = (30, 55)                 # MATLAB flipud(parula), clim [30 55]
                    if args.moho_fill_hk and os.path.isfile(args.moho_fill_hk):
                        # fill the north/gaps (beyond the CCP grid) with H-K station Moho,
                        # keep continuous CCP everywhere it has a value.
                        lo0, lo1, la0, la1 = tuple(args.extent) if args.extent else (112, 129.6, -36, -13.2)
                        dg = args.field_dgrid if args.field_dgrid != 0.05 else 0.1
                        glon = np.arange(lo0, lo1 + dg, dg); glat = np.arange(la0, la1 + dg, dg)
                        GLON, GLAT = np.meshgrid(glon, glat)
                        hlon, hlat, hval = _read_hk(args.moho_fill_hk, 'h_new')
                        HKg = _krige_points(hlon, hlat, hval, GLON.ravel(), GLAT.ravel(),
                                            args.hk_interp).reshape(GLON.shape)
                        from scipy.spatial import cKDTree
                        mx = 1.5 if args.field_max_dist == 0.6 else args.field_max_dist
                        dd, _ = cKDTree(np.column_stack([hlon, hlat])).query(
                            np.column_stack([GLON.ravel(), GLAT.ravel()]))
                        HKg[(dd > mx).reshape(HKg.shape)] = np.nan          # station footprint
                        from scipy.interpolate import griddata
                        gd = np.isfinite(V)
                        CCPg = griddata((LON2[gd], LAT2[gd]), V[gd], (GLON, GLAT), method='linear')
                        BV = np.where(np.isfinite(CCPg), CCPg, HKg)         # CCP core + H-K fill
                        BV[(dd > mx).reshape(BV.shape)] = np.nan            # clip ALL to station footprint (east like Vp/Vs)
                        BV = _nan_smooth(BV, 1.0)
                        BV[(dd > mx).reshape(BV.shape)] = np.nan            # keep footprint after smoothing
                        LON2, LAT2, V = GLON, GLAT, BV
                        print(f"  moho CCP + H-K fill: {int(np.isfinite(V).sum())} cells "
                              f"(CCP {int(np.isfinite(CCPg).sum())}, H-K filled the rest)")
                    field = dict(grid2d=(LON2, LAT2, V), label=args.field_label, alpha=args.field_alpha,
                                 cmap=_cm('parula_r'), clim=clim, ncolors=_nc(16), style=(args.field_style or 'contourf'))
                    if not args.moho_fill_hk:
                        print(f"  moho field (.mat native grid): {int(np.isfinite(V).sum())} nodes "
                              f"<- {args.moho_grid} [{args.moho_field}]")
            else:                                       # text table lon lat value -> RBF grid
                from plot_moho_map import read_moho
                flon, flat, fval = read_moho(args.moho_grid, args.moho_skip)
                field = dict(lon=flon, lat=flat, val=fval, label=args.field_label, cmap=_cm('parula_r'),
                             clim=clim, alpha=args.field_alpha, interp=args.field_interp,
                             dgrid=args.field_dgrid, max_dist=args.field_max_dist)
                print(f"  moho field (text grid): {len(flon)} nodes <- {args.moho_grid}")
        else:
            print(f"  moho grid not found ({args.moho_grid}) - no field drawn")

    # ── clip-stations: mask ANY field to within --clip-dist of the stations (Vp/Vs footprint) ──
    if args.clip_stations and os.path.isfile(args.clip_stations) and field is not None \
            and 'grid2d' in field and field['grid2d'] is not None:
        from scipy.spatial import cKDTree
        clon, clat, _ = _read_hk(args.clip_stations, 'k_new')
        LON2c, LAT2c, Vc = field['grid2d']
        dd, _ = cKDTree(np.column_stack([clon, clat])).query(
            np.column_stack([np.asarray(LON2c).ravel(), np.asarray(LAT2c).ravel()]))
        Vc = np.where((dd > args.clip_dist).reshape(np.shape(Vc)), np.nan, Vc)
        field['grid2d'] = (LON2c, LAT2c, Vc)
        print(f"  clip-stations: field masked to within {args.clip_dist} deg of "
              f"{len(clon)} stations -> {int(np.isfinite(Vc).sum())} cells")

    # ── cut-poly: clip ANY field (HK Vp/Vs, Vsv, xi, Moho) to the SW side of the cut line ──
    if cut_path is not None and field is not None:
        if 'grid2d' in field and field['grid2d'] is not None:
            LON2f, LAT2f, Vf = field['grid2d']
            inside = cut_path.contains_points(
                np.column_stack([np.asarray(LON2f).ravel(), np.asarray(LAT2f).ravel()])).reshape(np.shape(Vf))
            Vf = np.where(inside, Vf, np.nan)
            field['grid2d'] = (LON2f, LAT2f, Vf)
            print(f"  cut-poly: field clipped to SW side -> {int(np.isfinite(Vf).sum())} cells")
        elif 'lon' in field and 'val' in field:            # scattered (text-grid) field
            k = cut_path.contains_points(np.column_stack([field['lon'], field['lat']]))
            field['lon'], field['lat'], field['val'] = field['lon'][k], field['lat'][k], field['val'][k]
            print(f"  cut-poly: field points clipped to SW side -> {int(k.sum())}")

    geo = None
    if args.index_style != 'map2022':
        try:
            geo = px._load_geology('domain,boundaries,names',
                                   args.litho_npz, args.tect_npz, args.bnd_npz)
        except Exception as e:
            print(f"  geology unavailable ({e})")
    fp = None
    if args.footprint:
        try:
            fp = px._index_footprint(args)
        except Exception as e:
            print(f"  footprint unavailable ({e})")

    def _station_keep(lon, lat):                   # True = keep this station
        lon = np.asarray(lon, float); lat = np.asarray(lat, float)
        keep = np.ones(len(lat), bool)
        if args.station_south_of is not None:      # drop everything north of this latitude
            keep &= lat <= args.station_south_of
        for b in (args.station_exclude_box or []): # drop everything inside each lon/lat box
            lo0, lo1, la0, la1 = b
            keep &= ~((lon >= min(lo0, lo1)) & (lon <= max(lo0, lo1))
                      & (lat >= min(la0, la1)) & (lat <= max(la0, la1)))
        if cut_path is not None:                   # keep only the SW side of the cut line
            keep &= cut_path.contains_points(np.column_stack([lon, lat]))
        return keep
    if args.station_south_of is not None or args.station_exclude_box or cut_path is not None:
        if sta is not None:
            keep = _station_keep(sta[0], sta[1])
            sta = (np.asarray(sta[0])[keep], np.asarray(sta[1])[keep]) + tuple(sta[2:])
            print(f"  stations kept after clip: {int(keep.sum())}")
        cov = []
        for ov in overlays:
            keep = _station_keep(ov[0], ov[1])
            ov = list(ov); ov[0] = np.asarray(ov[0])[keep]; ov[1] = np.asarray(ov[1])[keep]
            if len(ov) > 7 and ov[7] is not None:
                ov[7] = [c for c, k in zip(ov[7], keep) if k]
            cov.append(tuple(ov))
            print(f"  '{ov[5]}' kept after clip: {int(keep.sum())}")
        overlays = cov

    if not sections and sta is None and not overlays and args.extent is None:
        ap.error("give something to draw: --sections, --stations, --map-stations, or --extent")

    plot_map(sections, args.out_dir, d=None,
             extent=tuple(args.extent) if args.extent else None,
             moho=args.moho,
             litho_npz=(None if args.index_style == 'plain' else
                        px._resolve_npz('LITHOLOGY_NPZ', 'wa_lithology.npz', args.litho_npz)),
             tick_km=args.index_tick, stations=sta, style=args.index_style,
             footprint=fp, footprint_veil=args.footprint_veil,
             tect_npz=px._resolve_npz('TECTONICS_NPZ', 'wa_tectonics.npz', args.tect_npz),
             bnd_npz=px._resolve_npz('BOUNDARIES_NPZ', 'wa_crustal_boundaries.npz', args.bnd_npz),
             projection=args.index_proj, legend=args.index_legend, labels=args.index_labels,
             label_scale=args.label_scale, polylines=poly, overlays=overlays, geo=geo,
             stations_label=args.stations_label,
             box=tuple(args.box) if args.box else None,
             box_color=args.box_color, box_lw=args.box_lw,
             station_marker=args.station_marker, station_color=args.station_color,
             station_open=args.station_open, out_name=args.out_name,
             legend_ages=args.legend_ages, bnd_loc=args.bnd_loc, field=field,
             yilgarn=yilgarn, cut_line=cut_line,
             deposit_args=(args if _have_deposits else None))


if __name__ == '__main__':
    main()
