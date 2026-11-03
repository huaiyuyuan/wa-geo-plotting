#!/usr/bin/env python3
"""
plot_xsection.py — Cross-section along a great circle path.
4×1 panels: Vsv | dlnVsv | Data misfit | Ray density (Ndat)

Usage:
  python3 plot_xsection.py \
      --fvs   Fvs.iter.1.Z.npz \
      --start -35.0,115.0 \
      --end   -20.0,128.0 \
      --label MW2 \
      --out-dir figures/xsections

  # Multiple sections in one run:
  python3 plot_xsection.py \
      --fvs   Fvs.iter.1.Z.npz \
      --sections "-35,115/-20,128/MW2" "-32,114/-24,127/SW1" \
      --out-dir figures/xsections

  # Surface-geology strips above the top panel + crustal-boundary lines:
  python3 plot_xsection.py --fvs Fvs.iter.2.Z.npz --load-sections sections.txt \
      # default strips: domain,boundaries,names (2022 tectonic-map colours + boundary arrows)
  python3 plot_xsection.py ... --strips litho,domain,boundaries,names   # + 500k lithology
      litho       GSWA 500k lithology strip (velocity-oriented colours)
      domain      terrane strip (10M TECTNAME, labelled in the strip)
      boundaries  dashed lines where Major Crustal Boundaries cross the section
      names       label the boundary crossings with their NAME
  npz paths come from config.py, else ./data/ (override: --litho-npz etc.)

H. Yuan / Claude, Sep 2026
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Repo root (config.py), basemap/ (wa_basemap) and plotting/ on the import path,
# so the script runs from anywhere.
_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / 'basemap', _REPO / 'plotting'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ── Section geology strips (lithology / terranes / crustal boundaries) ───────
_GEO = None          # loaded once in main(); workers reload only if not inherited
_STRIP_ITEMS = {'litho', 'domain', 'boundaries', 'names'}
_STRIPS_DEFAULT = 'domain,boundaries,names'   # same domain colours as the index map; add litho if wanted


def _resolve_npz(cfg_name, fname, override=None):
    """--*-npz override, else config.<cfg_name>, else <repo>/data/<fname>."""
    cands = [override] if override else []
    try:
        import config
        cands.append(getattr(config, cfg_name, None))
    except Exception:
        pass
    cands.append(str(_REPO / 'data' / fname))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def _read_gold_file(path, kinds=('gold',)):
    """Deposit list -> dict(lon, lat, size, ...). Columns 'lon lat [endowment] [kind]'
    (whitespace or comma). Rows whose 4th column names a non-matching kind are dropped;
    with no 4th column all rows are kept. 'size' is log-area-scaled from the endowment
    (3rd column) if present, else uniform."""
    import numpy as np
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.replace(',', ' ').split()
            rows.append(parts)
    lon, lat, endow, keep = [], [], [], []
    kinds = tuple(k.lower() for k in kinds)
    for p in rows:
        if len(p) < 2:
            continue
        try:
            lo, la = float(p[0]), float(p[1])
        except ValueError:
            continue
        kind = p[3].lower() if len(p) >= 4 else (p[2].lower() if len(p) == 3 and not _isfloat(p[2]) else '')
        if kind and kinds and kind not in kinds:
            continue
        lon.append(lo); lat.append(la)
        endow.append(float(p[2]) if len(p) >= 3 and _isfloat(p[2]) else np.nan)
    lon = np.array(lon); lat = np.array(lat); endow = np.array(endow)
    size = _endow_to_size(endow)
    print(f"  gold track: {len(lon)} deposits from {path}")
    return dict(lon=lon, lat=lat, size=size, marker='o', facecolor=(1.0, 0.84, 0.0))


def _isfloat(s):
    try:
        float(s); return True
    except ValueError:
        return False


def _endow_to_size(endow, s_min=4.0, s_max=55.0, gamma=2.0):
    """Marker area from endowment: log-scaled, s_min..s_max; uniform s_min where unknown."""
    import numpy as np
    e = np.asarray(endow, float)
    ok = np.isfinite(e) & (e > 0)
    s = np.full(e.shape, s_min, float)
    if ok.sum() >= 2:
        le = np.log10(e[ok])
        t = (le - le.min()) / max(float(np.ptp(le)), 1e-9)
        s[ok] = s_min + (s_max - s_min) * (t ** (1.0 / gamma))
    return s


def _deposit_points(args):
    """Load gold deposit points for the --geol-track. Prefers --gold-file; else tries a
    loader in deposits.py; returns None (greenstone-only band) if neither is available."""
    kinds = ['gold']
    if getattr(args, 'deposits', None):
        kinds = [s.strip().lower() for s in args.deposits.split(',') if s.strip()] or ['gold']
    if getattr(args, 'gold_file', None):
        try:
            d = _read_gold_file(args.gold_file, kinds)
            d['style'] = getattr(args, 'gold_style', 'markers')
            return d
        except Exception as e:
            print(f"  gold track: could not read --gold-file ({e}); greenstone only")
            return None
    # read the deposit .mat directly (same files as deposits.py / the maps):
    #   <deposit-dir>/<kind>_deposits.mat, struct named <kind>, fields lon lat size
    import numpy as np
    from scipy.io import loadmat
    ddir = getattr(args, 'deposit_dir', None)
    if not ddir:                                   # default: repo matlab/ (where deposits.py looks)
        ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'matlab')
    _COLOR = {'gold': (1.00, 0.90, 0.00), 'copper': (0.85, 0.0, 0.0),
              'lithium': (0.60, 0.20, 0.80), 'nickel': (0.20, 0.70, 0.20)}
    _MARK = {'gold': 'o', 'copper': '^', 'lithium': 'h', 'nickel': 'h'}
    lons, lats, szs = [], [], []
    fc, mk = (1.0, 0.90, 0.0), 'o'
    for k in kinds:
        path = os.path.join(ddir, f'{k}_deposits.mat')
        if not os.path.isfile(path):
            print(f"  gold track: {path} not found"); continue
        try:
            g = loadmat(path, squeeze_me=True, struct_as_record=False)[k]
            lons.append(np.asarray(g.lon, float).ravel())
            lats.append(np.asarray(g.lat, float).ravel())
            szs.append(np.asarray(g.size, float).ravel())
            fc, mk = _COLOR.get(k, fc), _MARK.get(k, mk)
        except Exception as e:
            print(f"  gold track: could not read {path} ({e})")
    if not lons:
        print("  gold track: no deposit .mat loaded (set --deposit-dir or use --gold-file); "
              "greenstone only")
        return None
    lon = np.concatenate(lons); lat = np.concatenate(lats); sz = np.concatenate(szs)
    print(f"  gold track: {len(lon)} {'/'.join(kinds)} deposits from {ddir}")
    return dict(lon=lon, lat=lat, size=_endow_to_size(sz), marker=mk, facecolor=fc,
                style=getattr(args, 'gold_style', 'markers'))


def _parse_faults(specs, default_half_km=40.0):
    """--fault 'LABEL:LON,LAT[:HALF_KM]' -> list of (label, lon, lat, half_km). HALF_KM is
    the arrow half-width in km (default 40). Returns None if no faults given."""
    out = []
    for s in (specs or []):
        try:
            parts = s.split(':')
            label = parts[0].strip()
            lon, lat = (float(x) for x in parts[1].split(','))
            half = float(parts[2]) if len(parts) > 2 and parts[2].strip() else default_half_km
            out.append((label, lon, lat, half))
        except (ValueError, IndexError):
            raise SystemExit(f"--fault: bad spec {s!r}; want 'LABEL:LON,LAT[:HALF_KM]' "
                             "e.g. 'Ida Fault:121.3,-29.5' or 'Ida Fault:121.3,-29.5:60'")
    return out or None


def _parse_highlight(specs):
    """--highlight-boundary 'NPZNAME:LABEL' -> list of (npz_name, label). LABEL optional
    (defaults to the npz name). Draws that crustal boundary (from wa_crustal_boundaries.npz)
    as the red fault arrow at its true per-profile crossing, hiding its normal black arrow.
    e.g. 'Youanmi east:Ida Fault' (the Youanmi/Eastern-Goldfields terrane margin = Ida Fault)."""
    out = []
    for s in (specs or []):
        name, _, lab = s.partition(':')
        name = name.strip(); lab = lab.strip() or name
        if name:
            out.append((name, lab))
    return out or None


def _load_geology(spec, litho_npz=None, tect_npz=None, bnd_npz=None, lines=False,
                  domain_colours='map2022', greenstone_npz=None, deposits=None,
                  dep_swath=50.0, craton_outline=None, faults=None, highlight_boundaries=None,
                  vpvs_track=None, vpvs_lim=(1.68, 1.86), vpvs_max_dist=0.8,
                  vpvs_cmap='gmt_polar_r', vpvs_ncolors=16):
    import xsection_strips as xs
    want = {w.strip() for w in spec.split(',') if w.strip()}
    bad = want - _STRIP_ITEMS
    if bad:
        raise SystemExit(f"--strips: unknown item(s) {sorted(bad)}; "
                         f"choose from {sorted(_STRIP_ITEMS)}")
    g = dict(litho=None, tect=None, bnd=None, names='names' in want, lines=lines)
    jobs = (('litho', 'litho', 'LITHOLOGY_NPZ', 'wa_lithology.npz', litho_npz,
             xs.PolygonIndex.from_npz),
            ('domain', 'tect', 'TECTONICS_NPZ', 'wa_tectonics.npz', tect_npz,
             lambda p: xs.PolygonIndex.from_npz(p, class_key='names', colours=domain_colours)),
            ('boundaries', 'bnd', 'BOUNDARIES_NPZ', 'wa_crustal_boundaries.npz',
             bnd_npz, xs.BoundarySet.from_npz))
    for item, key, cfg, fname, ov, loader in jobs:
        if item not in want:
            continue
        path = _resolve_npz(cfg, fname, ov)
        if path is None:
            print(f"  --strips {item}: {fname} not found (config.{cfg} or data/) - skipped")
            continue
        g[key] = loader(path)
        print(f"  strips: {item} <- {path}")
    g['greenstone'] = None
    g['deposits'] = deposits
    g['dep_swath'] = dep_swath
    g['faults'] = faults
    g['highlight_boundaries'] = highlight_boundaries
    g['craton'] = None
    if craton_outline and os.path.isfile(craton_outline):
        try:
            g['craton'] = xs.craton_index_from_outline(craton_outline)
            print(f"  strips: craton line <- {craton_outline}")
        except Exception as e:
            print(f"  craton line load failed: {e}")
    elif craton_outline:
        print(f"  craton outline not found ({craton_outline}) - line skipped")
    if greenstone_npz and os.path.isfile(greenstone_npz):
        try:
            g['greenstone'] = xs.greenstone_index_from_npz(greenstone_npz)
            print(f"  strips: greenstone <- {greenstone_npz}")
        except Exception as e:
            print(f"  greenstone track load failed: {e}")
    elif greenstone_npz:
        print(f"  greenstone npz not found ({greenstone_npz}) - band skipped")
    g['vpvs'] = None
    if vpvs_track and os.path.isfile(vpvs_track):
        try:
            from plot_wa_map import _read_hk
            vlon, vlat, vval = _read_hk(vpvs_track, 'k_new')
            g['vpvs'] = dict(lon=vlon, lat=vlat, val=vval, clim=tuple(vpvs_lim),
                             max_dist=vpvs_max_dist, label='Vp/Vs',
                             cmap=_cmap(vpvs_cmap, max(int(vpvs_ncolors), 2)))
            print(f"  strips: Vp/Vs track <- {vpvs_track} ({vlon.size} stations, "
                  f"clim {vpvs_lim[0]:.2f}-{vpvs_lim[1]:.2f}, cmap {vpvs_cmap} x{vpvs_ncolors})")
        except Exception as e:
            print(f"  Vp/Vs track load failed: {e}")
    elif vpvs_track:
        print(f"  Vp/Vs track table not found ({vpvs_track}) - strip skipped")
    return g


def _strip_height_in(g):
    """Vertical room the strips (and rotated boundary names) need above a panel."""
    if g is None:
        return 0.0
    h = 0.17 * ((g['litho'] is not None) + (g['tect'] is not None))
    if g.get('greenstone') is not None or g.get('deposits') is not None:
        h += 0.20                                  # greenstone + gold track
    if g.get('vpvs') is not None:
        h += 0.20                                  # Vp/Vs colour bar
    if g['bnd'] is not None:
        h += 0.28                                  # boundary down-arrows
        if g['names']:
            h += 0.55
    return h


def _add_geology(top_ax, axes, plon, plat, dist, names=None):
    if _GEO is None:
        return None
    import xsection_strips as xs
    res = xs.add_profile_strips(top_ax, axes, plon, plat, dist,
                                litho=_GEO['litho'], tectonic=_GEO['tect'], boundaries=_GEO['bnd'],
                                label_boundaries=_GEO['names'] if names is None else names,
                                boundary_lines=_GEO.get('lines', False),
                                greenstone=_GEO.get('greenstone'), deposits=_GEO.get('deposits'),
                                dep_swath_km=_GEO.get('dep_swath', 50.0),
                                craton=_GEO.get('craton'), faults=_GEO.get('faults'),
                                highlight_boundaries=_GEO.get('highlight_boundaries'),
                                vpvs=_GEO.get('vpvs'))
    if res.crossings and names is not False:
        print('  boundary crossings: ' + ', '.join(
            f"{c.dist:.0f} km" + (f" ({'/'.join(c.names)})" if c.names else '')
            for c in res.crossings))
    return res


def _geology_legend(fig, results, anchor_ax, ncol=4, below_in=0.62):
    """Lithology legend a fixed distance below anchor_ax (clear of tick labels and
    xlabel); only classes seen on these sections."""
    import xsection_strips as xs
    from matplotlib.transforms import offset_copy
    runs = [r.runs['litho'] for r in results if r is not None and 'litho' in r.runs]
    if not runs:
        return
    h = xs.legend_handles(runs)
    tr = offset_copy(anchor_ax.transAxes, fig=fig, y=-below_in, units='inches')
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.0),
               bbox_transform=tr,
               ncol=min(ncol, len(h)), fontsize=7, frameon=False,
               title='Surface lithology - GSWA 1:500k (CC-BY-4.0)', title_fontsize=7)

# ── Great circle path ─────────────────────────────────────────────────────────
def _great_circle_path(lat1, lon1, lat2, lon2, ds_deg=0.08):
    """
    Sample a great circle from (lat1,lon1) to (lat2,lon2) at ~ds_deg spacing.
    Returns arrays of (lat, lon, dist_km).
    """
    from math import radians, degrees, sin, cos, atan2, acos, sqrt
    R = 6371.0  # km
    φ1, λ1 = radians(lat1), radians(lon1)
    φ2, λ2 = radians(lat2), radians(lon2)

    # Total angular distance
    Δφ = φ2 - φ1; Δλ = λ2 - λ1
    a = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    d_total = 2 * R * atan2(sqrt(a), sqrt(1-a))

    n = max(2, int(round(d_total / (ds_deg * 111.195))))
    lats = np.zeros(n); lons = np.zeros(n); dists = np.zeros(n)

    for i, f in enumerate(np.linspace(0, 1, n)):
        # Interpolate on great circle
        A = sin((1-f)*2*atan2(sqrt(a),sqrt(1-a))) / sin(2*atan2(sqrt(a),sqrt(1-a))) \
            if a < 1 else 1-f
        B = sin(f*2*atan2(sqrt(a),sqrt(1-a))) / sin(2*atan2(sqrt(a),sqrt(1-a))) \
            if a < 1 else f
        x = A*cos(φ1)*cos(λ1) + B*cos(φ2)*cos(λ2)
        y = A*cos(φ1)*sin(λ1) + B*cos(φ2)*sin(λ2)
        z = A*sin(φ1)         + B*sin(φ2)
        lats[i] = degrees(atan2(z, sqrt(x**2+y**2)))
        lons[i] = degrees(atan2(y, x))
        dists[i] = f * d_total

    return lats, lons, dists


def _polyline_path(lons, lats, ds_deg=0.08):
    """(lat, lon, dist_km) along a multi-vertex polyline: great-circle segments between
    consecutive vertices, joined. Falls back to a straight line for 2 vertices."""
    lons = np.asarray(lons, float); lats = np.asarray(lats, float)
    if len(lons) < 2:
        return lats, lons, np.zeros(len(lons))
    PL, PO, DS, base = [], [], [], 0.0
    for i in range(len(lons) - 1):
        la, lo, ds = _great_circle_path(lats[i], lons[i], lats[i + 1], lons[i + 1], ds_deg)
        if i > 0:
            la, lo, ds = la[1:], lo[1:], ds[1:]
        PL.append(la); PO.append(lo); DS.append(ds + base)
        base = DS[-1][-1] if len(DS[-1]) else base
    return np.concatenate(PL), np.concatenate(PO), np.concatenate(DS)


# ── Profile interpolation ─────────────────────────────────────────────────────
_RBF_CACHE = []        # [(data3d, Lon, Lat, RBFInterpolator)] - same arrays, same fit
MAX_NODE_DIST = 0.5   # deg: profile points further from any model node are left blank
                      # (as the depth-slice maps; stops the fit extrapolating). 0 = off


def _sample_profile(Lon, Lat, data3d, prof_lon, prof_lat):
    """
    Interpolate data3d[npts, nz] onto profile points [n_prof].
    Builds the horizontal interpolator ONCE and applies to all depth columns at
    once (RBFInterpolator supports vector-valued data) — ~nz times faster than
    rebuilding per depth. Falls back to per-depth griddata only if needed.
    Returns [n_prof, nz].
    """
    from scipy.interpolate import RBFInterpolator
    nprof = len(prof_lon); nz = data3d.shape[1]
    pts  = np.column_stack([Lon, Lat])
    qpts = np.column_stack([prof_lon, prof_lat])
    out  = np.full((nprof, nz), np.nan)

    # Rows (nodes) finite at ALL depths → use them to fit one interpolator for
    # every depth at once. Most nodes are finite throughout; the few with NaNs
    # at some depths are handled by the per-depth fallback below.
    row_all_finite = np.isfinite(data3d).all(axis=1)
    if row_all_finite.sum() >= 10:
        try:
            # the fit depends only on the model, not the profile: build it once per
            # array and reuse it for every section (the dense solve is the slow part)
            rbf = None
            for arr, lo, la, f in _RBF_CACHE:
                if arr is data3d and lo is Lon and la is Lat:
                    rbf = f
            if rbf is None:
                rbf = RBFInterpolator(pts[row_all_finite], data3d[row_all_finite],
                                      kernel='thin_plate_spline', smoothing=0.01)
                _RBF_CACHE.append((data3d, Lon, Lat, rbf))
                del _RBF_CACHE[:-4]
            out[:, :] = rbf(qpts)        # evaluates all nz columns in one call
            # Depths where some needed node was NaN: those columns may still be
            # fine (we used all-finite rows); only redo columns that are all-nan.
            bad_cols = ~np.isfinite(out).any(axis=0)
            if not bad_cols.any():
                return _mask_far(out, Lon, Lat, data3d, prof_lon, prof_lat)
        except Exception:
            pass

    # Fallback: per-depth (only for columns not filled above)
    return _mask_far(_sample_fallback(Lon, Lat, data3d, pts, qpts, out, nz), Lon, Lat, data3d,
                     prof_lon, prof_lat)


def _mask_far(out, Lon, Lat, data3d, prof_lon, prof_lat):
    """Blank profile points more than MAX_NODE_DIST degrees from every node that has data."""
    if not MAX_NODE_DIST:
        return out
    from scipy.spatial import cKDTree
    has = np.isfinite(data3d).any(axis=1)
    # plain degrees, exactly like the depth-slice maps (plot_depth_slice max_dist)
    tree = cKDTree(np.column_stack([np.asarray(Lon)[has], np.asarray(Lat)[has]]))
    dd, _ = tree.query(np.column_stack([np.asarray(prof_lon), np.asarray(prof_lat)]))
    out[dd > MAX_NODE_DIST, :] = np.nan
    return out


def _sample_fallback(Lon, Lat, data3d, pts, qpts, out, nz):
    from scipy.interpolate import griddata, RBFInterpolator
    todo = range(nz) if not np.isfinite(out).any() else np.where(~np.isfinite(out).any(axis=0))[0]
    for iz in todo:
        col = data3d[:, iz]; fin = np.isfinite(col)
        if fin.sum() < 4: continue
        try:
            out[:, iz] = RBFInterpolator(pts[fin], col[fin],
                kernel='thin_plate_spline', smoothing=0.01)(qpts)
        except Exception:
            out[:, iz] = griddata(pts[fin], col[fin], qpts, method='linear')
    return out


def _sample_2d(Lon, Lat, data2d, prof_lon, prof_lat):
    """Interpolate 2D field (no depth) onto profile."""
    from scipy.interpolate import RBFInterpolator
    pts = np.column_stack([Lon, Lat])
    qpts = np.column_stack([prof_lon, prof_lat])
    fin = np.isfinite(data2d)
    if fin.sum() < 4:
        return np.full(len(prof_lon), np.nan)
    try:
        return RBFInterpolator(pts[fin], data2d[fin],
                               kernel='thin_plate_spline', smoothing=0.01)(qpts)
    except Exception:
        from scipy.interpolate import griddata
        return griddata(pts[fin], data2d[fin], qpts, method='linear')


# Colour convention (locked): absolute velocity = Spectral (warm = slow);
# xi diverging around 1.0 and all differences = RdBu (red = low/slow).
CMAP_ABS = 'cmapx'        # absolute Vsv: HY's MATLAB velocity map (matches the depth maps)
CMAP_VPVS = 'rgb'         # Vp/Vs absolute: red (slow) -> green -> blue (fast), as on the Vp/Vs map

# cmapx: HY's MATLAB absolute-velocity map (red -> orange -> yellow -> green -> cyan ->
# blue, warm = slow), from /data/25/yuan/mfiles/mat/cmapx.mat. Anchors as printed in MATLAB.
CMAPX_COLORS = [
    (0.8204, 0.0039, 0.0039), (1.0000, 0.0039, 0.0039), (1.0000, 0.2926, 0.0039),
    (1.0000, 0.5813, 0.0039), (1.0000, 0.7222, 0.0039), (0.9000, 0.9000, 0.0039),
    (0.0039, 0.9900, 0.0000), (0.0033, 0.7546, 1.0000), (0.0039, 0.5813, 1.0000),
    (0.0039, 0.3813, 1.0000), (0.0039, 0.0039, 1.0000), (0.0039, 0.0039, 0.8204)]


def cmapx(n=None):
    """HY's cmapx absolute-velocity colormap. n None/<=0 -> smooth (interpolated);
    n>0 -> n discrete steps resampled from the smooth ramp."""
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list('cmapx', CMAPX_COLORS, N=256)
    return cm.resampled(n) if n and n > 0 else cm

# ── Section scale (shared by per-section and --stack; plot_stack's math) ─────
PAGE_W_IN = 9.5     # longest section spans this many inches
DEFAULT_VE = 3.0   # approved flat-strip look; per-section and --stack share it


def _section_scale(maxd, zbot, ve=DEFAULT_VE, page_w_in=PAGE_W_IN):
    """km_per_in = longest/page_w_in ; panel height = depth*VE/km_per_in.
    Each section's width = its length/km_per_in. The panel box sets the drawn
    aspect (no set_aspect anywhere), so every section in both modes shares the
    same horizontal scale, the same height and the same VE."""
    km_per_in = maxd / page_w_in
    return km_per_in, zbot * ve / km_per_in


# ── Colormap helper ───────────────────────────────────────────────────────────
# Rule: RELATIVE / centred values (dln fields, xi around 1.0) -> RdBu with a narrow
# neutral (white) centre and a gentle gap; ABSOLUTE values (Vsv, Viso) -> plain,
# continuous Spectral. Gap/neutral settings below apply to the centred maps only; DIV_GAP = half-width of the skipped band
# (0 = classic RdBu through white). Set from --div-gap.
DIV_GAP = 0.20
DIVERGING = ('RdBu', 'RdYlBu')       # centred maps only; Spectral (absolute) stays continuous


ABS_SKIP = None     # (LO, HI) part of Spectral left out of absolute maps, e.g. (0.36, 0.70); --abs-skip
DIV_WHITE = 0.5     # neutral (white) band width in colour steps (0.5 = half a step); --div-white
RWB_GAP = 0.12      # rwb (xi): skip the palest +-gap of red-white-blue so the white centre is tight
                    # (fewer near-white bands); 0 = full-width white like a plain red-white-blue ramp


def _nice_ticks(vmin, vmax, n=8):
    """Round 'nice' tick values spanning [vmin,vmax] (steps of 1/2/5 x10^k), plus the
    decimal count to label them cleanly. Same as plot_wa_map.py so bars match the maps."""
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


def _apply_nice_ticks(cb, vmin, vmax, n=8):
    """Snap a colorbar to round tick values labelled to a consistent decimal count."""
    ticks, dec = _nice_ticks(vmin, vmax, n)
    if ticks:
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{t:.{dec}f}' for t in ticks])


def _ncd(ncolors, smooth):
    """Colour-level count for building a panel's colormap: a near-continuous 256 when
    --field-style smooth (MATLAB-style continuous shading), else the discrete band count."""
    return 256 if smooth else (ncolors or 16)


def _cmap(name, n=32):
    """Diverging maps: n colour levels split evenly either side of a neutral (white) band of
    DIV_WHITE levels - fractional allowed (0.5 = one white step half as wide as a colour step;
    1 = exactly one white step); DIV_GAP skips the palest +-gap of the base map.
    'cmapx'/'rwb'/'rgb' are the map colormaps (matches plot_wa_map.py) so absolute cross-section
    panels read like the maps: Vsv=cmapx, xi=rwb (red<1<blue), Vp/Vs=rgb (slow red -> fast blue)."""
    if name == 'cmapx':
        return cmapx(n)
    if name == 'rwb':
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        cm = LinearSegmentedColormap.from_list('rwb', [(0.78, 0, 0), (1, 1, 1), (0, 0, 0.78)], N=256)
        if not (n and n > 0):
            return cm
        gap = RWB_GAP                       # drop the palest +-gap so the centre white is tight
        side = n // 2
        lo = cm(np.linspace(0.0, 0.5 - gap, side))
        hi = cm(np.linspace(0.5 + gap, 1.0, side))
        cols = np.vstack([lo, hi]) if n % 2 == 0 else np.vstack([lo, np.asarray(cm(0.5))[None, :], hi])
        return ListedColormap(cols, name='rwb_gap')
    if name == 'rgb':
        from matplotlib.colors import LinearSegmentedColormap
        anchors = [(0.63, 0.09, 0.16), (0.88, 0.40, 0.22), (0.62, 0.70, 0.35),
                   (0.38, 0.70, 0.40), (0.26, 0.62, 0.62), (0.26, 0.42, 0.72),
                   (0.30, 0.22, 0.60)]
        cm = LinearSegmentedColormap.from_list('rgb', anchors, N=256)
        return cm.resampled(n) if n and n > 0 else cm
    if name in ('gmt_polar', 'gmt_polar_r'):        # GMT polar.cpt: blue -> white -> red
        from matplotlib.colors import LinearSegmentedColormap
        cols = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        if name.endswith('_r'):
            cols = cols[::-1]
        cm = LinearSegmentedColormap.from_list(name, cols, N=256)
        return cm.resampled(n) if n and n > 0 else cm
    if name in DIVERGING and (DIV_GAP > 0 or DIV_WHITE > 0):
        from matplotlib.colors import ListedColormap
        base = matplotlib.colormaps[name]
        w = max(0.0, min(float(DIV_WHITE), n - 2.0))
        side = max(1, int((n - np.floor(w)) // 2))
        f = 1 if abs(w - round(w)) < 1e-6 else 2            # sub-steps per colour level
        kw = int(round(w * f))
        lo = np.repeat(base(np.linspace(0.0, 0.5 - DIV_GAP, side)), f, axis=0)
        hi = np.repeat(base(np.linspace(0.5 + DIV_GAP, 1.0, side)), f, axis=0)
        mid = np.repeat(np.asarray(base(0.5))[None, :], kw, axis=0)
        return ListedColormap(np.vstack([lo, mid, hi]), name=f'{name}_gap')
    if name == 'Spectral' and ABS_SKIP:                    # --abs-skip LO HI: drop the pale middle
        from matplotlib.colors import ListedColormap
        base = matplotlib.colormaps[name]
        lo, hi = ABS_SKIP
        na = int(round(n * lo / (lo + 1.0 - hi))); nb = n - na
        return ListedColormap(np.vstack([base(np.linspace(0.0, lo, na)), base(np.linspace(hi, 1.0, nb))]),
                              name='Spectral_skip')
    try:
        return matplotlib.colormaps[name].resampled(n)
    except Exception:
        return plt.cm.get_cmap(name, n)


# ── xi reference for dlnXi ────────────────────────────────────────────────────
_REF_NODES = None     # --ref-domain: bool over model nodes (e.g. the Yilgarn); None = all nodes
_REF_NAME = None      # its label, e.g. 'Yilgarn Craton'
_REF_TAG = ''         # added to output names, e.g. '.ref-yilgarn'


def _node_ref(A):
    """dln reference at each depth: mean over the --ref-domain nodes, else all nodes."""
    A = np.asarray(A, float)
    return np.nanmean(A[_REF_NODES] if _REF_NODES is not None else A, axis=0)


def _ref_label():
    return f'{_REF_NAME} mean at each depth' if _REF_NAME else 'model mean at each depth'


def _xi_ref(Xi, mode='depth'):
    """Reference xi for dlnXi = (xi/ref - 1)*100, from ALL model nodes (or the
    --ref-domain nodes).
    depth : mean xi of the whole model at each depth (like dVsv) - removes the
            depth trend, shows lateral variation;
    global: one mean over the whole model, all depths - note dlnXi is then just
            xi linearly rescaled (same picture as the xi panel, re-centred)."""
    src = Xi[_REF_NODES] if _REF_NODES is not None else Xi
    if mode == 'global':
        return np.full(Xi.shape[1], np.nanmean(src))
    return np.nanmean(src, axis=0)


def _xi_ref_label(mode):
    if mode == 'global':
        return f'{_REF_NAME} mean' if _REF_NAME else 'whole-model mean'
    return _ref_label()


# ── Fvs loading (optionally converting Viso -> true Vsv) ──────────────────────
XI_SQ = False                 # --xi-squared: show conventional xi = (Vsh/Vsv)^2 = Xi^2
XI_LIM = (0.92, 1.08)
XI_TITLE = 'xi = Vsh/Vsv (blue = Vsh>Vsv)'


def _load_fvs(path, vel='model'):
    """Fvs npz as a dict. vel='vsv_true' replaces the 'Vsv' array of an xi (ZT) model
    (which holds Viso, Voigt / setup=3) by true Vsv = Viso/sqrt((2+Xi^2)/3), xi = Vsh/Vsv.
    Vsv_err stays the Viso posterior error (a close proxy)."""
    raw = np.load(path, allow_pickle=True)
    d = {k: raw[k] for k in raw.files}
    if vel == 'vsv_true':
        if 'Xi' not in d or np.nanstd(d['Xi']) <= 1e-4:
            sys.exit(f"--vel vsv_true needs an xi (ZT) model; in {os.path.basename(path)} "
                     f"the 'Vsv' array already is Vsv")
        d['Vsv'] = d['Vsv'] / np.sqrt((2.0 + d['Xi'] ** 2) / 3.0)
        d['_vel'] = 'vsv_true'
    if XI_SQ and 'Xi' in d:                  # after the Vsv conversion, which needs linear Xi
        d['Xi'] = np.asarray(d['Xi'], float) ** 2
    return d


# ── Reference model for model-vs-model comparison (--ref-fvs) ─────────────────
_REF = None


def _load_ref(path, vel='model', label=None, d_max=None):
    """Reference Vs model, sampled at its OWN nodes along each section (so the two
    models need not share nodes or depth grids)."""
    r = _load_fvs(path, vel)
    if vel == 'model' and _varies(r, 'Xi'):
        print(f"  NOTE reference {os.path.basename(path)} is an xi model: its 'Vsv' is Viso. "
              f"For Vsv-vs-Vsv use --ref-vel vsv_true.")
    z = np.asarray(r['z'], float)
    zm = z <= d_max if d_max else np.ones(len(z), bool)
    if label is None:
        import re
        m = re.search(r'iter\.?(\d+)', os.path.basename(path))
        label = f'iter{m.group(1)}' if m else os.path.basename(path).replace('.npz', '')
    print(f"  reference: {label} <- {path} ({len(r['Lon'])} nodes)")
    return dict(Lon=r['Lon'], Lat=r['Lat'], z=z[zm], V=r['Vsv'][:, zm], label=label)


def _ref_diff(v_model, plon, plat, z):
    """(model / ref - 1) * 100 along a profile; ref interpolated onto the model's z."""
    ref = _sample_profile(_REF['Lon'], _REF['Lat'], _REF['V'], plon, plat)
    if len(_REF['z']) != len(z) or not np.allclose(_REF['z'], z):
        ref = np.array([np.interp(z, _REF['z'], row, left=np.nan, right=np.nan) for row in ref])
    return (v_model / ref - 1.0) * 100.0


def _vname(d, is_zt):
    return 'Vsv' if d.get('_vel') == 'vsv_true' else ('Viso' if is_zt else 'Vsv')


def _varies(d, key):
    return key in d and np.nanstd(d[key]) > 1e-4


# ── Moho overlay: dashed line + optional grey mask below it ───────────────────
def _draw_moho(ax, dist, moho, zbot, mask_alpha=0.5, lw=1.1):
    """Dashed Moho; with mask_alpha (0-1) a grey veil from the Moho down to zbot
    (the bottom of the plotted depth range), so the crust stands out."""
    moho = np.asarray(moho, float)
    if mask_alpha:
        ok = np.isfinite(moho)
        ax.fill_between(dist, np.where(ok, np.minimum(moho, zbot), zbot), zbot,
                        where=ok, color='0.5', alpha=mask_alpha, lw=0, zorder=2)
    ax.plot(dist, moho, 'k--', lw=lw, alpha=0.7, zorder=3)


def _load_moho_interp(path, apply_mask=True):
    """Return f(lat, lon) -> Moho depth (km) from either:
      - an AusMoho-style gridded text file (lat lon depth, 11 header rows, regular grid)
        -> RectBivariateSpline (continuous everywhere); or
      - a CCP refined pick .mat (moho_refined: lat2/lon2/z_moho[/stn_mask], the most recent
        CCP Moho) -> LinearNDInterpolator over its finite nodes, NaN outside the CCP
        footprint so the overlaid line stops where the CCP data stop.
    Same .mat convention as plot_wa_map.py so the sections match the maps."""
    if str(path).lower().endswith('.mat'):
        from scipy.io import loadmat
        from scipy.interpolate import LinearNDInterpolator
        m = loadmat(path, squeeze_me=True, struct_as_record=False)['moho_refined']
        LON = np.asarray(m.lon2, float).ravel()
        LAT = np.asarray(m.lat2, float).ravel()
        Z = np.asarray(m.z_moho, float).ravel()
        if apply_mask and hasattr(m, 'stn_mask'):
            Z = np.where(np.asarray(m.stn_mask).astype(bool).ravel(), Z, np.nan)
        ok = np.isfinite(LON) & np.isfinite(LAT) & np.isfinite(Z)
        interp = LinearNDInterpolator(np.column_stack([LON[ok], LAT[ok]]), Z[ok])
        return lambda la, lo: interp(lo, la)          # x=lon, y=lat; NaN outside the hull
    from scipy.interpolate import RectBivariateSpline
    md = np.loadtxt(path, skiprows=11)
    mla = np.unique(md[:, 0]); mlo = np.unique(md[:, 1])
    spl = RectBivariateSpline(mla, mlo, md[:, 2].reshape(len(mla), len(mlo)), kx=1, ky=1)
    return lambda la, lo: spl(la, lo)


def _set_ccp_stack(args):
    """Load the CCP RF cube once (args.ccp_pmat) and park it, with its display settings,
    on plot_section so every section (serial or worker) samples the same volume. Also
    derives the pvh.N tag from the P.mat filename for the panel title."""
    import re
    plot_section._ccp = None
    plot_section._ccp_clim = args.ccp_clim
    plot_section._ccp_neutral = args.ccp_neutral
    plot_section._ccp_method = args.ccp_method
    plot_section._ccp_cmap = getattr(args, 'ccp_cmap', None)
    plot_section._ccp_tag = ''
    if not getattr(args, 'ccp_pmat', None):
        return
    try:
        import plot_ccp_profiles as pcp
        plot_section._ccp = pcp.load_ccp(args.ccp_pmat, ofile=args.ccp_cmp, amp_var=args.ccp_amp_var)
        m = re.search(r'pvh\.?(\d+)', os.path.basename(args.ccp_pmat))
        plot_section._ccp_tag = m.group(1) if m else ''
        print(f'  CCP RF stack loaded: {args.ccp_pmat}'
              + (f' (pvh.{plot_section._ccp_tag})' if plot_section._ccp_tag else ''))
    except Exception as e:
        print(f'  CCP RF stack load failed: {e}')
        plot_section._ccp = None


# ── Single cross-section ──────────────────────────────────────────────────────
def _nan_gauss(a, sigma):
    """Gaussian smooth of a 2-D array that ignores NaNs (normalised convolution). sigma in
    grid cells; the original NaN mask is preserved so masked regions stay blank."""
    from scipy.ndimage import gaussian_filter
    a = np.asarray(a, float)
    m = np.isfinite(a)
    if not sigma or sigma <= 0 or not m.any():
        return a
    num = gaussian_filter(np.where(m, a, 0.0), sigma, mode='nearest')
    den = gaussian_filter(m.astype(float), sigma, mode='nearest')
    out = np.where(den > 1e-6, num / den, np.nan)
    out[~m] = np.nan
    return out


def plot_section(d, lat1, lon1, lat2, lon2, label,
                 ds_deg=0.08, ncolors=16, d_max=None,
                 sigma_vsv=1.5, clim_rel=6.0,
                 out_dir='figures/xsections',
                 max_dist=None, km_per_in=None, panel_h_in=None, ref_mean=None,
                 long_ratio=3.0, moho_mask=0.5, xi_panel='xi', xi_ref='depth',
                 clim_xi=5.0, div_gap=None, div_white=None, clim_diff=4.0, vlim=None,
                 panels=None, var_ref='model', allprof_vsv_ref=None, allprof_xi_ref=None,
                 smooth=False, geol_tag='', vs_smooth=0.0, xi_gap=None, xi_cmap='rwb',
                 dvsv_cmap='RdBu', xi2_d=None, xi2_label='model 2', xi_label='',
                 xi_diff_lim=0.05):
    global DIV_GAP, DIV_WHITE, RWB_GAP
    if div_gap is not None:
        DIV_GAP = div_gap
    if div_white is not None:
        DIV_WHITE = div_white
    if xi_gap is not None:
        RWB_GAP = xi_gap           # xi white-centre width: 0 = smooth red->white->blue
    """One cross-section. Panels: Vsv/Viso | dlnVsv | [xi] | uncertainty (no misfit).
    Scale is set by the LONGEST section (drawn 2:1 per panel, full landscape width);
    km_per_in and panel_h_in are computed once from it and passed to every section so
    all panels share the SAME height and horizontal scale. Shorter sections → narrower
    boxes (width ∝ length), each filling its own frame. ref_mean: full-model mean Vsv."""
    Lon = d['Lon']; Lat = d['Lat']; z = d['z']
    if d_max:
        zm = z <= d_max; z = z[zm]
    else:
        zm = np.ones(len(z), dtype=bool)

    Vsv    = d['Vsv'][:, zm]
    Vsv_err= d['Vsv_err'][:, zm] if 'Vsv_err' in d else None
    Xi     = d['Xi'][:, zm] if 'Xi' in d else None
    Vpvs   = d['Vpvs'][:, zm] if _varies(d, 'Vpvs') else None   # only if actually inverted
    moho_prof = None
    moho_ccp_prof = None
    _mi = getattr(plot_section, '_moho_interp', None)
    _mc = getattr(plot_section, '_moho_ccp_interp', None)

    print(f"  Profile {label}: ({lat1:.2f},{lon1:.2f}) -> ({lat2:.2f},{lon2:.2f})")
    plat, plon, dist = _great_circle_path(lat1, lon1, lat2, lon2, ds_deg)
    print(f"  Length: {dist[-1]:.0f} km  |  {len(dist)} sample points")
    if _mi is not None:
        moho_prof = np.array([float(np.ravel(_mi(la, lo))[0]) for la, lo in zip(plat, plon)])
    if _mc is not None:
        moho_ccp_prof = np.array([float(np.ravel(_mc(la, lo))[0]) for la, lo in zip(plat, plon)])

    print("  Sampling ...")
    vsv_p = _sample_profile(Lon, Lat, Vsv, plon, plat)
    if vs_smooth and vs_smooth > 0:               # gentle NaN-aware smooth of the Vs field only
        vsv_p = _nan_gauss(vsv_p, vs_smooth)
    err_p = _sample_profile(Lon, Lat, Vsv_err, plon, plat) if Vsv_err is not None else None
    xi_p  = _sample_profile(Lon, Lat, Xi, plon, plat) if Xi is not None else None
    vpvs_p = _sample_profile(Lon, Lat, Vpvs, plon, plat) if Vpvs is not None else None
    xi2_p = None                                 # second model's xi (comparison panel)
    if xi2_d is not None and xi2_d.get('Xi') is not None:
        z2 = np.asarray(xi2_d['z'], float)
        _x2 = _sample_profile(xi2_d['Lon'], xi2_d['Lat'], np.asarray(xi2_d['Xi'], float), plon, plat)
        xi2_p = np.vstack([np.interp(z, z2, _x2[i], left=np.nan, right=np.nan)  # onto main z grid
                           for i in range(_x2.shape[0])])
        _f2 = xi2_p[np.isfinite(xi2_p)]
        print(f"  xi2 panel [{xi2_label}]: {_f2.size} finite cells, "
              + (f"xi {_f2.min():.3f}-{_f2.max():.3f} (std {_f2.std():.4f})" if _f2.size else "ALL NaN"))
    xidiff_p = (xi_p - xi2_p) if (xi_p is not None and xi2_p is not None) else None
    if xidiff_p is not None:
        _fd = xidiff_p[np.isfinite(xidiff_p)]
        if _fd.size:
            print(f"  xi-diff ({xi_label or 'model1'} - {xi2_label}): "
                  f"{-np.abs(_fd).max():.3f}..{np.abs(_fd).max():.3f} (rms {np.sqrt((_fd**2).mean()):.4f})")

    # CCP RF amplitude along the profile, interpolated onto the model depth grid z so it
    # shares the D,Z mesh (and _pcolor's contourf + Moho overlays + bar) with the other panels.
    ccp_p = None
    _ccp = getattr(plot_section, '_ccp', None)
    if _ccp is not None:
        import plot_ccp_profiles as pcp
        amp = pcp.sample_ccp(_ccp, plon, plat,
                             method=getattr(plot_section, '_ccp_method', 'linear'))  # [nq, nz_ccp]
        zg = _ccp['zg']
        ccp_p = np.full((amp.shape[0], len(z)), np.nan)
        for j in range(amp.shape[0]):
            col = amp[j]; ok = np.isfinite(col)
            if ok.sum() >= 2:                        # in-coverage column -> onto model z (NaN outside)
                ccp_p[j] = np.interp(z, zg[ok], col[ok], left=np.nan, right=np.nan)

    # dln reference baseline (--var-ref): model/yilgarn = model-node mean (passed in via
    # ref_mean, yilgarn when _REF_NODES set); profile = this profile's own along-line mean;
    # allprofiles = mean over every profile in the location file (precomputed in main).
    if var_ref == 'profile':
        ref_mean = np.nanmean(vsv_p, axis=0)
    elif var_ref == 'allprofiles' and allprof_vsv_ref is not None:
        ref_mean = np.asarray(allprof_vsv_ref, float)
    elif ref_mean is None:
        ref_mean = _node_ref(Vsv)
    dln_p = (vsv_p - ref_mean[None, :]) / ref_mean[None, :] * 100.0
    _ref_lbl = {'profile': 'profile mean at each depth',
                'allprofiles': 'all-profiles mean at each depth'}.get(var_ref, _ref_label())

    D, Z = np.meshgrid(dist, z, indexing='ij')

    fin_v = vsv_p[np.isfinite(vsv_p)]
    vmed  = np.nanmedian(fin_v); vstd = np.nanstd(fin_v)
    vmin_v = vmed - sigma_vsv*vstd; vmax_v = vmed + sigma_vsv*vstd
    if vlim is not None:                       # same scale across compared models
        vmin_v, vmax_v = vlim
    vdiff_p = _ref_diff(vsv_p, plon, plat, z) if _REF is not None else None

    have_xi  = xi_p  is not None and np.nanstd(xi_p) > 1e-4
    have_err = err_p is not None
    is_zt = 'ZT' in os.path.basename(getattr(plot_section, '_fvsname', '')) or have_xi
    vname = _vname(d, is_zt)

    dxi_p = None
    _xref_lbl = _xi_ref_label(xi_ref)
    if have_xi and xi_panel in ('dxi', 'both'):
        if var_ref == 'profile':
            xref = np.nanmean(xi_p, axis=0); _xref_lbl = 'profile mean at each depth'
        elif var_ref == 'allprofiles' and allprof_xi_ref is not None:
            xref = np.asarray(allprof_xi_ref, float); _xref_lbl = 'all-profiles mean at each depth'
        else:
            xref = _xi_ref(Xi, xi_ref)
        dxi_p = (xi_p / xref[None, :] - 1.0) * 100.0
    rows = ['abs', 'rel']
    if vdiff_p is not None: rows.append('vdiff')
    if have_xi and xi_panel in ('xi', 'both'): rows.append('xi')
    if xi2_p is not None and np.isfinite(xi2_p).any():   # explicitly requested -> always show
        rows.append('xi2')
    if xidiff_p is not None and np.isfinite(xidiff_p).any(): rows.append('xidiff')
    if dxi_p is not None: rows.append('dxi')
    if vpvs_p is not None: rows.append('vpvs')
    if ccp_p is not None: rows.append('ccp')
    if have_err: rows.append('err')
    if panels:                               # keep only the requested panels, in that order
        _alias = {'vsv': 'abs', 'dvsv': 'rel'}
        want = [_alias.get(p.strip().lower(), p.strip().lower()) for p in panels]
        sel = [r for r in want if r in rows]
        if sel:
            rows = sel
        else:
            print(f"  --panels {panels}: none available here ({rows}); drawing all")
    nrow = len(rows)

    xmax = dist[-1]; zbot = z[-1]
    # Scale set by the longest section (full landscape width @ 2:1 per panel).
    # If not supplied, derive from THIS section (standalone use).
    # A4 landscape usable panel width (A4 = 11.69 x 8.27 in; leave room for cbar)
    if max_dist is None: max_dist = xmax
    if km_per_in is None or panel_h_in is None:          # standalone use: same scale rule
        km_per_in, panel_h_in = _section_scale(max_dist, zbot)
    # This section: width ∝ its length at the shared km/in; height is the SHARED height
    panel_w_in = max(1.2, xmax / km_per_in)
    # Explicit layout in inches: every panel is EXACTLY panel_w_in x panel_h_in, so
    # the drawn VE is exact and identical across sections (plt.subplots margins and
    # colorbar(ax=...) used to shrink panels by different amounts).
    strip_h = _strip_height_in(_GEO)
    L_in, R_in = 0.75, 1.05             # ylabel+ticks | gap+colorbar+its labels
    T_in = 0.55 + 0.30 + strip_h        # suptitle + panel title + geology strips
    B_in, G_in = 0.65, 0.42             # xticks+xlabel | between panels (titles)
    fig_w = L_in + panel_w_in + R_in
    fig_h = T_in + nrow * panel_h_in + (nrow - 1) * G_in + B_in
    vexag = (xmax / panel_w_in) / (zbot / panel_h_in)   # drawn VE, exact
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for i in range(nrow):
        y0 = fig_h - T_in - (i + 1) * panel_h_in - i * G_in
        axes.append(fig.add_axes([L_in / fig_w, y0 / fig_h, panel_w_in / fig_w,
                                  panel_h_in / fig_h],
                                 sharex=axes[0] if axes else None))
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    fig.suptitle(f'{label}:  ({lat1:.1f}°,{lon1:.1f}°) -> ({lat2:.1f}°,{lon2:.1f}°)   '
                 f'[{xmax:.0f} km, VE {vexag:.1f}x]', fontsize=11, fontweight='bold',
                 y=1.0 - 0.12 / fig_h, va='top')
    print(f"  Panels {panel_w_in:.2f} x {panel_h_in:.2f} in  ->  VE {vexag:.2f}x")

    def _pcolor(ax, data, cmap, vmin, vmax, clabel, extend='both', title=''):
        Zm = np.ma.array(data, mask=~np.isfinite(data))
        if smooth:                            # MATLAB-style continuous shading (interp)
            im = ax.pcolormesh(D, Z, Zm, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        else:
            lv = np.linspace(vmin, vmax, (ncolors or 16) + 1)   # contourf bands, like the maps
            im = ax.contourf(D, Z, Zm, levels=lv, cmap=cmap, extend=extend, antialiased=True)
            try:
                im.set_edgecolor('face')
            except Exception:
                for c in getattr(im, 'collections', []):
                    c.set_edgecolor('face')
        ax.set_ylim(zbot, 0); ax.set_xlim(0, xmax)
        ax.set_ylabel('Depth (km)', fontsize=9); ax.tick_params(labelsize=8)
        pos = ax.get_position()           # colorbar in its own axes: panel keeps its size
        cax = fig.add_axes([pos.x1 + 0.10 / fig_w, pos.y0, 0.16 / fig_w, pos.height])
        cb = fig.colorbar(im, cax=cax, extend=extend)
        _apply_nice_ticks(cb, vmin, vmax)       # round 2-dec labels (4.53, 3.47, 0.90 ...)
        cb.set_label(clabel, fontsize=8); cb.ax.tick_params(labelsize=7)
        if title: ax.set_title(title, fontsize=9, loc='left')
        if moho_prof is not None:
            _draw_moho(ax, dist, moho_prof, zbot, moho_mask, lw=1.1)
        if moho_ccp_prof is not None:       # most recent CCP refined Moho pick, solid red
            ax.plot(dist, moho_ccp_prof, color='red', lw=1.4, alpha=0.95, zorder=4,
                    solid_capstyle='round')
        return im

    ncd = _ncd(ncolors, smooth)              # colours per map: 256 (smooth) or ncolors (bands)
    r = 0
    if 'abs' in rows:
        _pcolor(axes[r], vsv_p, _cmap(CMAP_ABS, ncd), vmin_v, vmax_v,
                f'{vname} (km/s)', title=f'{vname} (km/s)'); r += 1
    if 'rel' in rows:
        _relcm = _cmap('RdBu', ncd) if dvsv_cmap == 'RdBu' else \
            matplotlib.colormaps[dvsv_cmap].resampled(ncd)   # e.g. jet_r (flipped MATLAB jet)
        _pcolor(axes[r], dln_p, _relcm, -clim_rel, clim_rel,
                f'd{vname} (%)', title=f'd{vname} (%) vs {_ref_lbl}'); r += 1
    if 'vdiff' in rows and vdiff_p is not None:
        _pcolor(axes[r], vdiff_p, _cmap('RdBu', ncd), -clim_diff, clim_diff,
                f'{vname} vs {_REF["label"]} (%)',
                title=f'{vname}: this model vs {_REF["label"]} (%)  (blue = faster here)'); r += 1
    _xicm = _cmap('rwb', ncd) if xi_cmap == 'rwb' else matplotlib.colormaps[xi_cmap].resampled(ncd)
    if 'xi' in rows and have_xi and xi_panel in ('xi', 'both'):
        _xt = f'{XI_TITLE}   [{xi_label}]' if xi_label else XI_TITLE
        _pcolor(axes[r], xi_p, _xicm, XI_LIM[0], XI_LIM[1], 'xi', title=_xt); r += 1
    if 'xi2' in rows and xi2_p is not None:         # second model's xi (same scale/colormap)
        _pcolor(axes[r], xi2_p, _xicm, XI_LIM[0], XI_LIM[1], 'xi',
                title=f'{XI_TITLE}   [{xi2_label}]'); r += 1
    if 'xidiff' in rows and xidiff_p is not None:   # xi difference (model1 - model2)
        _dcm = _cmap('RdBu', ncd) if dvsv_cmap == 'RdBu' else matplotlib.colormaps[dvsv_cmap].resampled(ncd)
        _l1 = xi_label or 'model 1'
        _pcolor(axes[r], xidiff_p, _dcm, -xi_diff_lim, xi_diff_lim, 'Δxi',
                title=f'Δxi  ({_l1} − {xi2_label})   blue = {_l1} higher'); r += 1
    if 'dxi' in rows and dxi_p is not None:
        _pcolor(axes[r], dxi_p, _cmap('RdBu', ncd), -clim_xi, clim_xi,
                'dlnXi (%)', title=f'dlnXi (%) vs {_xref_lbl}'); r += 1
    if 'vpvs' in rows and vpvs_p is not None:
        fv = vpvs_p[np.isfinite(vpvs_p)]
        mv, sv = np.nanmedian(fv), np.nanstd(fv)
        _pcolor(axes[r], vpvs_p, _cmap(CMAP_VPVS, ncd), mv - sigma_vsv * sv,
                mv + sigma_vsv * sv, 'Vp/Vs', title='Vp/Vs'); r += 1
    if 'ccp' in rows and ccp_p is not None:
        import plot_ccp_profiles as pcp
        cclim = getattr(plot_section, '_ccp_clim', 0.08)
        _ccpname = getattr(plot_section, '_ccp_cmap', None)
        _use_soft = (not _ccpname) or str(_ccpname).lower() == 'soft'
        cmap_ccp = (pcp._ccp_cmap(getattr(plot_section, '_ccp_neutral', 0.25)).resampled(ncd)
                    if _use_soft else _cmap(_ccpname, ncd))
        ctag = getattr(plot_section, '_ccp_tag', '')
        _pcolor(axes[r], ccp_p, cmap_ccp, -cclim, cclim, 'RF amplitude', extend='both',
                title='CCP RF amplitude' + (f' (pvh.{ctag})' if ctag else '')); r += 1
    if 'err' in rows and have_err:
        vmax_e = np.nanpercentile(err_p[np.isfinite(err_p)], 90)
        _pcolor(axes[r], err_p, _cmap('YlOrRd', ncd), 0, vmax_e,
                f'{vname} IQR/2', extend='max',
                title='Model uncertainty (posterior IQR/2)'); r += 1

    axes[-1].set_xlabel('Distance along profile (km)', fontsize=9)
    axes[-1].set_xticks(np.arange(0, xmax+1, 100))

    geo = _add_geology(axes[0], axes, plon, plat, dist)
    if geo is not None:
        _geology_legend(fig, [geo], axes[-1], ncol=4 if panel_w_in > 5 else 2)

    os.makedirs(out_dir, exist_ok=True)
    _rtag = {'profile': '.ref-profile', 'allprofiles': '.ref-allprof'}.get(var_ref, _REF_TAG)
    out = os.path.join(out_dir, f'xsection.{label}{_rtag}{geol_tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.png','.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out} (+ .pdf)")



def _name_by_orientation(sections):
    """N-S lines -> NS1.. (west to east, start at the south end); W-E lines -> EW1..
    (north to south, start at the west end). Same split as --stack."""
    out = []
    for la1, lo1, la2, lo2, _ in sections:
        dy = abs(la2 - la1)
        dx = abs(lo2 - lo1) * np.cos(np.radians(0.5 * (la1 + la2)))
        if dy >= dx and la1 > la2:            # N-S: 0 km at the south end
            la1, lo1, la2, lo2 = la2, lo2, la1, lo1
        elif dy < dx and lo1 > lo2:           # W-E: 0 km at the west end
            la1, lo1, la2, lo2 = la2, lo2, la1, lo1
        out.append((la1, lo1, la2, lo2, ''))
    groups, _ = _split_by_orientation(out)
    named = []
    for key, pre in (('ns', 'NS'), ('ew', 'EW')):
        for k, (la1, lo1, la2, lo2, _) in enumerate(groups[key][0]):
            named.append((la1, lo1, la2, lo2, f'{pre}{k + 1}'))
    return named


def _ginput_sections(d, ref_sections=None, stations=None, tect_npz=None, bnd_npz=None):
    """Pick sections on the profile-location map (Albers, 2022 tectonic-map colours,
    crustal boundaries, unit names, stations; ref_sections dashed for reference).
    Left click = start, left click = end (the great-circle line is drawn at once);
    right click = undo the last point; Enter = done; Esc = quit without picking.
    Returns [(lat1, lon1, lat2, lon2, label)] named NS1.. / EW1.. by orientation."""
    # import everything first: some modules call matplotlib.use('Agg') on import, which
    # would switch an interactive backend chosen earlier straight back to Agg
    import cartopy.crs as ccrs
    from plot_depth_slice import _add_states_ocean
    import wa_basemap as wb
    webagg, errs = False, []
    print(f"  DISPLAY={os.environ.get('DISPLAY', '') or '(not set)'}")
    for bk in ['TkAgg', 'QtAgg', 'Qt5Agg', 'GTK3Agg']:
        try:
            plt.switch_backend(bk)
            break
        except Exception as e:
            errs.append(f"{bk}: {str(e).splitlines()[0]}")
    else:                                   # no window system: pick in a web browser
        try:
            plt.switch_backend('WebAgg')
            webagg = True
        except Exception as e:
            errs.append(f"WebAgg: {str(e).splitlines()[0]}")
    canvas = type(plt.figure().canvas).__name__; plt.close('all')
    if canvas == 'FigureCanvasAgg':
        print("ERROR: no interactive matplotlib backend:\n    " + "\n    ".join(errs) +
              "\n  Fix: run where a display reaches this shell (ssh -X / -Y, or the desktop),"
              "\n       or install a GUI toolkit in this env (conda install tk), or tornado for"
              "\n       the browser picker (conda install tornado).")
        return []
    if errs:
        print("  (skipped: " + "; ".join(errs) + ")")
    print(f"  picker backend: {plt.get_backend()}")
    if webagg:
        print("  No window system - the map opens in a WEB BROWSER: open the URL printed below\n"
              "  (from another machine: ssh -L 8988:localhost:8988 this-host, then\n"
              "  http://localhost:8988). Click inside the map; Enter when done.")
    Lon, Lat = np.asarray(d['Lon'], float), np.asarray(d['Lat'], float)
    tr = ccrs.PlateCarree()
    proj = ccrs.AlbersEqualArea(central_longitude=121.0, standard_parallels=(-17.5, -31.5))
    fig = plt.figure(figsize=(11, 11))
    if type(fig.canvas).__name__ == 'FigureCanvasAgg':
        plt.close(fig)
        print(f"ERROR: backend fell back to Agg ({plt.get_backend()}) - nothing to click on.")
        return []
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.88], projection=proj)
    cb = (Lon.min() - 1.0, Lon.max() + 1.0, Lat.min() - 1.0, Lat.max() + 1.0)
    try:
        wb.add_map2022_units(ax, tect_npz, transform=tr, clip_box=cb, zorder=0)
        wb.add_crustal_boundaries(ax, bnd_npz, transform=tr, clip_box=cb, zorder=4.2,
                                  lw_litho=1.1, lw_crust=0.5, alpha=0.85)
        wb.add_map2022_labels(ax, transform=tr, level=1, scale=1.2, clip_box=cb)
    except Exception as e:
        print(f"(map layers unavailable: {e})")
    _add_states_ocean(ax, ocean='white', states='#e8e8e8')
    gl = ax.gridlines(crs=tr, draw_labels=True, xlocs=np.arange(100, 151, 5),
                      ylocs=np.arange(-45, 1, 5), color='#858c8c', linewidth=0.35)
    gl.top_labels = gl.right_labels = False
    if stations is not None:
        ax.scatter(stations[0], stations[1], marker='^', s=12, c='k', edgecolors='w',
                   linewidths=0.3, zorder=5.5, transform=tr)
    for la1, lo1, la2, lo2, lab in (ref_sections or []):
        plat, plon, _ = _great_circle_path(la1, lo1, la2, lo2)
        ax.plot(plon, plat, '--', color='0.35', lw=1.2, zorder=6, transform=tr)
        ax.text(lo1, la1, lab, fontsize=7, color='0.35', zorder=6, transform=tr)
    ax.set_extent([Lon.min() - 0.5, Lon.max() + 0.5, Lat.min() - 0.5, Lat.max() + 0.5], crs=tr)

    pts, arts, done = [], [], {'ok': False}

    def lonlat(x, y):
        return tr.transform_point(x, y, proj)                  # -> (lon, lat)

    def counts():
        secs = [(pts[k][1], pts[k][0], pts[k + 1][1], pts[k + 1][0], '')
                for k in range(0, len(pts) - 1, 2)]
        g, _ = _split_by_orientation(secs) if secs else ({'ns': ([], ''), 'ew': ([], '')}, 0)
        return len(g['ns'][0]), len(g['ew'][0])

    def title():
        nns, new = counts()
        ax.set_title(f'Pick sections: click START then END.  Right click = undo,  Enter = done,  '
                     f'Esc = quit\nN-S: {nns}   W-E: {new}' +
                     ('   (dashed grey = current lines)' if ref_sections else ''), fontsize=10)
        fig.canvas.draw_idle()

    def on_click(ev):
        tb = getattr(fig.canvas, 'toolbar', None)
        if ev.inaxes is not ax or (tb is not None and getattr(tb, 'mode', '')):
            return                                             # zoom/pan active
        if ev.button == 1:
            lo, la = lonlat(ev.xdata, ev.ydata)
            pts.append((lo, la))
            if len(pts) % 2:
                arts.append([ax.plot(lo, la, 'o', color='k', ms=6, zorder=8, transform=tr)[0]])
            else:
                (lo1, la1), (lo2, la2) = pts[-2], pts[-1]
                plat, plon, dist = _great_circle_path(la1, lo1, la2, lo2)
                a = ax.plot(plon, plat, '-', color='k', lw=2.2, zorder=8, transform=tr)
                a += ax.plot(lo2, la2, 's', color='k', ms=6, zorder=8, transform=tr)
                a.append(ax.text(0.5 * (lo1 + lo2), 0.5 * (la1 + la2), f'{dist[-1]:.0f} km',
                                 fontsize=8, color='darkred', zorder=9, transform=tr,
                                 bbox=dict(fc='w', ec='none', alpha=0.7, pad=0.5)))
                arts.append(a)
                print(f"  line {len(pts) // 2}: ({la1:.2f},{lo1:.2f}) -> ({la2:.2f},{lo2:.2f}), "
                      f"{dist[-1]:.0f} km")
            title()
        elif ev.button == 3 and pts:
            pts.pop()
            for a in arts.pop():
                a.remove()
            title()

    def finish():
        plt.close(fig)
        if webagg:                                     # end plt.show()'s web server loop
            from tornado.ioloop import IOLoop
            loop = IOLoop.current()
            loop.add_callback(loop.stop)

    def on_key(ev):
        if ev.key == 'enter':
            done['ok'] = True
            finish()
        elif ev.key == 'escape':
            finish()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    title()
    print("\nPick: left click start, left click end (repeat); right click undoes; "
          "Enter when done, Esc to quit.")
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        pass
    if not done['ok']:
        print("Picking cancelled - nothing saved.")
        return []
    if len(pts) % 2:
        print("Odd number of clicks; dropping the last unpaired point.")
        pts.pop()
    secs = [(pts[k][1], pts[k][0], pts[k + 1][1], pts[k + 1][0], '')
            for k in range(0, len(pts), 2)]
    named = _name_by_orientation(secs)
    for la1, lo1, la2, lo2, lab in named:
        print(f"  {lab}: ({la1:.2f},{lo1:.2f}) -> ({la2:.2f},{lo2:.2f})")
    plt.switch_backend('Agg')
    return named

# ── Main ──────────────────────────────────────────────────────────────────────
def _index_footprint(args):
    if not args.footprint:
        return None
    from footprint import Footprint
    return Footprint(args.footprint_npz, args.footprint_outline)


def _load_stations(path, cols=(2, 1)):
    """Station lon/lat from a 'code lat lon' style file (whitespace or comma;
    header/comment/bad lines skipped). cols = (lon_col, lat_col), 0-indexed."""
    lon, lat = [], []
    with open(path) as f:
        for line in f:
            p = line.replace(',', ' ').split()
            if not p or p[0].startswith('#'):
                continue
            try:
                lon.append(float(p[cols[0]])); lat.append(float(p[cols[1]]))
            except (ValueError, IndexError):
                continue
    return np.array(lon), np.array(lat)




def _split_by_orientation(sections):
    """{'ns': (sections, title), 'ew': (...)} by the line's overall bearing (north-south if
    it spans more km in latitude than in longitude). N-S lines are ordered west to east,
    W-E lines north to south, as they sit on the map. Also returns the longest length
    (km) so both pages share one horizontal scale."""
    ns, ew, longest = [], [], 0.0
    for s in sections:
        la1, lo1, la2, lo2, _ = s
        dy = abs(la2 - la1)
        dx = abs(lo2 - lo1) * np.cos(np.radians(0.5 * (la1 + la2)))
        (ns if dy >= dx else ew).append(s)
        longest = max(longest, _great_circle_path(la1, lo1, la2, lo2)[2][-1])
    ns.sort(key=lambda s: 0.5 * (s[1] + s[3]))
    ew.sort(key=lambda s: -0.5 * (s[0] + s[2]))
    return {'ns': (ns, 'North-south'), 'ew': (ew, 'West-east')}, longest


def _save_sections(sections, path):
    if os.path.isfile(path):                          # keep the old list
        import shutil, time
        bak = f"{path}.bak.{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(path, bak)
        print(f"  previous sections kept as {bak}")
    with open(path, 'w') as f:
        f.write("# lat1 lon1 lat2 lon2 label\n")
        for lat1, lon1, lat2, lon2, lab in sections:
            f.write(f"{lat1:.4f} {lon1:.4f} {lat2:.4f} {lon2:.4f} {lab}\n")
    print(f"Saved sections: {path}")


def _load_sections(path):
    secs = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            p = line.split()
            secs.append((float(p[0]), float(p[1]), float(p[2]), float(p[3]),
                         p[4] if len(p) > 4 else f'sec{len(secs)+1}'))
    return secs



STACK_FIELDS = ('vsv', 'dvsv', 'xi', 'dxi', 'vpvs', 'dvpvs', 'vdiff', 'err')


def plot_stack(d, sections, field, out_dir, moho_file=None, ds_deg=0.08,
               ncolors=16, d_max=None, sigma_vsv=1.5, clim_rel=6.0,
               ref_mean=None, row_h_in=None, page_w_in=9.5, gap_in=0.55,
               ve=DEFAULT_VE, vmin=None, vmax=None, moho_mask=0.5,
               xi_ref='depth', clim_xi=5.0, div_gap=None, div_white=None, clim_vpvs=3.0,
               clim_diff=4.0, maxd_km=None, tag='', group='', col_w_in=7.0,
               moho_ccp_file=None, smooth=False):
    """Stack fields for all sections on one page. Rows = sections (same height, width
    proportional to length, longest spans the column width); columns = fields
    (field = 'vsv' or a list / comma string such as 'vsv,vpvs': side by side, same rows,
    same km/inch). One field uses page_w_in, several use col_w_in each.
    ve = vertical exaggeration. Each panel has its own colour bar."""
    global DIV_GAP, DIV_WHITE
    if div_gap is not None:
        DIV_GAP = div_gap
    if div_white is not None:
        DIV_WHITE = div_white
    import matplotlib.pyplot as plt
    fields = [f.strip() for f in (field.split(',') if isinstance(field, str) else field) if f.strip()]
    Lon = d['Lon']; Lat = d['Lat']; z = d['z']
    if d_max: zm = z <= d_max; z = z[zm]
    else: zm = np.ones(len(z), bool)
    zbot = z[-1]
    is_zt = 'ZT' in os.path.basename(getattr(plot_section,'_fvsname','')) or _varies(d, 'Xi')
    vname = _vname(d, is_zt)

    FKEY = {'vsv':'Vsv','dvsv':'Vsv','xi':'Xi','dxi':'Xi','err':'Vsv_err',
            'vpvs':'Vpvs','dvpvs':'Vpvs','vdiff':'Vsv'}
    ncd = _ncd(ncolors, smooth)              # colours per map: 256 (smooth) or ncolors (bands)
    F = []                                   # per-field settings
    for f in fields:
        arrname = FKEY.get(f, 'Vsv')
        if arrname not in d:
            print(f"  field {f}: {arrname} not in Fvs - skipped"); continue
        if f == 'vdiff' and _REF is None:
            sys.exit("--stack vdiff needs --ref-fvs <reference model>")
        if f in ('xi', 'dxi', 'vpvs', 'dvpvs') and not _varies(d, arrname):
            print(f"  field {f}: {arrname} is fixed at {np.nanmean(d[arrname]):.4f} in this "
                  f"model - skipped (use a model that inverts it)"); continue
        ARR = d[arrname][:, zm]
        st = dict(f=f, arr=ARR)
        if f == 'dvsv':
            st['ref'] = ref_mean if ref_mean is not None else _node_ref(ARR)
        elif f == 'dvpvs':
            st['ref'] = _node_ref(ARR)          # model (or --ref-domain) mean at each depth
        elif f == 'dxi':
            st['ref'] = _xi_ref(ARR, xi_ref)
            print(f"  dlnXi reference ({_xi_ref_label(xi_ref)}): "
                  f"{np.nanmin(st['ref']):.3f}-{np.nanmax(st['ref']):.3f}")
        st['cmap'], st['clab'], st['ext'] = {
            'vsv':   (_cmap(CMAP_ABS, ncd), f'{vname} (km/s)', 'both'),
            'dvsv':  (_cmap('RdBu', ncd), f'd{vname} (%)', 'both'),
            'xi':    (_cmap('rwb', ncd), 'xi', 'both'),
            'dxi':   (_cmap('RdBu', ncd), 'dlnXi (%)', 'both'),
            'vpvs':  (_cmap(CMAP_VPVS, ncd), 'Vp/Vs', 'both'),
            'dvpvs': (_cmap('RdBu', ncd), 'dln(Vp/Vs) (%)', 'both'),
            'vdiff': (_cmap('RdBu', ncd), f'{vname} vs {_REF["label"] if _REF else "ref"} (%)', 'both'),
        }.get(f, (_cmap('YlOrRd', ncd), f'{vname} IQR/2', 'max'))
        st['name'] = {'vsv': vname, 'dvsv': f'd{vname}', 'dxi': 'dlnXi', 'xi': 'xi', 'vpvs': 'Vp/Vs',
                      'dvpvs': 'dln(Vp/Vs)', 'err': f'{vname} error',
                      'vdiff': f'{vname} minus {_REF["label"] if _REF else "ref"}'}.get(f, f)
        F.append(st)
    if not F:
        print("  nothing to plot"); return

    _mi = None
    if moho_file:
        try: _mi = _load_moho_interp(moho_file)
        except Exception: pass
    _mc = None
    if moho_ccp_file:
        try: _mc = _load_moho_interp(moho_ccp_file)
        except Exception: pass

    prof=[]
    for (la1,lo1,la2,lo2,lab) in sections:
        plat,plon,dist=_great_circle_path(la1,lo1,la2,lo2,ds_deg)
        data = {}
        for st in F:
            samp = _sample_profile(Lon, Lat, st['arr'], plon, plat)
            if st['f'] == 'dvsv':
                samp = (samp - st['ref'][None, :]) / st['ref'][None, :] * 100
            elif st['f'] in ('dxi', 'dvpvs'):
                samp = (samp / st['ref'][None, :] - 1.0) * 100
            elif st['f'] == 'vdiff':
                samp = _ref_diff(samp, plon, plat, z)
            data[st['f']] = samp
        mp=None
        if _mi is not None:
            mp=np.array([float(np.ravel(_mi(a,o))[0]) for a,o in zip(plat,plon)])
        mp_ccp=None
        if _mc is not None:
            mp_ccp=np.array([float(np.ravel(_mc(a,o))[0]) for a,o in zip(plat,plon)])
        prof.append(dict(lab=lab,dist=dist,data=data,moho=mp,moho_ccp=mp_ccp,plat=plat,plon=plon))
    maxd=max(p['dist'][-1] for p in prof)
    if maxd_km:                 # shared scale across split pages (same km/inch on each)
        maxd=max(maxd, maxd_km)

    # Horizontal scale: longest fills the column width. Row height from VE:
    #   km_per_in = maxd/col_w ; row_h_in = zbot*ve/km_per_in
    ncol = len(F)
    col_w = page_w_in if ncol == 1 else col_w_in
    km_per_in, ve_h = _section_scale(maxd, zbot, ve, col_w)
    if row_h_in is None:
        row_h_in = ve_h

    def limits(f, data):
        if vmin is not None and vmax is not None and ncol == 1:
            return vmin, vmax
        if f in ('vsv', 'vpvs'):
            if f == 'vsv' and vmin is not None and vmax is not None:
                return vmin, vmax
            fv = data[np.isfinite(data)]
            med, sd = np.nanmedian(fv), np.nanstd(fv)
            return med - sigma_vsv * sd, med + sigma_vsv * sd
        if f == 'err':
            fv = data[np.isfinite(data)]; return 0, np.nanpercentile(fv, 90)
        return {'dvsv': (-clim_rel, clim_rel), 'xi': XI_LIM, 'dxi': (-clim_xi, clim_xi),
                'dvpvs': (-clim_vpvs, clim_vpvs), 'vdiff': (-clim_diff, clim_diff)}.get(f, (vmin, vmax))

    n=len(prof)
    strip_h=_strip_height_in(_GEO)          # room for geology strips above each row
    gap_in=gap_in+strip_h; top_in=0.9+strip_h+(0.25 if ncol > 1 else 0)
    fig_h=n*row_h_in + (n-1)*gap_in + 1.2 + strip_h
    left_in = 0.08 * (page_w_in + 1.8)                       # as the one-field page
    # next column starts right after this page's longest row (same km/inch throughout)
    page_long = max(p['dist'][-1] for p in prof)
    pitch = page_long / km_per_in + 1.05                     # panel + colour bar + label
    fig_w = left_in + ncol * pitch + 0.15 if ncol > 1 else page_w_in + 1.8
    fig=plt.figure(figsize=(fig_w,fig_h))
    names = ' | '.join(st['name'] for st in F)
    rel = [st['f'] for st in F if st['f'] in ('dvsv', 'dvpvs', 'dxi')]
    reftxt = ''
    if rel:                             # what the relative fields are relative to
        reftxt = ('\n' + ', '.join({'dvsv': f'd{vname}', 'dvpvs': 'dln(Vp/Vs)',
                                    'dxi': 'dlnXi'}[f] for f in rel) + ' vs ' +
                  (_xi_ref_label(xi_ref) if rel == ['dxi'] else _ref_label()))
    fig.suptitle(f'{names} — {group + " " if group else ""}{n} sections '
                 f'[depth 0-{zbot:.0f} km, column width {maxd:.0f} km, VE {ve:.0f}x]' + reftxt,
                 fontsize=11, fontweight='bold')
    rh=row_h_in/fig_h
    geo_res=[]
    for c, st in enumerate(F):
        x0_in = left_in + c * pitch
        if ncol > 1:
            fig.text((x0_in + col_w / 2) / fig_w, 1 - 0.75 / fig_h, st['name'], ha='center',
                     va='top', fontsize=11, fontweight='bold')
        for i,p in enumerate(prof):
            w_in = p['dist'][-1] / km_per_in
            y0=1-(top_in/fig_h)-(i+1)*rh-i*(gap_in/fig_h)
            ax=fig.add_axes([x0_in / fig_w, y0, w_in / fig_w, rh])
            D,Z=np.meshgrid(p['dist'],z,indexing='ij')
            data = p['data'][st['f']]
            vmn, vmx = limits(st['f'], data)
            Zm = np.ma.array(data, mask=~np.isfinite(data))
            if smooth:                        # MATLAB-style continuous shading
                im=ax.pcolormesh(D,Z,Zm,cmap=st['cmap'],vmin=vmn,vmax=vmx,shading='gouraud')
            else:
                lv = np.linspace(vmn, vmx, (ncolors or 16) + 1)   # contourf bands, like the maps
                im=ax.contourf(D,Z,Zm,levels=lv,cmap=st['cmap'],extend=st['ext'],antialiased=True)
                try:
                    im.set_edgecolor('face')
                except Exception:
                    for c in getattr(im,'collections',[]): c.set_edgecolor('face')
            ax.set_ylim(zbot,0); ax.set_xlim(0,p['dist'][-1])
            ax.set_ylabel('Depth',fontsize=7); ax.tick_params(labelsize=6)
            ax.set_xticks(np.arange(0,p['dist'][-1]+1,200))
            # section name bottom-left, on an opaque white bar above the colours and Moho veil
            ax.text(0.004, 0.03, f"{p['lab']} [{p['dist'][-1]:.0f} km]", transform=ax.transAxes,
                    fontsize=8, fontweight='bold', ha='left', va='bottom', zorder=20,
                    bbox=dict(fc='white', ec='none', alpha=1.0, pad=1.5))
            if p['moho'] is not None:
                _draw_moho(ax, p['dist'], p['moho'], zbot, moho_mask, lw=0.9)
            if p.get('moho_ccp') is not None:
                ax.plot(p['dist'], p['moho_ccp'], color='red', lw=1.1, alpha=0.95,
                        zorder=4, solid_capstyle='round')
            if i==n-1: ax.set_xlabel('Distance (km)',fontsize=8)
            # per-panel colorbar immediately right of THIS panel
            cax=fig.add_axes([(x0_in + w_in + 0.09) / fig_w, y0, 0.13 / fig_w, rh])
            cb=fig.colorbar(im,cax=cax,extend=st['ext']); cb.ax.tick_params(labelsize=6)
            _apply_nice_ticks(cb, vmn, vmx)
            if i==0: cb.set_label(st['clab'],fontsize=8)
            # boundary names only above the first column (arrows on every column)
            geo_res.append(_add_geology(ax, [ax], p['plon'], p['plat'], p['dist'],
                                        names=None if c == 0 else False))

    if geo_res:
        _geology_legend(fig, geo_res, ax, below_in=0.55)
    os.makedirs(out_dir,exist_ok=True)
    key = '_'.join(st['f'] for st in F)
    out=os.path.join(out_dir,f'xsection_stack.{key}{_REF_TAG}{tag}.png')
    fig.savefig(out,dpi=150,bbox_inches='tight')
    fig.savefig(out.replace('.png','.pdf'),bbox_inches='tight')
    plt.close(fig)
    print(f"Stack ({key}): {out} (+ .pdf)")


def main():
    global MAX_NODE_DIST, _REF_NODES, _REF_NAME, _REF_TAG
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs',     required=True)
    ap.add_argument('--start',   default=None,
                    help='Start point "lat,lon" (single section)')
    ap.add_argument('--end',     default=None,
                    help='End point "lat,lon" (single section)')
    ap.add_argument('--label',   default='section',
                    help='Label for single section (default: section)')
    ap.add_argument('--sections', nargs='+', default=None,
                    help='Multiple sections: "lat1,lon1/lat2,lon2/label" ...')
    ap.add_argument('--ref-domain', default=None,
                    help='dln fields (dVsv, dlnXi, dln Vp/Vs) relative to this tectonic '
                         'domain\'s mean at each depth, e.g. yilgarn (basemap/domains.csv); '
                         'outputs get a .ref-<domain> tag. Default: whole-model mean')
    ap.add_argument('--var-ref', default='model',
                    choices=['model', 'yilgarn', 'profile', 'allprofiles'],
                    help='baseline for the variation (dVsv/dlnXi) panels, per depth: '
                         'model = whole-model mean (default); yilgarn = Yilgarn-domain mean '
                         '(= --ref-domain yilgarn); profile = THIS profile\'s own along-line '
                         'mean; allprofiles = mean of all profiles in --load-sections. '
                         'profile/allprofiles tag outputs .ref-profile / .ref-allprof')
    ap.add_argument('--max-node-dist', type=float, default=MAX_NODE_DIST,
                    help='blank section points further than this (deg) from any model node, '
                         'as on the maps (default 0.5; 0 = no masking)')
    ap.add_argument('--ds',       type=float, default=0.08,
                    help='Sample spacing in degrees (default 0.08 ≈ 9 km)')
    ap.add_argument('--d-max',    type=float, default=60.0,
                    help='Max depth to plot (default 60 km)')
    ap.add_argument('--sigma-vsv', type=float, default=1.5,
                    help='Colorlim = median ± sigma*std for Vsv (default 1.5)')
    ap.add_argument('--clim-rel',  type=float, default=6.0,
                    help='±clim for dlnVsv %% (default 6)')
    ap.add_argument('--ncolors',   type=int,   default=16,
                    help='Discrete colour levels per colormap (default 16; e.g. 32 for finer)')
    ap.add_argument('--vs-smooth', dest='vs_smooth', type=float, default=0.0, metavar='SIGMA',
                    help='gently Gaussian-smooth the Vs panel only (NaN-aware, SIGMA in grid '
                         'cells; try 1-2). 0 = off. Does not touch xi / CCP.')
    ap.add_argument('--dvsv-cmap', dest='dvsv_cmap', default='RdBu', metavar='NAME',
                    help="dlnVs / dViso panel colormap (default RdBu). Any matplotlib name, "
                         "e.g. jet_r (flipped MATLAB jet), seismic, coolwarm.")
    ap.add_argument('--xi2-fvs', dest='xi2_fvs', default=None, metavar='MODEL.npz[:LABEL]',
                    help="add a SECOND xi panel from another ZT model (same scale/colormap) "
                         "below the main xi panel, to compare two xi models in one figure. "
                         "Use --panels xi,xi2 for a xi-vs-xi plot. e.g. "
                         "--xi2-fvs 'Fvs...iter.1.ZT.npz:no Moho'")
    ap.add_argument('--xi-label', dest='xi_label', default='', metavar='LABEL',
                    help='tag for the MAIN xi panel title (e.g. "with Moho"), for xi2 comparisons.')
    ap.add_argument('--xi-diff-lim', dest='xi_diff_lim', type=float, default=0.05, metavar='D',
                    help='symmetric +/-D range for the xi-difference panel (main - xi2). Add '
                         "'xidiff' to --panels to show it, e.g. --panels xi,xi2,xidiff. Default 0.05.")
    ap.add_argument('--xi-cmap', dest='xi_cmap', default='RdBu', metavar='NAME',
                    help="xi colormap: a general blue-red diverging map (RdBu default, or "
                         "coolwarm / bwr / seismic), or 'rwb' for the custom red-white-blue.")
    ap.add_argument('--xi-gap', dest='xi_gap', type=float, default=None, metavar='GAP',
                    help="xi white-centre width for the 'rwb' colormap only (default 0.12). "
                         'Lower = more gradual; 0 = fully smooth. Ignored for RdBu/others.')
    ap.add_argument('--field-style', default='contourf', choices=['contourf', 'smooth'],
                    help='contourf = discrete --ncolors bands (default); smooth = continuous '
                         'interpolated shading (pcolormesh gouraud), matching the MATLAB landscape')
    ap.add_argument('--ginput',  action='store_true',
                    help='Pick sections interactively on the profile-location map (2022 '
                         'colours, boundaries, stations); with --load-sections the current '
                         'lines are shown dashed. Lines are named NS1../EW1.. by orientation. '
                         'Combine with --save-sections FILE (old file kept as FILE.bak.<time>)')
    ap.add_argument('--moho',      default=None,
                    help='Moho overlay: AusMoho grid text (AR23-moho-hmp.txt, dashed black + '
                         'grey veil below) OR a CCP refined pick .mat (moho_refined)')
    ap.add_argument('--moho-ccp',  default=None,
                    help='most recent CCP refined Moho pick .mat (moho_ccp_refined.final.mat) '
                         'overlaid as a solid red line along the profile (no veil); pair with '
                         '--moho AR23... to show the CCP pick over the regional AusMoho')
    ap.add_argument('--ccp-pmat',  default=None,
                    help='CCP RF stack P.mat (pvh.4 current) -> adds a "ccp" panel: RF amplitude '
                         'sampled along the profile (contourf, same Moho overlays). Select it with '
                         "--panels vsv,xi,ccp. Reads the CMP cube from P.ofile1 (or --ccp-cmp).")
    ap.add_argument('--ccp-cmp',   default=None, help='override the CMP cube file (else from P.ofile1)')
    ap.add_argument('--ccp-amp-var', default='cmp', help='volume variable in the CMP file (default cmp)')
    ap.add_argument('--ccp-clim',  type=float, default=0.08, help='CCP panel symmetric colour limit (±)')
    ap.add_argument('--ccp-neutral', type=float, default=0.25,
                    help='CCP soft white centre: fraction of ±clim near zero that fades to white')
    ap.add_argument('--ccp-method', default='linear', choices=['linear', 'nearest', 'cubic'],
                    help='CCP smooth-pass interpolant')
    ap.add_argument('--ccp-cmap', default='PuOr', metavar='NAME',
                    help="colormap for the CCP RF panel, kept distinct from the Vsv/xi/Vp-Vs "
                         "palettes (default PuOr = orange -/ purple +). Pass a matplotlib name; "
                         "'' or 'soft' restores the red-white-blue --ccp-neutral map.")
    ap.add_argument('--xi-lim', nargs=2, type=float, default=None, metavar=('LO', 'HI'),
                    help='xi colour range for the absolute xi panel, e.g. 0.95 1.05 '
                         '(default 0.92 1.08)')
    ap.add_argument('--xi-squared', action='store_true',
                    help='show conventional xi = (Vsh/Vsv)^2 (the Fvs Xi is linear Vsh/Vsv), 0.88-1.12')
    ap.add_argument('--xi-panel', default='xi', choices=['xi', 'dxi', 'both'],
                    help='per-section xi row(s): xi, dlnXi, or both (ZT models)')
    ap.add_argument('--panels', default=None,
                    help='per-section: keep only these panels, in this order (comma list of '
                         'vsv,dvsv,xi,dxi,vpvs,vdiff,err). e.g. --panels vsv,xi for a 2-panel plot. '
                         'Default: all available.')
    ap.add_argument('--xi-ref', default='depth', choices=['depth', 'global'],
                    help='dlnXi reference: whole-model mean at each depth (default) or '
                         'one whole-model mean over all depths')
    ap.add_argument('--vel', default='model', choices=['model', 'vsv_true'],
                    help="velocity in the Vs panels/stacks: model = the Fvs 'Vsv' array (Viso "
                         "for xi/ZT models); vsv_true = Viso/sqrt((2+Xi^2)/3) for xi models")
    ap.add_argument('--clim-vpvs', type=float, default=3.0,
                    help='dln(Vp/Vs) colour limit +/- %% (default 3)')
    ap.add_argument('--ref-fvs', default=None,
                    help='reference model: adds a "difference vs ref" row to each section '
                         '(and enables --stack vdiff): (Vs_model / Vs_ref - 1) * 100')
    ap.add_argument('--ref-vel', default='model', choices=['model', 'vsv_true'],
                    help='velocity taken from the reference model (as --vel)')
    ap.add_argument('--ref-label', default=None,
                    help='name of the reference in titles (default from the file, e.g. iter2)')
    ap.add_argument('--clim-diff', type=float, default=4.0,
                    help='difference-vs-ref colour limit +/- %% (default 4)')
    ap.add_argument('--vlim', nargs=2, type=float, default=None, metavar=('VMIN', 'VMAX'),
                    help='fixed colour limits (km/s) for the absolute Vs panel of every '
                         'per-section figure (use the same values for models you compare)')
    ap.add_argument('--clim-xi', type=float, default=5.0,
                    help='dlnXi colour limit, +/- %% (default 5)')
    ap.add_argument('--div-gap', type=float, default=0.20,
                    help='diverging colour maps skip +/- this band around white '
                         '(0 = classic RdBu through white; default 0.20)')
    ap.add_argument('--abs-skip', nargs=2, type=float, default=None, metavar=('LO', 'HI'),
                    help='absolute (Spectral) colour bar without its pale yellow middle: leave out LO-HI '
                         '(e.g. 0.36 0.70 = orange straight to green)')
    ap.add_argument('--div-white', type=float, default=0.5,
                    help='white band at the centre of diverging maps, in colour steps (default 0.5; 0 = none)')
    ap.add_argument('--moho-mask-alpha', type=float, default=0.5,
                    help='grey veil below the Moho to the bottom of the section (0-1, default 0.5)')
    ap.add_argument('--no-moho-mask', action='store_true', help='Moho line only, no grey veil')
    ap.add_argument('--save-sections', default=None,
                    help='Save picked/used section list to this file')
    ap.add_argument('--load-sections', default=None,
                    help='Load section list from a file (skip ginput)')
    ap.add_argument('--stack', default=None,
                    help='Stack fields for all sections, one page per orientation (rows = '
                         'sections). One field, or a comma list for columns side by side, e.g. '
                         'vsv,vpvs. Fields: ' + ', '.join(STACK_FIELDS))
    ap.add_argument('--vmin', type=float, default=None,
                    help='Fixed colour min for the stacked field (all panels same scale)')
    ap.add_argument('--vmax', type=float, default=None,
                    help='Fixed colour max for the stacked field')
    ap.add_argument('--stack-split', default='orient', choices=['orient', 'none'],
                    help='--stack: orient = two pages, north-south lines (.ns, ordered W->E) and '
                         'west-east lines (.ew, ordered N->S), same km/inch (default); '
                         'none = all sections on one page')
    ap.add_argument('--stack-ve', type=float, default=DEFAULT_VE,
                    help='Vertical exaggeration for stacked sections (1=true scale/flat, higher=taller; default 3).')
    ap.add_argument('--index-map', action='store_true',
                    help='Also draw an index map of all sections on the tectonic geology')
    ap.add_argument('--load-polylines', default=None,
                    help='profiles polylines npz (from mat_to_profiles.py): draw each line on '
                         'the index map along its TRUE picked path (curves kept) instead of a '
                         'straight chord. Lines are matched to the sections by label.')
    ap.add_argument('--map-stations', nargs='+', action='append', default=[],
                    metavar='CSV COLOUR MARKER LABEL [EDGE]',
                    help='extra station layer on the index map from a CSV with Latitude/Longitude '
                         'columns, drawn as MARKER (e.g. s=square, ^=triangle) in COLOUR with the '
                         'legend LABEL and optional marker EDGE colour (default white). Repeatable, '
                         'e.g. --map-stations phase4.csv lime s "Phase 4" '
                         '--map-stations perm.csv red ^ Permanent black')
    ap.add_argument('--long-ratio', type=float, default=None,
                    help='LEGACY: width:height of the longest panel (overrides --ve). '
                         'Default: size by VE like --stack.')
    ap.add_argument('--ve', type=float, default=None,
                    help='Vertical exaggeration for per-section figures '
                         '(default = --stack-ve, 3): same sizing as --stack.')
    ap.add_argument('--nproc', type=int, default=1,
                    help='Parallelise section plotting across N cores (default 1)')
    ap.add_argument('--strips', nargs='?', const=_STRIPS_DEFAULT, default=_STRIPS_DEFAULT,
                    help='Surface-geology strips above the top panel: comma list of '
                         'litho,domain,boundaries,names (default: domain,boundaries,names - '
                         'the index-map colours plus crustal-boundary arrows)')
    ap.add_argument('--no-strips', action='store_true', help='plain sections, no geology strips')
    ap.add_argument('--geol-track', action='store_true',
                    help='add a combined greenstone-fill + gold-deposit track above the domain bar. '
                         'Greenstone from --greenstone-npz; gold projected onto the profile within '
                         '--gold-swath km. Tags outputs .gold-gs')
    ap.add_argument('--greenstone-npz', default='data/geology/yilgarn_units.npz',
                    help='Yilgarn rock-type polygons for the greenstone band (default data/geology/)')
    ap.add_argument('--gold-swath', type=float, default=50.0,
                    help='project gold deposits within this many km of the profile line (default 50)')
    ap.add_argument('--gold-style', default='markers', choices=['markers', 'lines'],
                    help='markers = log-area circles sized by endowment (default); '
                         'lines = thin vertical ticks, no size (cleaner where deposits are dense)')
    ap.add_argument('--craton-line', action='store_true',
                    help='draw a thick red line along the top of the domain bar where the profile '
                         'is inside the Yilgarn/Pilbara craton, with red margin ticks')
    ap.add_argument('--craton-outline', default='data/geology/yilgarn_craton_outline.txt',
                    help='craton outline (lon lat, blank-line segments) for --craton-line')
    ap.add_argument('--vpvs-track', default=None, metavar='HK',
                    help='add a colour-coded crustal Vp/Vs bar above the panels (like the domain / '
                         'Au-greenstone bars), sampled (IDW) along the profile from an H-K table '
                         "k_new column, e.g. --vpvs-track data/stations/HK.final.2026.merged.txt")
    ap.add_argument('--vpvs-lim', nargs=2, type=float, default=(1.68, 1.86), metavar=('LO', 'HI'),
                    help='Vp/Vs colour range for --vpvs-track (default 1.68 1.86; ends saturate)')
    ap.add_argument('--vpvs-max-dist', type=float, default=0.8,
                    help='blank the Vp/Vs bar where no station is within this many degrees (default 0.8)')
    ap.add_argument('--vpvs-cmap', default='gmt_polar_r', metavar='NAME',
                    help='colormap for the --vpvs-track bar (default gmt_polar_r = GMT polar reversed, '
                         "red low -> white -> blue high, so blue = high value); discretised to --ncolors. "
                         "e.g. 'gmt_polar', 'bwr', 'RdBu', 'jet_r'.")
    ap.add_argument('--fault', action='append', default=[], metavar='LABEL:LON,LAT',
                    help='mark a fault with a red down-arrow on the top strip (same style as the '
                         'boundary crossings, but red + thick) at the profile point nearest '
                         "LON,LAT, with a red LABEL above. Repeatable. e.g. --fault 'Ida Fault:121.3,-29.5'")
    ap.add_argument('--highlight-boundary', dest='highlight_boundary', action='append', default=[],
                    metavar='NPZNAME:LABEL',
                    help='draw a named crustal boundary (from wa_crustal_boundaries.npz) as the '
                         'red fault arrow at its TRUE per-profile crossing, and hide its normal '
                         "black arrow. LABEL optional. Repeatable. e.g. "
                         "--highlight-boundary 'Youanmi east:Ida Fault' (the Ida Fault margin).")
    ap.add_argument('--gold-file', default=None,
                    help='deposit list: whitespace/comma columns "lon lat [endowment] [kind]". '
                         'If omitted, reads <deposit-dir>/<kind>_deposits.mat (as the maps do).')
    ap.add_argument('--deposit-dir', default=None,
                    help='dir holding <kind>_deposits.mat (gold_deposits.mat ...); default repo matlab/')
    ap.add_argument('--deposits', default='gold',
                    help='deposit kinds for the track, comma-separated (default gold; '
                         'gold,copper,lithium,nickel)')
    ap.add_argument('--index-only', action='store_true',
                    help='with --index-map: draw only the map, do not re-plot any sections')
    ap.add_argument('--domain-colours', default='map2022',
                    choices=['map2022', 'tectcolour', 'palette'],
                    help='domain strip colours: map2022 = GSWA simplified tectonic map 2022 '
                         '(default); tectcolour = shapefile TECTCOLOUR; palette = pastel')
    ap.add_argument('--bnd-lines', action='store_true',
                    help='also draw boundary crossings as dashed lines through the model '
                         'panels (default: down-arrows on the domain strip only)')
    ap.add_argument('--litho-npz', default=None, help='override lithology npz')
    ap.add_argument('--index-tick', type=float, default=100,
                    help='--index-map distance tick/label spacing in km (default 100)')
    ap.add_argument('--stations', default=None,
                    help='--index-map station file (default config.STATIONS; '
                         'columns from config.STATION_COLS)')
    ap.add_argument('--no-stations', action='store_true', help='--index-map without stations')
    ap.add_argument('--footprint', action='store_true',
                    help='--index-map: draw the WA Array coverage footprint (outline + veil)')
    ap.add_argument('--footprint-npz', default=None, help='override config.FOOTPRINT_NPZ')
    ap.add_argument('--footprint-outline', default=None, help='override config.FOOTPRINT_OUTLINE')
    ap.add_argument('--footprint-veil', type=float, default=0.5,
                    help='veil opacity outside the footprint on the index map (0 = outline only)')
    ap.add_argument('--index-style', default='map2022',
                    choices=['map2022', 'gswa', 'craton', 'plain'],
                    help='--index-map colours: map2022 = 10M units in the GSWA 2022 simplified '
                         'tectonic map colours + major crustal boundaries (default); '
                         'gswa = GSWA 500k unit colours; '
                         'craton = tectonic domains + Yilgarn/Pilbara granite-greenstone; '
                         'plain = tectonic domains only')
    ap.add_argument('--index-proj', default='albers', choices=['albers', 'plate'],
                    help='--index-map projection: albers = as the GSWA 2022 simplified tectonic '
                         'map (Albers, 121E, 17.5S/31.5S; default); plate = plain lon/lat')
    ap.add_argument('--index-legend', default='upper left',
                    choices=['upper left', 'upper right', 'lower left', 'lower right', 'none'],
                    help='--index-map (map2022 style): corner for the GSWA 2022 map legend '
                         '(rock type x age chart); none = omit')
    ap.add_argument('--index-labels', default='major', choices=['major', 'all', 'none'],
                    help='--index-map (map2022 style): unit names at the GSWA map positions - '
                         'major = cratons, terranes, orogens, main basins (default); all')
    ap.add_argument('--label-scale', type=float, default=1.4,
                    help='unit label size relative to the A4 GSWA map (default 1.4)')
    ap.add_argument('--index-plain', action='store_true',
                    help='--index-map: tectonic colours only (no craton granite/greenstone)')
    ap.add_argument('--tect-npz',  default=None, help='override tectonics npz')
    ap.add_argument('--bnd-npz',   default=None, help='override crustal-boundaries npz')
    ap.add_argument('--out-dir',   default='figures/xsections')
    args = ap.parse_args()
    if args.abs_skip:
        global ABS_SKIP
        ABS_SKIP = tuple(args.abs_skip)
    global XI_SQ, XI_LIM, XI_TITLE
    if args.xi_squared:
        XI_SQ, XI_LIM = True, (0.88, 1.12)
        XI_TITLE = 'xi = (Vsh/Vsv)$^2$ (blue = Vsh>Vsv)'
    if args.xi_lim:                          # explicit xi colour range, e.g. 0.95 1.05
        XI_LIM = (float(args.xi_lim[0]), float(args.xi_lim[1]))
    MAX_NODE_DIST = args.max_node_dist

    global _GEO
    geo_args = None
    if args.no_strips:
        args.strips = None
    if args.index_only:
        args.index_map = True
    _deposits = _deposit_points(args) if args.geol_track else None
    _gs_npz = args.greenstone_npz if args.geol_track else None
    _craton = args.craton_outline if args.craton_line else None
    _faults = _parse_faults(args.fault)
    _highlight = _parse_highlight(args.highlight_boundary)
    geol_tag = '.gold-gs' if args.geol_track else ''
    if args.strips and not args.index_only:
        geo_args = (args.strips, args.litho_npz, args.tect_npz, args.bnd_npz, args.bnd_lines,
                    args.domain_colours, _gs_npz, _deposits, args.gold_swath, _craton, _faults,
                    _highlight, args.vpvs_track, tuple(args.vpvs_lim), args.vpvs_max_dist,
                    args.vpvs_cmap, args.ncolors)
        _GEO = _load_geology(*geo_args)

    print(f"Loading {args.fvs} ...")
    d = _load_fvs(args.fvs, args.vel)
    plot_section._fvsname = args.fvs   # for Viso/Vsv label detection
    _xi2_d = None; _xi2_label = 'model 2'
    if args.xi2_fvs:                    # second xi model for the comparison panel
        _p2, _, _lab2 = args.xi2_fvs.partition(':')
        _xi2_d = _load_fvs(_p2.strip(), 'model')
        _xi2_label = _lab2.strip() or os.path.basename(_p2.strip())
        print(f"  xi2 (compare): {_p2.strip()}  [{_xi2_label}]")
    if args.var_ref == 'yilgarn' and not args.ref_domain:
        args.ref_domain = 'yilgarn'    # --var-ref yilgarn == --ref-domain yilgarn
    if args.ref_domain:                # dln reference = a tectonic domain's mean
        from domains import Domains
        D = Domains(_resolve_npz('TECTONICS_NPZ', 'wa_tectonics.npz', args.tect_npz))
        _REF_NODES = D.node_mask(d, args.ref_domain)
        _REF_NAME, _REF_TAG = D[args.ref_domain]['label'], f'.ref-{args.ref_domain}'
        print(f"  dln reference: {_REF_NAME} mean at each depth ({_REF_NODES.sum()} nodes, "
              f"Ndat >= 35)")
        if not _REF_NODES.any():
            sys.exit(f"no model nodes in domain '{args.ref_domain}'")
    global _REF
    ref_args = None
    if args.ref_fvs:
        ref_args = (args.ref_fvs, args.ref_vel, args.ref_label, args.d_max)
        _REF = _load_ref(*ref_args)
    # Optional Moho interpolator
    plot_section._moho_interp = None
    plot_section._moho_ccp_interp = None
    if args.moho:
        try:
            plot_section._moho_interp = _load_moho_interp(args.moho)
            print(f'  Moho overlay loaded: {args.moho}')
        except Exception as e:
            print(f'  Moho load failed: {e}')
    if args.moho_ccp:
        try:
            plot_section._moho_ccp_interp = _load_moho_interp(args.moho_ccp)
            print(f'  CCP Moho overlay loaded: {args.moho_ccp}')
        except Exception as e:
            print(f'  CCP Moho load failed: {e}')
    _set_ccp_stack(args)
    print(f"  {len(d['Lon'])} nodes, {len(d['z'])} depth levels "
          f"({d['z'][0]:.1f}-{d['z'][-1]:.1f} km)")

    moho_mask = None if args.no_moho_mask else args.moho_mask_alpha
    kw = dict(ds_deg=args.ds, ncolors=args.ncolors, d_max=args.d_max,
              sigma_vsv=args.sigma_vsv, clim_rel=args.clim_rel,
              out_dir=args.out_dir, moho_mask=moho_mask,
              xi_panel=args.xi_panel, xi_ref=args.xi_ref, clim_xi=args.clim_xi,
              div_gap=args.div_gap, div_white=args.div_white,
              clim_diff=args.clim_diff, vlim=args.vlim, var_ref=args.var_ref,
              smooth=(args.field_style == 'smooth'), geol_tag=geol_tag, vs_smooth=args.vs_smooth,
              xi_gap=args.xi_gap, xi_cmap=args.xi_cmap, dvsv_cmap=args.dvsv_cmap,
              xi2_d=_xi2_d, xi2_label=_xi2_label, xi_label=args.xi_label,
              xi_diff_lim=args.xi_diff_lim,
              panels=([p for p in args.panels.split(',') if p.strip()] if args.panels else None))

    # --- Assemble the section list (ginput / load / sections / start-end) ---
    sections = []
    if args.ginput:                       # --load-sections then only shows the current lines
        ref = _load_sections(args.load_sections) if args.load_sections else None
        sta = None
        try:
            import config
            if os.path.isfile(args.stations or config.STATIONS):
                sta = _load_stations(args.stations or config.STATIONS, config.STATION_COLS)
        except Exception:
            pass
        sections = _ginput_sections(
            d, ref_sections=ref, stations=sta,
            tect_npz=_resolve_npz('TECTONICS_NPZ', 'wa_tectonics.npz', args.tect_npz),
            bnd_npz=_resolve_npz('BOUNDARIES_NPZ', 'wa_crustal_boundaries.npz', args.bnd_npz))
        if not sections:
            sys.exit("No sections picked.")
    elif args.load_sections:
        sections = _load_sections(args.load_sections)
        print(f"Loaded {len(sections)} sections from {args.load_sections}")
    elif args.sections:
        for k, sec in enumerate(args.sections):
            parts = sec.split('/')
            if len(parts) < 2:
                print(f"WARNING: skipping malformed '{sec}'"); continue
            lat1, lon1 = map(float, parts[0].split(','))
            lat2, lon2 = map(float, parts[1].split(','))
            lab = parts[2] if len(parts) > 2 else f'sec{k+1}'
            sections.append((lat1, lon1, lat2, lon2, lab))
    elif args.start and args.end:
        lat1, lon1 = map(float, args.start.split(','))
        lat2, lon2 = map(float, args.end.split(','))
        sections.append((lat1, lon1, lat2, lon2, args.label))
    else:
        ap.error('Provide --ginput, --load-sections, --sections, or --start/--end')

    # Auto-number any unlabeled sections consistently as sec1, sec2, ...
    sections = [(la1,lo1,la2,lo2, lab if lab and not lab.startswith('section')
                 else f'sec{i+1}')
                for i,(la1,lo1,la2,lo2,lab) in enumerate(sections)]

    if args.save_sections:
        _save_sections(sections, args.save_sections)

    # --- Index map (all lines on geology, numbered, distance ticks) ---
    if args.index_map:
        sta = None
        if not args.no_stations:
            try:
                import config
                sta_path, sta_cols = args.stations or config.STATIONS, config.STATION_COLS
            except Exception:
                sta_path, sta_cols = args.stations, (2, 1)
            if sta_path and os.path.isfile(sta_path):
                sta = _load_stations(sta_path, sta_cols)
                print(f"  stations: {len(sta[0])} <- {sta_path}")
            else:
                print(f"  stations: file not found ({sta_path}) - none drawn")
        _poly = None
        if args.load_polylines:
            _p = np.load(args.load_polylines, allow_pickle=True)
            _poly = {str(_p['short'][i]): (np.asarray(_p['lon'][i], float),
                                           np.asarray(_p['lat'][i], float))
                     for i in range(len(_p['short']))}
            print(f"  index map: drawing {len(_poly)} lines along their picked polylines")
        from plot_wa_map import plot_map, _load_station_csv   # map plotter lives here now
        _overlays = []
        for _spec in args.map_stations:
            if len(_spec) < 4:
                ap.error(f"--map-stations needs CSV COLOUR MARKER LABEL [EDGE]; got {_spec}")
            csv_path, col, mark, lab = _spec[0], _spec[1], _spec[2], _spec[3]
            _edge = _spec[4] if len(_spec) > 4 else 'w'
            if not os.path.isfile(csv_path):
                print(f"  map-stations: file not found ({csv_path}) - skipped"); continue
            olon, olat, _ = _load_station_csv(csv_path)   # _load_station_csv returns codes too
            _overlays.append((olon, olat, col, mark, 22, lab, _edge))
            print(f"  map-stations: {len(olon)} '{lab}' <- {csv_path} ({col} {mark}, edge {_edge})")
        plot_map(sections, args.out_dir, d=d, tick_km=args.index_tick, stations=sta,
                 polylines=_poly, overlays=_overlays, geo=_GEO,
                 style='plain' if args.index_plain else args.index_style,
                 footprint=_index_footprint(args), footprint_veil=args.footprint_veil,
                 tect_npz=_resolve_npz('TECTONICS_NPZ', 'wa_tectonics.npz', args.tect_npz),
                 bnd_npz=_resolve_npz('BOUNDARIES_NPZ', 'wa_crustal_boundaries.npz',
                                      args.bnd_npz),
                 projection=args.index_proj, legend=args.index_legend,
                 labels=args.index_labels, label_scale=args.label_scale,
                 litho_npz=None if args.index_plain else
                 _resolve_npz('LITHOLOGY_NPZ', 'wa_lithology.npz', args.litho_npz))
        if args.index_only:
            return

    # Stacked single-field figure (one page, all sections) — then return
    if args.stack:
        bad = [f for f in args.stack.split(',') if f.strip() and f.strip() not in STACK_FIELDS]
        if bad:
            ap.error(f"--stack: unknown field(s) {bad}; choose from {', '.join(STACK_FIELDS)}")
        zmask=(d['z']<=args.d_max) if args.d_max else np.ones(len(d['z']),bool)
        rmean=_node_ref(d['Vsv'][:,zmask])
        kw = dict(moho_file=args.moho, moho_ccp_file=args.moho_ccp,
                  smooth=(args.field_style == 'smooth'),
                  ds_deg=args.ds, d_max=args.d_max, ref_mean=rmean,
                  ve=args.stack_ve, vmin=args.vmin, vmax=args.vmax, ncolors=args.ncolors,
                  moho_mask=None if args.no_moho_mask else args.moho_mask_alpha,
                  xi_ref=args.xi_ref, clim_xi=args.clim_xi, div_gap=args.div_gap,
                  div_white=args.div_white, clim_vpvs=args.clim_vpvs, clim_diff=args.clim_diff)
        if args.stack_split == 'none':
            plot_stack(d, sections, args.stack, args.out_dir, **kw)
            return
        groups, longest = _split_by_orientation(sections)
        for key, (secs, title) in groups.items():
            if secs:
                print(f"  {title}: {', '.join(s[4] for s in secs)}")
                plot_stack(d, secs, args.stack, args.out_dir, maxd_km=longest,
                           tag=f'.{key}', group=title.lower(), **kw)
        return

    # Longest section sets the scale (drawn 2:1 at landscape width); all sections
    # then share the same km/inch and panel height.
    maxd = 0.0
    for (la1,lo1,la2,lo2,_lab) in sections:
        _,_,dd = _great_circle_path(la1,lo1,la2,lo2, args.ds)
        maxd = max(maxd, dd[-1])
    zb = d['z'][d['z'] <= args.d_max][-1] if args.d_max else d['z'][-1]
    ve = args.ve if args.ve else args.stack_ve
    km_per_in, panel_h_in = _section_scale(maxd, zb, ve)
    if args.long_ratio:                      # legacy: longest drawn long_ratio:1
        panel_h_in = PAGE_W_IN / args.long_ratio
    print(f'Longest {maxd:.0f} km → {km_per_in:.0f} km/in, panel height {panel_h_in:.2f} in '
          f'(VE {km_per_in / (zb / panel_h_in):.1f}x)')

    # Full-model mean Vsv per depth → dln reference (regional, consistent)
    zmask = (d['z'] <= args.d_max) if args.d_max else np.ones(len(d['z']), bool)
    Vsv_all = d['Vsv'][:, zmask]
    ref_mean = _node_ref(Vsv_all)

    if args.var_ref == 'allprofiles':        # pooled mean of every profile in the location file
        av, axi = [], []
        has_xi = 'Xi' in d
        Xi_all = d['Xi'][:, zmask] if has_xi else None
        for (la1, lo1, la2, lo2, _lab) in sections:
            pla, plo, _ = _great_circle_path(la1, lo1, la2, lo2, args.ds)
            av.append(_sample_profile(d['Lon'], d['Lat'], Vsv_all, plo, pla))
            if has_xi:
                axi.append(_sample_profile(d['Lon'], d['Lat'], Xi_all, plo, pla))
        kw['allprof_vsv_ref'] = np.nanmean(np.vstack(av), axis=0)
        if axi:
            kw['allprof_xi_ref'] = np.nanmean(np.vstack(axi), axis=0)
        print(f"  --var-ref allprofiles: pooled mean over {len(sections)} profile(s)")

    kw['max_dist']   = maxd
    kw['km_per_in']  = km_per_in
    kw['panel_h_in'] = panel_h_in
    kw['long_ratio'] = args.long_ratio or (PAGE_W_IN / panel_h_in)
    kw['ref_mean']   = ref_mean

    # --- Plot each section (optionally in parallel) ---
    if args.nproc > 1 and len(sections) > 1:
        from multiprocessing import Pool
        from functools import partial
        print(f"Plotting {len(sections)} sections on {args.nproc} cores ...")
        worker = partial(_section_worker, fvs=args.fvs, moho=args.moho, kw=kw,
                         geo_args=geo_args, vel=args.vel, ref_args=ref_args,
                         moho_ccp=args.moho_ccp, ccp_args=args)
        with Pool(args.nproc) as pool:
            pool.map(worker, sections)
    else:
        for sec in sections:
            plot_section(d, *sec, **kw)


def _section_worker(sec, fvs, moho, kw, geo_args=None, vel='model', ref_args=None,
                    moho_ccp=None, ccp_args=None):
    """Standalone worker for parallel section plotting (reloads Fvs per process)."""
    import numpy as np
    global _GEO
    if geo_args and _GEO is None:     # not inherited (spawn/forkserver start method)
        _GEO = _load_geology(*geo_args)
    global _REF
    if ref_args and _REF is None:
        _REF = _load_ref(*ref_args)
    d = _load_fvs(fvs, vel)
    plot_section._fvsname = fvs
    plot_section._moho_interp = None
    plot_section._moho_ccp_interp = None
    if moho:
        try: plot_section._moho_interp = _load_moho_interp(moho)
        except Exception: pass
    if moho_ccp:
        try: plot_section._moho_ccp_interp = _load_moho_interp(moho_ccp)
        except Exception: pass
    if ccp_args is not None:          # reload the CCP cube in this process
        _set_ccp_stack(ccp_args)
    plot_section(d, *sec, **kw)


if __name__ == '__main__':
    main()
