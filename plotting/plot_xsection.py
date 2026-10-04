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
      --strips                                  # = litho,boundaries
  python3 plot_xsection.py ... --strips litho,domain,boundaries,names
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
_STRIPS_DEFAULT = 'litho,domain,boundaries,names'   # geology strips are on unless --no-strips


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


def _load_geology(spec, litho_npz=None, tect_npz=None, bnd_npz=None):
    import xsection_strips as xs
    want = {w.strip() for w in spec.split(',') if w.strip()}
    bad = want - _STRIP_ITEMS
    if bad:
        raise SystemExit(f"--strips: unknown item(s) {sorted(bad)}; "
                         f"choose from {sorted(_STRIP_ITEMS)}")
    g = dict(litho=None, tect=None, bnd=None, names='names' in want)
    jobs = (('litho', 'litho', 'LITHOLOGY_NPZ', 'wa_lithology.npz', litho_npz,
             xs.PolygonIndex.from_npz),
            ('domain', 'tect', 'TECTONICS_NPZ', 'wa_tectonics.npz', tect_npz,
             lambda p: xs.PolygonIndex.from_npz(p, class_key='names')),
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
    return g


def _strip_height_in(g):
    """Vertical room the strips (and rotated boundary names) need above a panel."""
    if g is None:
        return 0.0
    h = 0.17 * ((g['litho'] is not None) + (g['tect'] is not None))
    if g['names'] and g['bnd'] is not None:
        h += 0.55
    return h


def _add_geology(top_ax, axes, plon, plat, dist):
    if _GEO is None:
        return None
    import xsection_strips as xs
    res = xs.add_profile_strips(top_ax, axes, plon, plat, dist,
                                litho=_GEO['litho'], tectonic=_GEO['tect'],
                                boundaries=_GEO['bnd'], label_boundaries=_GEO['names'])
    if res.crossings:
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


# ── Profile interpolation ─────────────────────────────────────────────────────
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
            rbf = RBFInterpolator(pts[row_all_finite], data3d[row_all_finite],
                                  kernel='thin_plate_spline', smoothing=0.01)
            out[:, :] = rbf(qpts)        # evaluates all nz columns in one call
            # Depths where some needed node was NaN: those columns may still be
            # fine (we used all-finite rows); only redo columns that are all-nan.
            bad_cols = ~np.isfinite(out).any(axis=0)
            if not bad_cols.any():
                return out
        except Exception:
            pass

    # Fallback: per-depth (only for columns not filled above)
    from scipy.interpolate import griddata
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
CMAP_ABS = 'Spectral'

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
# Diverging maps (RdBu, and Spectral for absolute velocity) skip a band around their
# pale middle (white / yellow) so values near the centre keep a colour; DIV_GAP = half-width of the skipped band
# (0 = classic RdBu through white). Set from --div-gap.
DIV_GAP = 0.15
DIVERGING = ('RdBu', 'Spectral')     # Spectral: skips the pale-yellow middle


DIV_WHITE = 2       # neutral (centre-colour) levels kept in the middle; --div-white


def _cmap(name, n=32):
    if name in DIVERGING and (DIV_GAP > 0 or DIV_WHITE > 0):
        from matplotlib.colors import ListedColormap
        base = matplotlib.colormaps[name]
        k = max(0, min(DIV_WHITE, n - 2))
        if (n - k) % 2:                      # keep the neutral band centred
            k += 1
        side = (n - k) // 2
        lo = base(np.linspace(0.0, 0.5 - DIV_GAP, side))
        hi = base(np.linspace(0.5 + DIV_GAP, 1.0, side))
        mid = np.repeat(np.asarray(base(0.5))[None, :], k, axis=0)
        return ListedColormap(np.vstack([lo, mid, hi]), name=f'{name}_gap')
    try:
        return matplotlib.colormaps[name].resampled(n)
    except Exception:
        return plt.cm.get_cmap(name, n)


# ── xi reference for dlnXi ────────────────────────────────────────────────────
def _xi_ref(Xi, mode='depth'):
    """Reference xi for dlnXi = (xi/ref - 1)*100, from ALL model nodes.
    depth : mean xi of the whole model at each depth (like dVsv) - removes the
            depth trend, shows lateral variation;
    global: one mean over the whole model, all depths - note dlnXi is then just
            xi linearly rescaled (same picture as the xi panel, re-centred)."""
    if mode == 'global':
        return np.full(Xi.shape[1], np.nanmean(Xi))
    return np.nanmean(Xi, axis=0)


def _xi_ref_label(mode):
    return 'whole-model mean' if mode == 'global' else 'model mean at each depth'


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


# ── Single cross-section ──────────────────────────────────────────────────────
def plot_section(d, lat1, lon1, lat2, lon2, label,
                 ds_deg=0.08, ncolors=16, d_max=None,
                 sigma_vsv=1.5, clim_rel=6.0,
                 out_dir='figures/xsections',
                 max_dist=None, km_per_in=None, panel_h_in=None, ref_mean=None,
                 long_ratio=3.0, moho_mask=0.5, xi_panel='xi', xi_ref='depth',
                 clim_xi=5.0, div_gap=None, div_white=None):
    global DIV_GAP, DIV_WHITE
    if div_gap is not None:
        DIV_GAP = div_gap
    if div_white is not None:
        DIV_WHITE = div_white
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
    moho_prof = None
    _mi = getattr(plot_section, '_moho_interp', None)

    print(f"  Profile {label}: ({lat1:.2f},{lon1:.2f}) -> ({lat2:.2f},{lon2:.2f})")
    plat, plon, dist = _great_circle_path(lat1, lon1, lat2, lon2, ds_deg)
    print(f"  Length: {dist[-1]:.0f} km  |  {len(dist)} sample points")
    if _mi is not None:
        moho_prof = np.array([float(np.ravel(_mi(la, lo))[0]) for la, lo in zip(plat, plon)])

    print("  Sampling ...")
    vsv_p = _sample_profile(Lon, Lat, Vsv, plon, plat)
    err_p = _sample_profile(Lon, Lat, Vsv_err, plon, plat) if Vsv_err is not None else None
    xi_p  = _sample_profile(Lon, Lat, Xi, plon, plat) if Xi is not None else None

    if ref_mean is None:
        ref_mean = np.array([np.nanmean(Vsv[:, k]) for k in range(Vsv.shape[1])])
    dln_p = (vsv_p - ref_mean[None, :]) / ref_mean[None, :] * 100.0

    D, Z = np.meshgrid(dist, z, indexing='ij')

    fin_v = vsv_p[np.isfinite(vsv_p)]
    vmed  = np.nanmedian(fin_v); vstd = np.nanstd(fin_v)
    vmin_v = vmed - sigma_vsv*vstd; vmax_v = vmed + sigma_vsv*vstd

    have_xi  = xi_p  is not None and np.nanstd(xi_p) > 1e-4
    have_err = err_p is not None
    is_zt = 'ZT' in os.path.basename(getattr(plot_section, '_fvsname', '')) or have_xi
    vname = 'Viso' if is_zt else 'Vsv'

    dxi_p = None
    if have_xi and xi_panel in ('dxi', 'both'):
        xref = _xi_ref(Xi, xi_ref)
        dxi_p = (xi_p / xref[None, :] - 1.0) * 100.0
    rows = ['abs', 'rel']
    if have_xi and xi_panel in ('xi', 'both'): rows.append('xi')
    if dxi_p is not None: rows.append('dxi')
    if have_err: rows.append('err')
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
        im = ax.pcolormesh(D, Z, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_ylim(zbot, 0); ax.set_xlim(0, xmax)
        ax.set_ylabel('Depth (km)', fontsize=9); ax.tick_params(labelsize=8)
        pos = ax.get_position()           # colorbar in its own axes: panel keeps its size
        cax = fig.add_axes([pos.x1 + 0.10 / fig_w, pos.y0, 0.16 / fig_w, pos.height])
        cb = fig.colorbar(im, cax=cax, extend=extend)
        cb.set_label(clabel, fontsize=8); cb.ax.tick_params(labelsize=7)
        if title: ax.set_title(title, fontsize=9, loc='left')
        if moho_prof is not None:
            _draw_moho(ax, dist, moho_prof, zbot, moho_mask, lw=1.1)
        return im

    r = 0
    _pcolor(axes[r], vsv_p, _cmap(CMAP_ABS, ncolors), vmin_v, vmax_v,
            f'{vname} (km/s)', title=f'{vname} (km/s)'); r += 1
    _pcolor(axes[r], dln_p, _cmap('RdBu', ncolors), -clim_rel, clim_rel,
            f'd{vname} (%)', title=f'd{vname} (%) vs model mean'); r += 1
    if have_xi and xi_panel in ('xi', 'both'):
        _pcolor(axes[r], xi_p, _cmap('RdBu', ncolors), 0.90, 1.10,
                'xi', title='xi = Vsh/Vsv (blue = Vsh>Vsv)'); r += 1
    if dxi_p is not None:
        _pcolor(axes[r], dxi_p, _cmap('RdBu', ncolors), -clim_xi, clim_xi,
                'dlnXi (%)', title=f'dlnXi (%) vs {_xi_ref_label(xi_ref)}'); r += 1
    if have_err:
        vmax_e = np.nanpercentile(err_p[np.isfinite(err_p)], 90)
        _pcolor(axes[r], err_p, _cmap('YlOrRd', ncolors), 0, vmax_e,
                f'{vname} IQR/2', extend='max',
                title='Model uncertainty (posterior IQR/2)'); r += 1

    axes[-1].set_xlabel('Distance along profile (km)', fontsize=9)
    axes[-1].set_xticks(np.arange(0, xmax+1, 100))

    geo = _add_geology(axes[0], axes, plon, plat, dist)
    if geo is not None:
        _geology_legend(fig, [geo], axes[-1], ncol=4 if panel_w_in > 5 else 2)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'xsection.{label}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.png','.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out} (+ .pdf)")



def _ginput_sections(d, moho_file=None):
    """Show Ndat map, let user pick pairs of (start, end) points interactively.
    Returns list of (lat1,lon1,lat2,lon2,label). Needs an interactive display."""
    import matplotlib
    # Try to switch to an interactive backend; Agg (headless) can't do ginput.
    for bk in ['TkAgg', 'Qt5Agg', 'QtAgg', 'GTK3Agg']:
        try:
            matplotlib.use(bk, force=True)
            break
        except Exception:
            continue
    import matplotlib.pyplot as plt2
    if matplotlib.get_backend().lower() == 'agg':
        print("ERROR: no interactive backend available (headless?). "
              "Run on a machine with a display, or use --start/--end / --sections.")
        return []

    Lon = d['Lon']; Lat = d['Lat']
    Ndat = d['Ndat'].astype(float) if 'Ndat' in d else np.ones(len(Lon))

    # Build a GeoAxes (PlateCarree → clicks come back as lon/lat) with ocean mask
    _geo = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        tr = ccrs.PlateCarree()
        fig2 = plt2.figure(figsize=(10, 10))
        ax2 = fig2.add_axes([0.06,0.06,0.88,0.88], projection=tr)
        _geo = True
    except Exception:
        fig2, ax2 = plt2.subplots(figsize=(10, 10)); tr = None

    gkw = {'transform': tr} if tr else {}
    # Tectonic background (faint) so sections are picked over geology
    try:
        from wa_basemap import add_tectonic_background, add_tectonic_outlines
        add_tectonic_background(ax2, alpha=0.40, zorder=0, transform=tr)
        add_tectonic_outlines(ax2, lw=0.4, alpha=0.6, zorder=3, major_only=True, transform=tr)
        _have_tect = True
    except Exception as _e:
        print(f"(tectonic background unavailable: {_e})")
        _have_tect = False
    # Coverage dots (semi-transparent) so you also see data density
    sc = ax2.scatter(Lon, Lat, c=Ndat, cmap='Greys', s=26, vmin=0, vmax=Ndat.max(),
                     alpha=0.5, zorder=4, edgecolor='none', **gkw)
    # Grey ocean mask + coastline ON TOP (so offshore/uncovered areas are masked)
    if _geo:
        ax2.add_feature(cfeature.OCEAN, facecolor='lightgrey', zorder=5)
        ax2.coastlines('50m', linewidth=0.8, color='k', zorder=6)
        ax2.set_extent([Lon.min()-1.0, Lon.max()+1.0,
                        Lat.min()-1.0, Lat.max()+1.0], crs=tr)
    else:
        if not _have_tect:
            plt2.colorbar(sc, ax=ax2, label='N measurements', shrink=0.7)
        ax2.set_aspect('equal'); ax2.grid(alpha=0.25)
    ax2.set_title('Click section endpoints: START, END, START, END ...\n'
                  'over tectonic domains; middle-click undoes; Enter when done',
                  fontsize=10)
    plt2.tight_layout(); plt2.show(block=False)

    print("\nClick pairs: start, end, start, end ... Enter to finish.")
    pts = plt2.ginput(n=-1, timeout=0, show_clicks=True,
                      mouse_add=1, mouse_pop=2, mouse_stop=3)
    plt2.close(fig2)

    if len(pts) < 2:
        print("Need at least 2 points."); return []
    if len(pts) % 2 != 0:
        print(f"Odd number of points ({len(pts)}); dropping the last unpaired click.")
        pts = pts[:-(1)]

    sections = []
    for i in range(0, len(pts)-1, 2):
        lon1, lat1 = pts[i]
        lon2, lat2 = pts[i+1]
        label = f'sec{i//2+1}'
        sections.append((lat1, lon1, lat2, lon2, label))
        print(f"  {label}: ({lat1:.2f},{lon1:.2f}) -> ({lat2:.2f},{lon2:.2f})")
    return sections

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


def _plot_index_map(d, sections, out_dir, label='sections', moho=None,
                    litho_npz=None, tick_km=100, stations=None, style='gswa',
                    footprint=None, footprint_veil=0.5):
    """Section lines on the WA tectonic map, with distance ticks + labels every
    tick_km. If litho_npz is given, the GSWA 500k granite + mafic/greenstone/
    granite-greenstone polygons are drawn inside the Yilgarn and Pilbara cratons."""
    import matplotlib.pyplot as plt
    Lon = d['Lon']; Lat = d['Lat']
    try:
        from plot_depth_slice import _make_ax, _add_states_ocean, _HAS_CARTOPY
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(13, 13))
        ax = _make_ax(fig, [0.06, 0.05, 0.88, 0.90])
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
        add_tectonic_background(ax, alpha=0.5, zorder=0, transform=tr, clip_box=cb)
        litho_drawn = None
        if style == 'gswa':
            add_gswa_units(ax, litho_npz, transform=tr, clip_box=cb, zorder=1)
        elif style == 'craton' and litho_npz:
            litho_drawn = add_craton_geology(ax, litho_npz, zorder=1, transform=tr,
                                             clip_box=cb)
        add_tectonic_outlines(ax, lw=0.4, alpha=0.6, zorder=3, major_only=True,
                              transform=tr, clip_box=cb)
        _tect = True
    except Exception as e:
        print(f"(tectonic bg unavailable: {e})")
        ax.scatter(Lon, Lat, c='0.7', s=10, zorder=1,
                   **({'transform': tr} if tr else {})); _tect = False

    lkw = {'transform': tr} if tr else {}
    if footprint is not None:          # outline + veil, no zoom (sections may extend past it)
        footprint.focus(ax, tr, veil_alpha=footprint_veil, zoom=False,
                        veil_zorder=4.4)     # over the geology, under stations + sections
    n_sta = 0
    if stations is not None:            # above the ocean mask (5), below section lines
        slon, slat = stations
        n_sta = len(slon)
        ax.scatter(slon, slat, marker='^', s=16, c='k', edgecolors='w',
                   linewidths=0.35, zorder=5.5, **lkw)
    if _GEO is not None and _GEO['bnd'] is not None:
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(_GEO['bnd'].lines, colors='firebrick',
                                         linewidths=0.9, alpha=0.8, zorder=4, **lkw))
    for i, (lat1, lon1, lat2, lon2, lab) in enumerate(sections):
        plat, plon, dist = _great_circle_path(lat1, lon1, lat2, lon2)
        ax.plot(plon, plat, 'w-', lw=3.4, zorder=5, **lkw)   # white halo
        ax.plot(plon, plat, 'k-', lw=2.0, zorder=6, **lkw)
        ax.plot(lon1, lat1, 'ko', ms=6, zorder=7, **lkw)
        ax.plot(lon2, lat2, 'ks', ms=6, zorder=7, **lkw)
        for dkm in np.arange(0, dist[-1], tick_km):
            j = np.argmin(np.abs(dist - dkm))
            ax.plot(plon[j], plat[j], '|', color='k', ms=7, mew=1.2, zorder=7, **lkw)
            ax.annotate(f'{int(dkm)}', (plon[j], plat[j]), fontsize=5.5,
                        ha='center', va='bottom', zorder=8,
                        xytext=(0, 3), textcoords='offset points',
                        bbox=dict(fc='w', ec='none', alpha=0.6, pad=0.3))
        ax.annotate(lab, (lon1, lat1), fontsize=9, fontweight='bold',
                    color='darkred', zorder=8, xytext=(5,5),
                    textcoords='offset points')

    # Grey ocean mask + coastline on top (like the depth-slice panels)
    if _geo:
        _add_states_ocean(ax)
        ax.set_extent([Lon.min()-0.5, Lon.max()+0.5,
                       Lat.min()-0.5, Lat.max()+0.5], crs=tr)
    else:
        ax.set_xlim(Lon.min()-0.5, Lon.max()+0.5)
        ax.set_ylim(Lat.min()-0.5, Lat.max()+0.5); ax.set_aspect('equal')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(f'Cross-section locations ({len(sections)} lines, ticks every {tick_km:g} km'
                 + (f'; {n_sta} stations)' if n_sta else ')'),
                 fontsize=11)
    if _tect and style == 'gswa':
        ax.text(0.01, 0.99, 'Geology: GSWA 1:500k tectonic units, GSWA colours\n'
                '(\u00a9 Geological Survey of Western Australia, CC-BY-4.0)',
                transform=ax.transAxes, ha='left', va='top', fontsize=7, zorder=12,
                bbox=dict(fc='w', ec='0.6', alpha=0.9, pad=3))
    elif _tect:
        tectonic_legend(ax, loc='upper left', fontsize=7,
                        clip_box=(Lon.min()-0.5, Lon.max()+0.5, Lat.min()-0.5, Lat.max()+0.5),
                        exclude=('STATE',),
                        litho=litho_drawn)
    import os
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'xsection_index_map' +
                       ('.fp' if footprint is not None else '') + '.png')   # whole-WA name unchanged
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.png','.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"Index map: {out}")


def _save_sections(sections, path):
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



def plot_stack(d, sections, field, out_dir, moho_file=None, ds_deg=0.08,
               ncolors=16, d_max=None, sigma_vsv=1.5, clim_rel=6.0,
               ref_mean=None, row_h_in=None, page_w_in=9.5, gap_in=0.55,
               ve=DEFAULT_VE, vmin=None, vmax=None, moho_mask=0.5,
               xi_ref='depth', clim_xi=5.0, div_gap=None, div_white=None):
    global DIV_GAP, DIV_WHITE
    if div_gap is not None:
        DIV_GAP = div_gap
    if div_white is not None:
        DIV_WHITE = div_white
    """Stack ONE field for all sections on a single page. Rows = sections, same
    height, width proportional to length, longest spans the full page width.
    ve = vertical exaggeration (depth stretched x this; 1 = true scale, flat;
    higher = taller). Each panel has its own colorbar."""
    import matplotlib.pyplot as plt
    Lon = d['Lon']; Lat = d['Lat']; z = d['z']
    if d_max: zm = z <= d_max; z = z[zm]
    else: zm = np.ones(len(z), bool)
    zbot = z[-1]

    FKEY = {'vsv':'Vsv','dvsv':'Vsv','xi':'Xi','dxi':'Xi','err':'Vsv_err'}
    arrname = FKEY.get(field,'Vsv')
    if arrname not in d:
        print(f"  field {field}: {arrname} not in Fvs"); return
    ARR = d[arrname][:, zm]
    if ref_mean is None and field=='dvsv':
        ref_mean = np.array([np.nanmean(d['Vsv'][:,zm][:,k]) for k in range(ARR.shape[1])])
    if field=='dxi':
        xref = _xi_ref(ARR, xi_ref)
        print(f"  dlnXi reference ({_xi_ref_label(xi_ref)}): "
              f"{np.nanmin(xref):.3f}-{np.nanmax(xref):.3f}")

    _mi = None
    if moho_file:
        try:
            from scipy.interpolate import RectBivariateSpline
            md=np.loadtxt(moho_file, skiprows=11)
            mla=np.unique(md[:,0]); mlo=np.unique(md[:,1])
            _mi=RectBivariateSpline(mla,mlo,md[:,2].reshape(len(mla),len(mlo)),kx=1,ky=1)
        except Exception: pass

    is_zt = 'ZT' in os.path.basename(getattr(plot_section,'_fvsname',''))
    vname = 'Viso' if (is_zt or arrname=='Xi') else 'Vsv'

    prof=[]
    for (la1,lo1,la2,lo2,lab) in sections:
        plat,plon,dist=_great_circle_path(la1,lo1,la2,lo2,ds_deg)
        samp=_sample_profile(Lon,Lat,ARR,plon,plat)
        if field=='dvsv':
            samp=(samp-ref_mean[None,:])/ref_mean[None,:]*100
        elif field=='dxi':
            samp=(samp/xref[None,:]-1.0)*100
        mp=None
        if _mi is not None:
            mp=np.array([float(np.ravel(_mi(a,o))[0]) for a,o in zip(plat,plon)])
        prof.append(dict(lab=lab,dist=dist,data=samp,moho=mp,plat=plat,plon=plon))
    maxd=max(p['dist'][-1] for p in prof)

    # Horizontal scale: longest fills page_w_in. Row height from VE:
    #   km_per_in = maxd/page_w_in ; row_h_in = zbot*ve/km_per_in
    km_per_in, ve_h = _section_scale(maxd, zbot, ve, page_w_in)
    if row_h_in is None:
        row_h_in = ve_h

    if field=='vsv':
        cmap=_cmap(CMAP_ABS,ncolors); clab=f'{vname} (km/s)'; ext='both'
    elif field=='dvsv':
        cmap=_cmap('RdBu',ncolors); clab=f'd{vname} (%)'; ext='both'
    elif field=='xi':
        cmap=_cmap('RdBu',ncolors); clab='xi'; ext='both'
    elif field=='dxi':
        cmap=_cmap('RdBu',ncolors); clab=f'dlnXi (%) vs {_xi_ref_label(xi_ref)}'; ext='both'
    else:
        cmap=_cmap('YlOrRd',ncolors); clab=f'{vname} IQR/2'; ext='max'

    n=len(prof)
    strip_h=_strip_height_in(_GEO)          # room for geology strips above each row
    gap_in=gap_in+strip_h; top_in=0.9+strip_h
    fig_h=n*row_h_in + (n-1)*gap_in + 1.2 + strip_h
    fig_w=page_w_in + 1.8                       # + per-panel colorbar room
    fig=plt.figure(figsize=(fig_w,fig_h))
    fname = {'dxi': 'dlnXi'}.get(field, field.upper())
    fig.suptitle(f'{fname} ({vname}) — {n} sections [depth 0-{zbot:.0f} km, '
                 f'longest {maxd:.0f} km, VE {ve:.0f}x]',
                 fontsize=11, fontweight='bold')

    usable=page_w_in/fig_w
    rh=row_h_in/fig_h
    geo_res=[]
    for i,p in enumerate(prof):
        w=(p['dist'][-1]/maxd)*usable
        y0=1-(top_in/fig_h)-(i+1)*rh-i*(gap_in/fig_h)
        ax=fig.add_axes([0.08, y0, w, rh])
        D,Z=np.meshgrid(p['dist'],z,indexing='ij')
        # colour limits: explicit --vmin/--vmax win; else per-field defaults
        if vmin is not None and vmax is not None:
            vmn,vmx=vmin,vmax
        elif field=='vsv':
            fv=p['data'][np.isfinite(p['data'])]
            med,std=np.nanmedian(fv),np.nanstd(fv); vmn,vmx=med-sigma_vsv*std,med+sigma_vsv*std
        elif field=='err':
            fv=p['data'][np.isfinite(p['data'])]; vmn,vmx=0,np.nanpercentile(fv,90)
        elif field=='dvsv':
            vmn,vmx=-clim_rel,clim_rel
        elif field=='xi':
            vmn,vmx=0.90,1.10
        elif field=='dxi':
            vmn,vmx=-clim_xi,clim_xi
        else:
            vmn,vmx=vmin,vmax
        im=ax.pcolormesh(D,Z,p['data'],cmap=cmap,vmin=vmn,vmax=vmx,shading='auto')
        ax.set_ylim(zbot,0); ax.set_xlim(0,p['dist'][-1])
        ax.set_ylabel('Depth',fontsize=7); ax.tick_params(labelsize=6)
        ax.set_xticks(np.arange(0,p['dist'][-1]+1,200))
        ax.text(0.01,0.90,f"{p['lab']} [{p['dist'][-1]:.0f} km]",transform=ax.transAxes,
                fontsize=8,fontweight='bold',va='top',
                bbox=dict(fc='white',ec='none',alpha=0.7,pad=1))
        if p['moho'] is not None:
            _draw_moho(ax, p['dist'], p['moho'], zbot, moho_mask, lw=0.9)
        if i==n-1: ax.set_xlabel('Distance (km)',fontsize=8)
        # per-panel colorbar immediately right of THIS panel
        cax=fig.add_axes([0.08+w+0.008, y0, 0.012, rh])
        cb=fig.colorbar(im,cax=cax,extend=ext); cb.ax.tick_params(labelsize=6)
        if i==0: cb.set_label(clab,fontsize=8)
        geo_res.append(_add_geology(ax,[ax],p['plon'],p['plat'],p['dist']))

    if geo_res:
        _geology_legend(fig, geo_res, ax, below_in=0.55)
    os.makedirs(out_dir,exist_ok=True)
    out=os.path.join(out_dir,f'xsection_stack.{field}.png')
    fig.savefig(out,dpi=150,bbox_inches='tight')
    fig.savefig(out.replace('.png','.pdf'),bbox_inches='tight')
    plt.close(fig)
    print(f"Stack ({field}): {out} (+ .pdf)")


def main():
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
    ap.add_argument('--ginput',  action='store_true',
                    help='Pick section endpoints interactively from Ndat map. '
                         'Click pairs of points (start,end) — Enter to finish.')
    ap.add_argument('--moho',      default=None,
                    help='AR23 Moho file (AR23-moho-hmp.txt) to overlay as dashed line')
    ap.add_argument('--xi-panel', default='xi', choices=['xi', 'dxi', 'both'],
                    help='per-section xi row(s): xi, dlnXi, or both (ZT models)')
    ap.add_argument('--xi-ref', default='depth', choices=['depth', 'global'],
                    help='dlnXi reference: whole-model mean at each depth (default) or '
                         'one whole-model mean over all depths')
    ap.add_argument('--clim-xi', type=float, default=5.0,
                    help='dlnXi colour limit, +/- %% (default 5)')
    ap.add_argument('--div-gap', type=float, default=0.15,
                    help='diverging colour maps skip +/- this band around white '
                         '(0 = classic RdBu through white; default 0.15)')
    ap.add_argument('--div-white', type=int, default=2,
                    help='neutral colour levels at the centre of diverging maps (0 = none)')
    ap.add_argument('--moho-mask-alpha', type=float, default=0.5,
                    help='grey veil below the Moho to the bottom of the section (0-1, default 0.5)')
    ap.add_argument('--no-moho-mask', action='store_true', help='Moho line only, no grey veil')
    ap.add_argument('--save-sections', default=None,
                    help='Save picked/used section list to this file')
    ap.add_argument('--load-sections', default=None,
                    help='Load section list from a file (skip ginput)')
    ap.add_argument('--stack', default=None,
                    choices=['vsv','dvsv','xi','dxi','err'],
                    help='Stack ONE field for all sections on a single A4 page '
                         '(rows=sections, same height, width proportional to length).')
    ap.add_argument('--vmin', type=float, default=None,
                    help='Fixed colour min for the stacked field (all panels same scale)')
    ap.add_argument('--vmax', type=float, default=None,
                    help='Fixed colour max for the stacked field')
    ap.add_argument('--stack-ve', type=float, default=DEFAULT_VE,
                    help='Vertical exaggeration for stacked sections (1=true scale/flat, higher=taller; default 3).')
    ap.add_argument('--index-map', action='store_true',
                    help='Also draw an index map of all sections on the tectonic geology')
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
                         'litho,domain,boundaries,names (default: all four)')
    ap.add_argument('--no-strips', action='store_true', help='plain sections, no geology strips')
    ap.add_argument('--index-only', action='store_true',
                    help='with --index-map: draw only the map, do not re-plot any sections')
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
    ap.add_argument('--index-style', default='gswa', choices=['gswa', 'craton', 'plain'],
                    help='--index-map colours: gswa = GSWA 500k unit colours (default); '
                         'craton = tectonic domains + Yilgarn/Pilbara granite-greenstone; '
                         'plain = tectonic domains only')
    ap.add_argument('--index-plain', action='store_true',
                    help='--index-map: tectonic colours only (no craton granite/greenstone)')
    ap.add_argument('--tect-npz',  default=None, help='override tectonics npz')
    ap.add_argument('--bnd-npz',   default=None, help='override crustal-boundaries npz')
    ap.add_argument('--out-dir',   default='figures/xsections')
    args = ap.parse_args()

    global _GEO
    geo_args = None
    if args.no_strips:
        args.strips = None
    if args.index_only:
        args.index_map = True
    if args.strips and not args.index_only:
        geo_args = (args.strips, args.litho_npz, args.tect_npz, args.bnd_npz)
        _GEO = _load_geology(*geo_args)

    print(f"Loading {args.fvs} ...")
    d = np.load(args.fvs, allow_pickle=True)
    plot_section._fvsname = args.fvs   # for Viso/Vsv label detection
    # Optional Moho interpolator
    plot_section._moho_interp = None
    if args.moho:
        try:
            from scipy.interpolate import RectBivariateSpline
            md = np.loadtxt(args.moho, skiprows=11)
            mlats = np.unique(md[:,0]); mlons = np.unique(md[:,1])
            mg = md[:,2].reshape(len(mlats), len(mlons))
            plot_section._moho_interp = RectBivariateSpline(mlats, mlons, mg, kx=1, ky=1)
            print(f'  Moho overlay loaded: {args.moho}')
        except Exception as e:
            print(f'  Moho load failed: {e}')
    print(f"  {len(d['Lon'])} nodes, {len(d['z'])} depth levels "
          f"({d['z'][0]:.1f}-{d['z'][-1]:.1f} km)")

    moho_mask = None if args.no_moho_mask else args.moho_mask_alpha
    kw = dict(ds_deg=args.ds, ncolors=args.ncolors, d_max=args.d_max,
              sigma_vsv=args.sigma_vsv, clim_rel=args.clim_rel,
              out_dir=args.out_dir, moho_mask=moho_mask,
              xi_panel=args.xi_panel, xi_ref=args.xi_ref, clim_xi=args.clim_xi,
              div_gap=args.div_gap, div_white=args.div_white)

    # --- Assemble the section list (ginput / load / sections / start-end) ---
    sections = []
    if args.load_sections:
        sections = _load_sections(args.load_sections)
        print(f"Loaded {len(sections)} sections from {args.load_sections}")
    elif args.ginput:
        sections = _ginput_sections(d)
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
        _plot_index_map(d, sections, args.out_dir, tick_km=args.index_tick, stations=sta,
                        style='plain' if args.index_plain else args.index_style,
                        footprint=_index_footprint(args), footprint_veil=args.footprint_veil,
                        litho_npz=None if args.index_plain else
                        _resolve_npz('LITHOLOGY_NPZ', 'wa_lithology.npz', args.litho_npz))
        if args.index_only:
            return

    # Stacked single-field figure (one page, all sections) — then return
    if args.stack:
        zmask=(d['z']<=args.d_max) if args.d_max else np.ones(len(d['z']),bool)
        rmean=np.array([np.nanmean(d['Vsv'][:,zmask][:,k]) for k in range(zmask.sum())])
        plot_stack(d, sections, args.stack, args.out_dir, moho_file=args.moho,
                   ds_deg=args.ds, d_max=args.d_max, ref_mean=rmean,
                   ve=args.stack_ve, vmin=args.vmin, vmax=args.vmax,
                   ncolors=args.ncolors,
                   moho_mask=None if args.no_moho_mask else args.moho_mask_alpha,
                   xi_ref=args.xi_ref, clim_xi=args.clim_xi, div_gap=args.div_gap,
                   div_white=args.div_white)
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
    ref_mean = np.array([np.nanmean(Vsv_all[:, k]) for k in range(Vsv_all.shape[1])])

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
                         geo_args=geo_args)
        with Pool(args.nproc) as pool:
            pool.map(worker, sections)
    else:
        for sec in sections:
            plot_section(d, *sec, **kw)


def _section_worker(sec, fvs, moho, kw, geo_args=None):
    """Standalone worker for parallel section plotting (reloads Fvs per process)."""
    import numpy as np
    global _GEO
    if geo_args and _GEO is None:     # not inherited (spawn/forkserver start method)
        _GEO = _load_geology(*geo_args)
    d = np.load(fvs, allow_pickle=True)
    plot_section._fvsname = fvs
    plot_section._moho_interp = None
    if moho:
        try:
            from scipy.interpolate import RectBivariateSpline
            md = np.loadtxt(moho, skiprows=11)
            mlats = np.unique(md[:,0]); mlons = np.unique(md[:,1])
            mg = md[:,2].reshape(len(mlats), len(mlons))
            plot_section._moho_interp = RectBivariateSpline(mlats, mlons, mg, kx=1, ky=1)
        except Exception:
            pass
    plot_section(d, *sec, **kw)


if __name__ == '__main__':
    main()
