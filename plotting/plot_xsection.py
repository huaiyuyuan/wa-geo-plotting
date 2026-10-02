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

H. Yuan / Claude, Sep 2026
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


# ── Colormap helper ───────────────────────────────────────────────────────────
def _cmap(name, n=32):
    try:
        return matplotlib.colormaps[name].resampled(n)
    except Exception:
        return plt.cm.get_cmap(name, n)


# ── Single cross-section ──────────────────────────────────────────────────────
def plot_section(d, lat1, lon1, lat2, lon2, label,
                 ds_deg=0.08, ncolors=32, d_max=None,
                 sigma_vsv=1.5, clim_rel=6.0,
                 out_dir='figures/xsections',
                 max_dist=None, km_per_in=None, panel_h_in=None, ref_mean=None,
                 long_ratio=3.0):
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

    rows = ['abs', 'rel']
    if have_xi: rows.append('xi')
    if have_err: rows.append('err')
    nrow = len(rows)

    xmax = dist[-1]; zbot = z[-1]
    # Scale set by the longest section (full landscape width @ 2:1 per panel).
    # If not supplied, derive from THIS section (standalone use).
    # A4 landscape usable panel width (A4 = 11.69 x 8.27 in; leave room for cbar)
    A4_PANEL_W = 10.0
    if max_dist is None: max_dist = xmax
    if km_per_in is None:  km_per_in  = max_dist / A4_PANEL_W        # longest → A4 width
    if panel_h_in is None: panel_h_in = A4_PANEL_W / long_ratio      # long_ratio:1 for longest
    # This section: width ∝ its length at the shared km/in; height is the SHARED height
    panel_w_in = max(1.2, xmax / km_per_in)
    cbar_in    = 1.5
    fig_w = panel_w_in + cbar_in
    fig_h = panel_h_in * nrow + 0.9
    vexag = (max_dist / zbot) / long_ratio     # implied VE (longest drawn long_ratio:1)
    fig, axes = plt.subplots(nrow, 1, figsize=(fig_w, fig_h), sharex=True)
    if nrow == 1: axes = [axes]
    fig.suptitle(f'{label}:  ({lat1:.1f}°,{lon1:.1f}°) -> ({lat2:.1f}°,{lon2:.1f}°)   '
                 f'[{xmax:.0f} km, VE {vexag:.0f}x]', fontsize=11, fontweight='bold')

    def _pcolor(ax, data, cmap, vmin, vmax, clabel, extend='both', title=''):
        im = ax.pcolormesh(D, Z, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_ylim(zbot, 0); ax.set_xlim(0, xmax)
        ax.set_ylabel('Depth (km)', fontsize=9); ax.tick_params(labelsize=8)
        cb = plt.colorbar(im, ax=ax, shrink=0.95, pad=0.01, extend=extend, aspect=10)
        cb.set_label(clabel, fontsize=8); cb.ax.tick_params(labelsize=7)
        if title: ax.set_title(title, fontsize=9, loc='left')
        if moho_prof is not None:
            ax.plot(dist, moho_prof, 'k--', lw=1.1, alpha=0.7)
        return im

    r = 0
    _pcolor(axes[r], vsv_p, _cmap('RdBu', ncolors), vmin_v, vmax_v,
            f'{vname} (km/s)', title=f'{vname} (km/s)'); r += 1
    _pcolor(axes[r], dln_p, _cmap('RdBu', ncolors), -clim_rel, clim_rel,
            f'd{vname} (%)', title=f'd{vname} (%) vs model mean'); r += 1
    if have_xi:
        _pcolor(axes[r], xi_p, _cmap('RdBu', ncolors), 0.90, 1.10,
                'xi', title='xi = Vsh/Vsv (blue = Vsh>Vsv)'); r += 1
    if have_err:
        vmax_e = np.nanpercentile(err_p[np.isfinite(err_p)], 90)
        _pcolor(axes[r], err_p, _cmap('YlOrRd', ncolors), 0, vmax_e,
                f'{vname} IQR/2', extend='max',
                title='Model uncertainty (posterior IQR/2)'); r += 1

    axes[-1].set_xlabel('Distance along profile (km)', fontsize=9)
    axes[-1].set_xticks(np.arange(0, xmax+1, 100))

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
def _plot_index_map(d, sections, out_dir, label='sections', moho=None):
    """Draw all section lines on the WA tectonic map with distance ticks + labels."""
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
        from wa_basemap import add_tectonic_background, add_tectonic_outlines, tectonic_legend
        add_tectonic_background(ax, alpha=0.5, zorder=0, transform=tr, clip_box=cb)
        add_tectonic_outlines(ax, lw=0.4, alpha=0.6, zorder=3, major_only=True,
                              transform=tr, clip_box=cb)
        _tect = True
    except Exception as e:
        print(f"(tectonic bg unavailable: {e})")
        ax.scatter(Lon, Lat, c='0.7', s=10, zorder=1,
                   **({'transform': tr} if tr else {})); _tect = False

    lkw = {'transform': tr} if tr else {}
    for i, (lat1, lon1, lat2, lon2, lab) in enumerate(sections):
        plat, plon, dist = _great_circle_path(lat1, lon1, lat2, lon2)
        ax.plot(plon, plat, 'w-', lw=3.4, zorder=5, **lkw)   # white halo
        ax.plot(plon, plat, 'k-', lw=2.0, zorder=6, **lkw)
        ax.plot(lon1, lat1, 'ko', ms=6, zorder=7, **lkw)
        ax.plot(lon2, lat2, 'ks', ms=6, zorder=7, **lkw)
        for dkm in np.arange(0, dist[-1], 200):
            j = np.argmin(np.abs(dist - dkm))
            ax.plot(plon[j], plat[j], '|', color='k', ms=10, mew=1.5, zorder=7, **lkw)
            ax.annotate(f'{int(dkm)}', (plon[j], plat[j]), fontsize=6,
                        ha='center', va='bottom', zorder=8,
                        xytext=(0,3), textcoords='offset points')
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
    ax.set_title(f'Cross-section locations ({len(sections)} lines, ticks every 200 km)',
                 fontsize=11)
    if _tect:
        tectonic_legend(ax, loc='upper left', fontsize=10)
    import os
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'xsection_index_map.png')
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
               ncolors=32, d_max=None, sigma_vsv=1.5, clim_rel=6.0,
               ref_mean=None, row_h_in=None, page_w_in=9.5, gap_in=0.55,
               ve=5.0, vmin=None, vmax=None):
    """Stack ONE field for all sections on a single page. Rows = sections, same
    height, width proportional to length, longest spans the full page width.
    ve = vertical exaggeration (depth stretched x this; 1 = true scale, flat;
    higher = taller). Each panel has its own colorbar."""
    import matplotlib.pyplot as plt
    Lon = d['Lon']; Lat = d['Lat']; z = d['z']
    if d_max: zm = z <= d_max; z = z[zm]
    else: zm = np.ones(len(z), bool)
    zbot = z[-1]

    FKEY = {'vsv':'Vsv','dvsv':'Vsv','xi':'Xi','err':'Vsv_err'}
    arrname = FKEY.get(field,'Vsv')
    if arrname not in d:
        print(f"  field {field}: {arrname} not in Fvs"); return
    ARR = d[arrname][:, zm]
    if ref_mean is None and field=='dvsv':
        ref_mean = np.array([np.nanmean(d['Vsv'][:,zm][:,k]) for k in range(ARR.shape[1])])

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
        mp=None
        if _mi is not None:
            mp=np.array([float(np.ravel(_mi(a,o))[0]) for a,o in zip(plat,plon)])
        prof.append(dict(lab=lab,dist=dist,data=samp,moho=mp))
    maxd=max(p['dist'][-1] for p in prof)

    # Horizontal scale: longest fills page_w_in. Row height from VE:
    #   km_per_in = maxd/page_w_in ; row_h_in = zbot*ve/km_per_in
    km_per_in = maxd / page_w_in
    if row_h_in is None:
        row_h_in = zbot * ve / km_per_in

    if field=='vsv':
        cmap=_cmap('RdBu',ncolors); clab=f'{vname} (km/s)'; ext='both'
    elif field=='dvsv':
        cmap=_cmap('RdBu',ncolors); clab=f'd{vname} (%)'; ext='both'
    elif field=='xi':
        cmap=_cmap('RdBu',ncolors); clab='xi'; ext='both'
    else:
        cmap=_cmap('YlOrRd',ncolors); clab=f'{vname} IQR/2'; ext='max'

    n=len(prof)
    fig_h=n*row_h_in + (n-1)*gap_in + 1.2
    fig_w=page_w_in + 1.8                       # + per-panel colorbar room
    fig=plt.figure(figsize=(fig_w,fig_h))
    fig.suptitle(f'{field.upper()} ({vname}) — {n} sections [depth 0-{zbot:.0f} km, '
                 f'longest {maxd:.0f} km, VE {ve:.0f}x]',
                 fontsize=11, fontweight='bold')

    usable=page_w_in/fig_w
    rh=row_h_in/fig_h
    for i,p in enumerate(prof):
        w=(p['dist'][-1]/maxd)*usable
        y0=1-(0.9/fig_h)-(i+1)*rh-i*(gap_in/fig_h)
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
            ax.plot(p['dist'],p['moho'],'k--',lw=0.9,alpha=0.7)
        if i==n-1: ax.set_xlabel('Distance (km)',fontsize=8)
        # per-panel colorbar immediately right of THIS panel
        cax=fig.add_axes([0.08+w+0.008, y0, 0.012, rh])
        cb=fig.colorbar(im,cax=cax,extend=ext); cb.ax.tick_params(labelsize=6)
        if i==0: cb.set_label(clab,fontsize=8)

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
    ap.add_argument('--ncolors',   type=int,   default=32)
    ap.add_argument('--ginput',  action='store_true',
                    help='Pick section endpoints interactively from Ndat map. '
                         'Click pairs of points (start,end) — Enter to finish.')
    ap.add_argument('--moho',      default=None,
                    help='AR23 Moho file (AR23-moho-hmp.txt) to overlay as dashed line')
    ap.add_argument('--save-sections', default=None,
                    help='Save picked/used section list to this file')
    ap.add_argument('--load-sections', default=None,
                    help='Load section list from a file (skip ginput)')
    ap.add_argument('--stack', default=None,
                    choices=['vsv','dvsv','xi','err'],
                    help='Stack ONE field for all sections on a single A4 page '
                         '(rows=sections, same height, width proportional to length).')
    ap.add_argument('--vmin', type=float, default=None,
                    help='Fixed colour min for the stacked field (all panels same scale)')
    ap.add_argument('--vmax', type=float, default=None,
                    help='Fixed colour max for the stacked field')
    ap.add_argument('--stack-ve', type=float, default=5.0,
                    help='Vertical exaggeration for stacked sections (1=true scale/flat, higher=taller; default 5).')
    ap.add_argument('--index-map', action='store_true',
                    help='Also draw an index map of all sections on the tectonic geology')
    ap.add_argument('--long-ratio', type=float, default=3.0,
                    help='Width:height of the LONGEST section panel (default 1.5). '
                         'Lower = more vertical exaggeration; higher = flatter.')
    ap.add_argument('--nproc', type=int, default=1,
                    help='Parallelise section plotting across N cores (default 1)')
    ap.add_argument('--out-dir',   default='figures/xsections')
    args = ap.parse_args()

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

    kw = dict(ds_deg=args.ds, ncolors=args.ncolors, d_max=args.d_max,
              sigma_vsv=args.sigma_vsv, clim_rel=args.clim_rel,
              out_dir=args.out_dir)

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
        _plot_index_map(d, sections, args.out_dir)

    # Stacked single-field figure (one page, all sections) — then return
    if args.stack:
        zmask=(d['z']<=args.d_max) if args.d_max else np.ones(len(d['z']),bool)
        rmean=np.array([np.nanmean(d['Vsv'][:,zmask][:,k]) for k in range(zmask.sum())])
        plot_stack(d, sections, args.stack, args.out_dir, moho_file=args.moho,
                   ds_deg=args.ds, d_max=args.d_max, ref_mean=rmean,
                   ve=args.stack_ve, vmin=args.vmin, vmax=args.vmax)
        return

    # Longest section sets the scale (drawn 2:1 at landscape width); all sections
    # then share the same km/inch and panel height.
    maxd = 0.0
    for (la1,lo1,la2,lo2,_lab) in sections:
        _,_,dd = _great_circle_path(la1,lo1,la2,lo2, args.ds)
        maxd = max(maxd, dd[-1])
    A4_PANEL_W = 10.0             # A4 landscape usable panel width (inches)
    km_per_in  = maxd / A4_PANEL_W
    panel_h_in = A4_PANEL_W / args.long_ratio   # longest = long_ratio:1 → shared height
    print(f'Longest {maxd:.0f} km → {km_per_in:.0f} km/in, panel height {panel_h_in:.1f} in (longest {args.long_ratio:.1f}:1)')

    # Full-model mean Vsv per depth → dln reference (regional, consistent)
    zmask = (d['z'] <= args.d_max) if args.d_max else np.ones(len(d['z']), bool)
    Vsv_all = d['Vsv'][:, zmask]
    ref_mean = np.array([np.nanmean(Vsv_all[:, k]) for k in range(Vsv_all.shape[1])])

    kw['max_dist']   = maxd
    kw['km_per_in']  = km_per_in
    kw['panel_h_in'] = panel_h_in
    kw['long_ratio'] = args.long_ratio
    kw['ref_mean']   = ref_mean

    # --- Plot each section (optionally in parallel) ---
    if args.nproc > 1 and len(sections) > 1:
        from multiprocessing import Pool
        from functools import partial
        print(f"Plotting {len(sections)} sections on {args.nproc} cores ...")
        worker = partial(_section_worker, fvs=args.fvs, moho=args.moho, kw=kw)
        with Pool(args.nproc) as pool:
            pool.map(worker, sections)
    else:
        for sec in sections:
            plot_section(d, *sec, **kw)


def _section_worker(sec, fvs, moho, kw):
    """Standalone worker for parallel section plotting (reloads Fvs per process)."""
    import numpy as np
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
