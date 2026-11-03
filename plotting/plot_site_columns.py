#!/usr/bin/env python3
"""
plot_site_columns.py - 'lithospheric columns' at the study sites: one vertical bar per
site, coloured by velocity in fixed bins, depth down, site names on top in the terrane
colours. Two sources:

  TransD model (km scale): Vsv every 0.25 km/s from 3.0 to 4.5 (default), to 50 km
    python3 plotting/plot_site_columns.py --fvs $FVS2 --moho $MOHO \\
        --out $FIG/iter2/sites/site_columns.iter2.png
  shallow station profiles (m scale, plot_site_shallow.py files): Vs in m/s, 250 m/s bins
  from the slowest value to the 90th percentile of the profile maxima (faster = arrow
  colour), to --z-max m (default 650)
    python3 plotting/plot_site_columns.py --shallow station_files.zip \\
        --out $FIG/sites/site_columns.shallow.png

Colours: vivid turbo, reversed so warm = slow (--cmap Spectral for the softer house map);
values beyond the bins take the end colours (arrowed colour bar). Model profiles as in
plot_site_profiles.py (nearest node, or --idw K); a site with a distant node / station or
low Ndat gets '*' and a hatched column; a site without a profile an empty 'no data'
column; a shallow profile ending above --z-max stops there. --order terrane groups the
columns by terrane. AR23 Moho (--moho, model mode) = black bar across each column.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import tomo2d as t2                                           # noqa: E402
from plot_site_profiles import sample, model_label, LABEL, gc_km   # noqa: E402

DEFAULT_BINS = {'vsv': (3.0, 4.5, 0.25), 'viso': (3.0, 4.5, 0.25), 'xi': (0.9, 1.1, 0.025),
                'xi2': (0.88, 1.12, 0.02),
                'vpvs': (1.65, 1.90, 0.025)}


def make_cmap(name, nb):
    """nb colours + under/over from `name`; turbo is trimmed and reversed (warm = slow)."""
    if name in ('turbo', 'turbo_r'):
        # turbo runs blue (low) -> red (high); turbo_r = red for slow, blue for fast
        x = np.linspace(0.05, 0.95, nb + 2) if name == 'turbo' else np.linspace(0.95, 0.05, nb + 2)
        cols = matplotlib.colormaps['turbo'](x)
    else:
        cols = matplotlib.colormaps[name](np.linspace(0, 1, nb + 2))
    cm = ListedColormap(cols[1:-1]); cm.set_under(cols[0]); cm.set_over(cols[-1])
    return cm


def _moho_sharpness(path, lons, lats):
    """From a CCP refined-Moho .mat (struct moho_refined: lon2/lat2/z_moho + amp_moho|confidence
    + is_missing): per-site Moho depth, a 0-1 sharpness (amp_moho else confidence, normalised over
    the grid's 10-90th percentile) and a missing flag. Scattered interpolation, so grid shape
    doesn't matter. Returns (depth, sharp, missing) arrays."""
    from scipy.io import loadmat
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    m = loadmat(path, squeeze_me=True, struct_as_record=False)['moho_refined']
    LON = np.asarray(m.lon2, float).ravel(); LAT = np.asarray(m.lat2, float).ravel()
    Z = np.asarray(m.z_moho, float).ravel()
    amp = np.asarray(getattr(m, 'amp_moho', getattr(m, 'confidence', np.ones_like(Z))), float).ravel()
    miss = (np.asarray(m.is_missing).astype(float).ravel() if hasattr(m, 'is_missing')
            else np.zeros_like(Z))
    good = np.isfinite(LON) & np.isfinite(LAT) & np.isfinite(Z)
    pts = np.column_stack([LON[good], LAT[good]]); q = np.column_stack([lons, lats])
    def _interp(v):
        lin = LinearNDInterpolator(pts, v)(q)
        nn = NearestNDInterpolator(pts, v)(q)
        return np.where(np.isfinite(lin), lin, nn)
    depth = _interp(Z[good]); qa = _interp(np.nan_to_num(amp[good], nan=np.nanmedian(amp[good])))
    qm = _interp(miss[good]) > 0.5
    av = amp[good][np.isfinite(amp[good])]
    lo, hi = (np.percentile(av, [10, 90]) if av.size else (0.0, 1.0))
    sharp = np.clip((qa - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return depth, sharp, qm


def from_model(a, sites, fvs=None, field=None):
    fvs = fvs or a.fvs
    field = field or a.field
    zq = np.arange(0, a.z_max + 1e-6, a.dz)
    moho = None
    lons = [s['lon'] for s in sites]; lats = [s['lat'] for s in sites]
    sharp = miss = None
    if getattr(a, 'moho_mat', None) and os.path.isfile(a.moho_mat):
        moho, sharp, miss = _moho_sharpness(a.moho_mat, lons, lats)      # depth + sharpness + missing
        print(f"  Moho sharpness from {a.moho_mat}: {int(np.sum(miss))}/{len(sites)} missing")
    elif a.moho and os.path.isfile(a.moho):
        from plot_domain_profiles import moho_at
        moho = moho_at(a.moho, lons, lats)
    path, label = model_label(fvs)
    M = sample(path, label, field, sites, zq, a, moho)
    note = 'inverse-distance mean of %d nearest nodes' % a.idw if a.idw > 1 else 'nearest model node'
    flags = [i['flag'] for i in M['info']]
    if any(flags):
        note += f"; * hatched: node > {a.max_km:g} km away or Ndat < {a.min_ndat}"
    _moho = moho if moho is not None else [i['moho'] for i in M['info']]
    return dict(z=zq, P=M['P'], flag=flags, moho=list(_moho), field=field,
                moho_sharp=(list(sharp) if sharp is not None else None),
                moho_missing=(list(miss) if miss is not None else None),
                title=f"{label} {LABEL[field].split(' (')[0]} columns at the study sites",
                clabel=LABEL[field], zlabel='depth (km)', note=note)


def _blockify(v, z, mo, nlayers=3, z_max=60.0):
    """Collapse a fine profile into `nlayers` equal-thickness crustal mean blocks (0->Moho)
    plus one sub-Moho (mantle) mean block, each filled with its layer mean so pcolormesh draws
    solid layers. Returns (blocky_profile, internal_crust_boundary_depths)."""
    v = np.asarray(v, float)
    out = np.full_like(v, np.nan)
    top = mo if (np.isfinite(mo) and mo > 0) else z_max
    edges = np.linspace(0.0, top, nlayers + 1)
    for i in range(nlayers):
        lo, hi = edges[i], edges[i + 1]
        sel = (z >= lo) & (z <= hi)
        fin = sel & np.isfinite(v)
        if fin.any():
            out[sel] = np.nanmean(v[fin])
    if np.isfinite(mo):                                   # sub-Moho mantle block
        sel = (z >= mo) & np.isfinite(v)
        if sel.any():
            out[z >= mo] = np.nanmean(v[sel])
    out[~np.isfinite(v)] = np.nan                         # keep real no-data as blank
    return out, list(edges[1:-1])


def _bins_for(a, R, field, shallow):
    if a.bins:
        lo, hi, st = a.bins
    elif shallow:
        st = 250.0
        vv = R['P'][np.isfinite(R['P'])]
        mx = [np.nanmax(p) for p in R['P'] if np.isfinite(p).any()]
        lo, hi = st * np.floor(vv.min() / st), st * np.ceil(np.percentile(mx, 90) / st)
    else:
        lo, hi, st = DEFAULT_BINS[field]
    bounds = np.round(np.arange(lo, hi + st / 2, st), 6)
    return lo, hi, st, bounds


def _fabric_sticks(ax, k, edges, vblocky, z, axw, axh, ns, z_max, vmin=0.0):
    """Anisotropy strokes per crustal block on the xi panel. ORIENTATION = sign: xi>=1 (Vsh>Vsv)
    -> horizontal strokes; xi<1 (Vsv>Vsh) -> vertical strokes. STRENGTH = |xi-1| -> number of
    strokes (1-3). axw/axh (axes inches) correct the stroke length/spacing for the data aspect."""
    sx = (ns + 0.4) / max(axw, 1e-6)            # data-x per inch (xlim span ~ ns+0.4)
    sy = z_max / max(axh, 1e-6)                 # data-y per inch
    Lh, sep = 0.075, 0.055                       # stroke half-length / spacing (inch)
    for d0, d1 in zip(edges[:-1], edges[1:]):
        m = (z >= d0) & (z <= d1) & np.isfinite(vblocky)
        if not m.any():
            continue
        xi = float(np.nanmean(vblocky[m]))
        aniso = xi - 1.0
        if abs(aniso) < vmin:                                      # near-isotropic -> no strokes
            continue
        nper = int(np.clip(1 + round(abs(aniso) / 0.025), 1, 3))   # strength -> 1..3 strokes
        horiz = aniso >= 0.0
        cx, cy = float(k), 0.5 * (d0 + d1)
        for o in np.linspace(-(nper - 1) / 2.0, (nper - 1) / 2.0, nper):
            if horiz:                            # positive: horizontal strokes, stacked in depth
                bx, by, dx, dy = cx, cy + o * sep * sy, Lh * sx, 0.0
            else:                                # negative: vertical strokes, side by side
                bx, by, dx, dy = cx + o * sep * sx, cy, 0.0, Lh * sy
            ax.plot([bx - dx, bx + dx], [by - dy, by + dy], color='k', lw=1.1,
                    solid_capstyle='round', zorder=5)


def _fabric_ellipse(ax, k, edges, xi_prof, z, axw, axh, ns, z_max, w, usc=1.0, vmin=0.02):
    """Hollow deformation-fabric ellipse per crustal block. ORIENTATION: xi>=1 (Vsh>Vsv) ->
    horizontal long axis; xi<1 (Vsv>Vsh) -> tilted 45 deg. SIZE and ELONGATION both grow with
    |xi-1| (isotropic ~ a small circle). `usc` scales the overall size. Each ellipse is clamped
    inside the column/block. Drawn in inch space -> data (sx/sy) so the aspect reads true."""
    sx = (ns + 0.4) / max(axw, 1e-6)
    sy = z_max / max(axh, 1e-6)
    t = np.linspace(0, 2 * np.pi, 64)
    for d0, d1 in zip(edges[:-1], edges[1:]):
        m = (z >= d0) & (z <= d1) & np.isfinite(xi_prof)
        if not m.any():
            continue
        xi = float(np.nanmean(xi_prof[m])); an = xi - 1.0
        if abs(an) < vmin:                                      # near-isotropic -> no ellipse
            continue
        strg = min(abs(an) / 0.05, 1.0)                         # saturate at |xi-1|=0.05
        r0 = (0.055 + 0.038 * strg) * usc                       # base radius (inch)
        aspect = 1.0 + 3.6 * strg                               # 1 (circle) .. 4.6 (strongly elongated)
        A, B = r0 * np.sqrt(aspect), r0 / np.sqrt(aspect)       # long / short semi-axis (inch)
        th = 0.0 if an >= 0 else np.radians(45.0)
        ex = A * np.cos(t) * np.cos(th) - B * np.sin(t) * np.sin(th)
        ey = A * np.cos(t) * np.sin(th) + B * np.sin(t) * np.cos(th)
        xext = (ex.max() - ex.min()) * sx; yext = (ey.max() - ey.min()) * sy
        fit = min(1.0, 0.9 * w / max(xext, 1e-6), 0.82 * (d1 - d0) / max(yext, 1e-6))
        cx, cy = float(k), 0.5 * (d0 + d1)
        ax.fill(cx + ex * sx * fit, cy + ey * sy * fit, facecolor='white', edgecolor='k',
                lw=1.3, zorder=5)                              # white fill, black edge


def _draw_fabric(ax, k, edges, xi_prof, z, axw, axh, ns, z_max, w, kind, usc=1.0, vmin=0.02):
    if kind == 'sticks':
        _fabric_sticks(ax, k, edges, xi_prof, z, axw, axh, ns, z_max, vmin)
    elif kind != 'none':
        _fabric_ellipse(ax, k, edges, xi_prof, z, axw, axh, ns, z_max, w, usc, vmin)


def _draw_panel(fig, ax, R, sites, a, field, shallow, nlayers, show_top, show_bottom, overlay_xi=None):
    """Draw one row of site columns on `ax` (its own bins + colorbar). show_top = site-name
    labels above; show_bottom = terrane x-tick labels below."""
    lo, hi, st, bounds = _bins_for(a, R, field, shallow)
    nb = len(bounds) - 1
    base = a.cmap or {'xi': 'RdBu', 'xi2': 'RdBu', 'vpvs': 'viridis'}.get(
        field if not shallow else 'vs', 'turbo_r')   # xi: RdBu = red low -> blue high (blue = high)
    cmap = make_cmap(base, nb); norm = BoundaryNorm(bounds, cmap.N)
    print(f"  {field}: colour bins {lo:g}-{hi:g} every {st:g} ({nb} colours, {base})")
    z = R['z']; ns = len(sites); w = a.width
    ze = np.r_[z[0], 0.5 * (z[1:] + z[:-1]), z[-1]]
    pos = ax.get_position(); axw = pos.width * fig.get_figwidth(); axh = pos.height * fig.get_figheight()
    fabric = getattr(a, 'xi_fabric', 'ellipse')
    want_fabric = bool(nlayers) and fabric != 'none' and (field in ('xi', 'xi2') or overlay_xi is not None)
    pm = None
    mohos = np.full(ns, np.nan)
    m_sharp = R.get('moho_sharp'); m_miss = R.get('moho_missing')    # None unless --moho-mat given
    for k, s in enumerate(sites):
        v = np.asarray(R['P'][k], float)
        mo = R['moho'][k]; mohos[k] = mo
        bnds = []
        if nlayers and np.isfinite(v).any():
            v, bnds = _blockify(v, z, mo, nlayers, a.z_max)
        flag = R['flag'][k]
        al = 0.5 if flag else 1.0                         # flagged -> 50% transparent fill (keep hatch)
        if np.isfinite(v).any():
            pm = ax.pcolormesh([k - w / 2, k + w / 2], ze, np.ma.masked_invalid(v)[:, None],
                               cmap=cmap, norm=norm, shading='flat', alpha=al, zorder=1)
            zb = z[np.isfinite(v)].max()
        else:
            zb = a.z_max
            ax.text(k, a.z_max / 2, R['sub'][k] if shallow else 'no data', rotation=90,
                    ha='center', va='center', fontsize=8, color='0.5')
        ax.add_patch(plt.Rectangle((k - w / 2, 0), w, zb, fill=False, ec='k', lw=0.6,
                                   hatch='///' if flag else None, zorder=3))
        for b in bnds:                                    # thin crust-layer separators
            ax.plot([k - w / 2, k + w / 2], [b, b], color='0.2', lw=0.5, zorder=3.5)
        if np.isfinite(mo) and mo <= a.z_max:
            x0, x1 = k - w / 2 - 0.08, k + w / 2 + 0.08
            if m_miss is not None and m_miss[k]:          # missing Moho -> thin dashed grey line
                ax.plot([x0, x1], [mo, mo], color='0.55', lw=1.4, ls=(0, (3, 2)), zorder=6)
            else:                                         # sharpness -> bar shade (dark=strong) + thickness
                sh = 1.0 if m_sharp is None else float(m_sharp[k])
                g = 0.80 * (1.0 - sh)                     # 0 (black, sharp) -> 0.8 (light grey, weak)
                ax.plot([x0, x1], [mo, mo], color=str(g), lw=1.6 + 2.0 * sh, zorder=6,
                        solid_capstyle='butt')
        if want_fabric and np.isfinite(v).any():          # anisotropy fabric (ellipse or sticks)
            edges = [0.0] + list(bnds) + [mo if np.isfinite(mo) else a.z_max]
            if field in ('xi', 'xi2'):
                xi_prof = v                                # this panel's own (blockified) xi
            else:                                          # overlay the xi model onto the Vsv panel
                xi_prof, _ = _blockify(np.asarray(overlay_xi['P'][k], float), z, mo, nlayers, a.z_max)
            _draw_fabric(ax, k, edges, xi_prof, z, axw, axh, ns, a.z_max, w, fabric,
                         getattr(a, 'fabric_scale', 1.0), getattr(a, 'fabric_min', 0.005))
        if show_top:
            col = t2.terrane_colour(s['terrane'], k)
            ax.text(k - w / 2, -0.02 * a.z_max, f"{s['no']} {s['name']}" + ('*' if flag else ''),
                    rotation=45, ha='left', va='bottom', fontsize=8.5, fontweight='bold',
                    color=col, rotation_mode='anchor')
    # translucent grey sub-crustal lithosphere mask: top edge FLAT under each bar at its Moho,
    # connected bar-to-bar across the gaps (so it meets the ends of the black Moho bars)
    xs_top, ys_top = [], []
    for k in range(ns):
        if np.isfinite(mohos[k]) and mohos[k] <= a.z_max:
            xs_top += [k - w / 2, k + w / 2]
            ys_top += [mohos[k], mohos[k]]
    if len(xs_top) >= 2:
        ax.fill_between(xs_top, ys_top, a.z_max, color='0.5', alpha=0.30, lw=0, zorder=2.5)
    ax.set_xlim(-0.7, ns - 0.3); ax.set_ylim(a.z_max, 0)
    ax.set_xticks(range(ns))
    if show_bottom:
        sub = [f"{s['terrane']}\n{R['sub'][k]}" if shallow and R['sub'][k] not in ('no data', 'Vp/Vs only')
               else s['terrane'] for k, s in enumerate(sites)]
        ax.set_xticklabels(sub, rotation=45, ha='right', fontsize=7.5)
        for tl, s in zip(ax.get_xticklabels(), sites):
            tl.set_color(t2.terrane_colour(s['terrane']))
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(R['zlabel'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.grid(True, axis='y', lw=0.3, color='0.85'); ax.set_axisbelow(True)
    tick_every = max(1, int(np.ceil(nb / 10)))
    cb = fig.colorbar(pm, ax=ax, orientation='vertical', fraction=0.025, pad=0.015, extend='both',
                      ticks=bounds[::tick_every])
    cb.set_label(f"{R['clabel']}  (bins of {st:g})"); cb.ax.tick_params(labelsize=8)
    return pm


def from_shallow(a, sites):
    from plot_site_shallow import load_dir
    Pd = load_dir(a.shallow)
    zmax = a.z_max
    zq = np.arange(0, zmax + 1e-6, a.dz)
    P, flags, info = [], [], []
    for s in sites:
        p = Pd.get(s['no'])
        if p is None or p['kind'] != 'vs':
            P.append(np.full(len(zq), np.nan))
            flags.append('')
            info.append('no data' if p is None else 'Vp/Vs only')
            continue
        P.append(np.where(zq <= p['z'].max() + 1e-6, np.interp(zq, p['z'], p['v']), np.nan))
        d = gc_km(s['lon'], s['lat'], p['lon'], p['lat'])
        flags.append(f"station {d:.0f} km" if d > a.max_km else '')
        info.append(p['sta'])
    note = 'median Vs at the nearest station (label under each column)'
    if any(flags):
        note += f"; * hatched: station > {a.max_km:g} km from the site"
    deeper = [s['name'] for s, p in ((s, Pd.get(s['no'])) for s in sites)
              if p is not None and p['z'].max() > zmax]
    if deeper:
        note += f"; cut at {zmax:g} m: {', '.join(deeper)}"
    return dict(z=zq, P=np.array(P), flag=flags, moho=[np.nan] * len(sites), sub=info,
                title='Shallow Vs columns at the study sites', clabel='Vs (m/s)', zlabel='depth (m)',
                note=note)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs', default=os.environ.get('FVS2'), help='Fvs npz[:LABEL] (default $FVS2)')
    ap.add_argument('--shallow', default=None, help='folder or .zip of shallow station profiles instead of --fvs')
    ap.add_argument('--field', default='vsv', choices=list(LABEL))
    ap.add_argument('--bins', nargs=3, type=float, default=None, metavar=('LO', 'HI', 'STEP'),
                    help='colour bins (default: model vsv 3.0 4.5 0.25; shallow: slowest value to the '
                         '90th percentile of the profile maxima, 250 m/s steps)')
    ap.add_argument('--cmap', default=None, help="default turbo_r (warm = slow); RdBu for xi, viridis for Vp/Vs")
    ap.add_argument('--sites', default=os.path.join(_HERE, '..', 'examples', 'yilgarn_sites_19.txt'))
    ap.add_argument('--order', choices=['file', 'terrane'], default='file')
    ap.add_argument('--z-max', type=float, default=None, help='default 50 km (model) / 650 m (shallow)')
    ap.add_argument('--dz', type=float, default=None, help='default 0.5 km / 1 m')
    ap.add_argument('--idw', type=int, default=0)
    ap.add_argument('--max-km', type=float, default=50.0)
    ap.add_argument('--min-ndat', type=int, default=35)
    ap.add_argument('--moho', default=os.environ.get('MOHO'))
    ap.add_argument('--moho-mat', dest='moho_mat', default=None,
                    help='CCP refined-Moho .mat (moho_ccp_refined.final.mat): shades each Moho bar by '
                         'sharpness (amp_moho/confidence; dark=strong -> light=weak) and dashes it where '
                         'is_missing. Overrides --moho for the bar depth.')
    ap.add_argument('--width', type=float, default=0.62, help='column width (fraction of the spacing)')
    ap.add_argument('--figsize', nargs=2, type=float, default=(13.0, 6.2))
    ap.add_argument('--xi-fvs', dest='xi_fvs', default=None,
                    help='add a BOTTOM xi panel from this (ZT) model (npz[:LABEL]); top panel stays '
                         '--fvs/--field. e.g. --xi-fvs $SMOO:"smoomoho xi"')
    ap.add_argument('--crust-layers', dest='crust_layers', type=int, default=0,
                    help='collapse each crust (0->Moho) into N equal-thickness mean blocks + one '
                         'sub-Moho (mantle) mean block, e.g. 3. 0 (default) = full-resolution columns')
    ap.add_argument('--xi-overlay', dest='xi_overlay', action='store_true',
                    help='ONE panel: overlay the --xi-fvs fabric ON the Vsv columns (no separate xi panel)')
    ap.add_argument('--xi-fabric', dest='xi_fabric', choices=['ellipse', 'sticks', 'none'],
                    default='ellipse', help='how to draw the xi fabric on 3-layer plots (default ellipse)')
    ap.add_argument('--fabric-scale', dest='fabric_scale', type=float, default=1.0,
                    help='overall size multiplier for the xi ellipses (default 1.0; try 1.3-1.6 if faint)')
    ap.add_argument('--fabric-min', dest='fabric_min', type=float, default=0.005,
                    help='drop the xi fabric marker where |xi-1| < this (default 0.005 -> only ~0.995-1.005 blank; 0 = all blocks)')
    ap.add_argument('--out', default='figures/sites/site_columns.png')
    a = ap.parse_args()
    shallow = a.shallow is not None
    if a.z_max is None:
        a.z_max = 650.0 if shallow else 50.0
    if a.dz is None:
        a.dz = 1.0 if shallow else 0.5
    if not shallow and not a.fvs:
        sys.exit('give --fvs (or source env_stage6.sh for $FVS2), or --shallow')
    overlay = bool(a.xi_fvs) and a.xi_overlay and not shallow
    two = bool(a.xi_fvs) and not a.xi_overlay and not shallow

    sites = t2.load_sites(a.sites)
    if a.order == 'terrane':
        first = {}
        for s in sites:
            first.setdefault(s['terrane'], len(first))
        sites = sorted(sites, key=lambda s: (first[s['terrane']], sites.index(s)))
    R = from_shallow(a, sites) if shallow else from_model(a, sites)         # top panel (Vsv)
    nl = a.crust_layers
    _topfield = R.get('field', a.field)

    if overlay:                                                            # ONE panel: Vsv + xi fabric
        Rxi = from_model(a, sites, fvs=a.xi_fvs, field='xi')
        fig, ax = plt.subplots(figsize=tuple(a.figsize))
        _draw_panel(fig, ax, R, sites, a, _topfield, shallow, nl, True, True, overlay_xi=Rxi)
        title = (f"{model_label(a.fvs)[1]} {LABEL[_topfield].split(' (')[0]}  +  "
                 f"{model_label(a.xi_fvs)[1]} xi fabric, study sites")
        srcs = [model_label(a.fvs)[0], model_label(a.xi_fvs)[0]]
        note = R['note']
    elif two:
        R2 = from_model(a, sites, fvs=a.xi_fvs, field='xi')                 # bottom panel (xi)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(a.figsize[0], a.figsize[1] * 1.95), sharex=True)
        _draw_panel(fig, ax1, R, sites, a, _topfield, shallow, nl, show_top=True, show_bottom=False)
        _draw_panel(fig, ax2, R2, sites, a, 'xi', shallow, nl, show_top=False, show_bottom=True)
        title = (f"{model_label(a.fvs)[1]} {LABEL[_topfield].split(' (')[0]} (top)  /  "
                 f"{model_label(a.xi_fvs)[1]} xi (bottom)  columns at the study sites")
        srcs = [model_label(a.fvs)[0], model_label(a.xi_fvs)[0]]
        note = R['note']
    else:
        fig, ax = plt.subplots(figsize=tuple(a.figsize))
        _draw_panel(fig, ax, R, sites, a, _topfield, shallow, nl, show_top=True, show_bottom=True)
        title = R['title']
        srcs = [] if shallow else [model_label(a.fvs)[0]]
        note = R['note']

    if np.isfinite(R['moho']).any():
        note += ('; Moho bar shade = sharpness (dark=strong), dashed = missing'
                 if R.get('moho_sharp') is not None else '; black bar: Moho')
    if nl:
        note += f'; crust split into {nl} mean layers + mantle'
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.text(0.5, 0.005, note, ha='center', va='bottom', fontsize=8, color='0.3')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if srcs:
        t2.source_note(fig, srcs)
    fig.savefig(a.out, dpi=220); fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")


if __name__ == '__main__':
    main()
