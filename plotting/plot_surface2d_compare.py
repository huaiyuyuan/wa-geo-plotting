#!/usr/bin/env python3
"""
plot_surface2d_compare.py - compare surface-wave 2-D tomography runs side by side
(e.g. the same period/component at different transdimensional cell caps).

1 x N maps (one column per run directory), all on ONE shared colour scale:
  --field anom  (default) velocity anomaly, % vs a common reference
  --field std   posterior standard deviation (Standard_deviation.out)
  --field both  2 x N: anomaly row + std row (each row its own shared scale)
Each panel is labelled with its run label, mean cell count, hierarchical sigma
(and travel-time residual if given); the array footprint is outlined.

  python3 plot_surface2d_compare.py \\
      --dirs $P/OUT.iter.2.k200 $P/OUT.iter.2 $P/OUT.iter.2.n700 \\
      --labels "cap 200" "cap 400" "cap 700" \\
      --ncells 196 320 553 --sigmas 1.641 1.889 2.528 --resid 2.170 2.166 2.194 \\
      --outline $FP/wa_array_outline.txt --title "T phase 15 s" \\
      --out $P/ncell_compare_T15.png

Grid files (Average.out, Standard_deviation.out), as written by rj_tomo:
  line 1  nvt nvp          (n latitude, n longitude - verified on period.15.0)
  line 2  latmax lonmin
  line 3  dlat dlon
  then (nvp+2)*(nvt+2) values, one per line (first column), longitude index
  outermost and latitude innermost, with a one-cell ghost ring; latitude rows
  start at latmax and step south.
The file layout (header order of line 1, byte order) is chosen from the data:
a wrong byte order gives stripes, and a swapped header gives a transposed map that
fails the footprint test (posterior std must be lowest inside the array). The
candidate table is printed; --layout forces one.
"""
import argparse, importlib.util, os, sys
import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.path import Path as MPath


# ----------------------------------------------------------------------------- reading
def _parse(path):
    with open(path) as f:
        h = [f.readline().split() for _ in range(3)]
    n1, n2 = int(h[0][0]), int(h[0][1])
    latmax, lonmin = float(h[1][0]), float(h[1][1])
    dlat, dlon = float(h[2][0]), float(h[2][1])
    vals = np.loadtxt(path, skiprows=3, usecols=0, ndmin=1)
    need = (n1 + 2) * (n2 + 2)
    if vals.size < need:
        sys.exit(f"{path}: {vals.size} values, expected (n1+2)*(n2+2) = {need}")
    return n1, n2, latmax, lonmin, dlat, dlon, vals[:need]


def _roughness(g):
    """Mean |neighbour difference| relative to the field's spread (stripes -> large)."""
    g = np.asarray(g, float)
    s = np.nanstd(g) or 1.0
    return (np.nanmean(np.abs(np.diff(g, axis=0))) + np.nanmean(np.abs(np.diff(g, axis=1)))) / s


LAYOUTS = [('nvt nvp', 'latfast'), ('nvt nvp', 'lonfast'),
           ('nvp nvt', 'latfast'), ('nvp nvt', 'lonfast')]


def read_grid(path, layout=('nvt nvp', 'latfast')):
    """-> lat (ascending), lon, grid [lat, lon] without the ghost ring.
    layout = (header order of line 1, byte order): 'latfast' = latitude index fastest
    (longitude outer); 'lonfast' = longitude fastest. Rows start at latmax, step south."""
    n1, n2, latmax, lonmin, dlat, dlon, v = _parse(path)
    hdr, order = layout
    nvt, nvp = (n1, n2) if hdr == 'nvt nvp' else (n2, n1)
    if order == 'latfast':
        g = v.reshape(nvp + 2, nvt + 2).T[1:-1, 1:-1]
    else:
        g = v.reshape(nvt + 2, nvp + 2)[1:-1, 1:-1]
    lat = latmax - dlat * np.arange(nvt)
    lon = lonmin + dlon * np.arange(nvp)
    return lat[::-1], lon, g[::-1]


def choose_layout(avg_path, std_path=None, inside_fn=None):
    """Try the 4 header/byte-order layouts on one run. A wrong byte order gives
    stripes (rough); a swapped header + other byte order gives a smooth but
    TRANSPOSED map, which only geography can catch: posterior std is lowest where
    paths are dense, i.e. inside the footprint. Returns the chosen layout."""
    rows = []
    for lay in LAYOUTS:
        lat, lon, g = read_grid(avg_path, lay)
        r = _roughness(g)
        c = np.nan
        if std_path and inside_fn is not None:
            _, _, s_ = read_grid(std_path, lay)
            ins = inside_fn(lat, lon)
            if ins.any() and (~ins).any():
                c = np.nanmean(s_[~ins]) / max(np.nanmean(s_[ins]), 1e-12)
        rows.append((lay, r, c, lat, lon))
    rmin = min(r for _, r, _, _, _ in rows)
    smooth = [x for x in rows if x[1] <= 1.3 * rmin]
    if all(np.isfinite(x[2]) for x in smooth):
        best = max(smooth, key=lambda x: x[2])
        why = 'smoothest + lowest std inside footprint'
    else:
        best = next(x for x in smooth if x[0] == ('nvt nvp', 'latfast')) \
            if any(x[0] == ('nvt nvp', 'latfast') for x in smooth) else smooth[0]
        why = 'smoothest (no footprint/std to separate transposed layouts)'
    print("grid layout candidates (first run):")
    for lay, r, c, lat, lon in rows:
        mark = '  <== chosen' if lay == best[0] else ''
        print(f"  header '{lay[0]}', {lay[1]:7s}: roughness {r:6.3f}  "
              f"std out/in footprint {c:5.2f}  lat {lat[0]:.2f}..{lat[-1]:.2f} "
              f"lon {lon[0]:.2f}..{lon[-1]:.2f}{mark}")
    print(f"  -> {best[0]} ({why})")
    return best[0]


def footprint_orientation_check(lat, lon, std, inside, label=''):
    """Posterior std is lowest where paths are dense (inside the footprint). Compare
    the outside/inside std contrast for the 4 N-S/E-W flips: 'as read' should win."""
    out = {}
    for name, f in (('as read', lambda x: x), ('N-S flipped', lambda x: x[::-1]),
                    ('E-W flipped', lambda x: x[:, ::-1]), ('both', lambda x: x[::-1, ::-1])):
        s = f(std)
        out[name] = np.nanmean(s[~inside]) / max(np.nanmean(s[inside]), 1e-12)
    best = max(out, key=out.get)
    msg = ', '.join(f"{k} {v:.2f}" for k, v in out.items())
    flag = '' if best == 'as read' else '   <-- WARNING: a flipped grid fits the footprint better'
    print(f"  orientation check {label}: std outside/inside footprint: {msg}{flag}")
    return best == 'as read'


def load_pipeline_reader(module_path):
    """Take imports, functions, classes and constants from the pipeline script and
    run only those - its top-level code (argparse, main()) is never executed."""
    import ast, types
    src = open(module_path).read()
    tree = ast.parse(src, module_path)
    ns = {'__name__': 'pipeline_reader', '__file__': module_path}
    sys.path.insert(0, os.path.dirname(os.path.abspath(module_path)))
    is_const = lambda v: isinstance(v, (ast.Constant, ast.Tuple, ast.List, ast.Dict,
                                        ast.UnaryOp, ast.BinOp))
    for node in tree.body:
        keep = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) \
            or (isinstance(node, ast.Assign) and is_const(node.value))
        if not keep:
            continue
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), module_path, 'exec'), ns)
        except Exception as e:            # an unavailable import etc. - skip it
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            raise
    if 'read_grid' not in ns:
        sys.exit(f"{module_path}: no read_grid() found")
    return types.SimpleNamespace(read_grid=ns['read_grid'])


def check_reader(module_path, grid_path, layout):
    """Compare our grid with the pipeline's read_grid(); report orientation match."""
    mod = load_pipeline_reader(module_path)
    ref = mod.read_grid(grid_path)
    arrays = [ref] if isinstance(ref, np.ndarray) else \
        [v for v in (ref.values() if isinstance(ref, dict) else ref)
         if isinstance(v, np.ndarray)]
    arrays = [a for a in arrays if a.ndim == 2]
    lat, lon, mine = read_grid(grid_path, layout)
    print(f"  our reader: header '{layout[0]}', {layout[1]}")
    print(f"check-reader: pipeline read_grid returned 2-D arrays "
          f"{[a.shape for a in arrays]}; ours {mine.shape} (lat ascending, lon)")
    cands = {'identical': lambda a: a, 'lat flipped': lambda a: a[::-1],
             'lon flipped': lambda a: a[:, ::-1], 'transposed': lambda a: a.T,
             'transposed+lat flipped': lambda a: a.T[::-1]}
    for a in arrays:
        for core in (a, a[1:-1, 1:-1] if min(a.shape) > 2 else a):
            for name, fn in cands.items():
                b = fn(mine)
                if b.shape == core.shape and np.allclose(b, core, equal_nan=True):
                    tag = '' if core is a else ' (after dropping its ghost ring)'
                    print(f"  MATCH: pipeline grid = ours {name}{tag}")
                    if name == 'identical':
                        print("  -> orientation agrees with the pipeline reader.")
                    elif name == 'lat flipped':
                        print("  -> same grid; pipeline stores north row first. Maps agree.")
                    else:
                        print("  -> ORIENTATION DIFFERS - send this output before using the figure.")
                    return True
    print("  NO MATCH in any orientation - the format differs from the spec; "
          "send assemble_surface2d_results.py.")
    return False


def read_outline(path):
    xy = np.loadtxt(path, comments='#', usecols=(0, 1))
    return xy[:, 0], xy[:, 1]


# ----------------------------------------------------------------------------- plotting
def _cmap(name, n):
    return ListedColormap(plt.get_cmap(name)(np.linspace(0, 1, n)))


def _map_axes(fig, rect, extent, cartopy_dir):
    try:
        import cartopy
        if cartopy_dir and os.path.isdir(cartopy_dir):
            cartopy.config['data_dir'] = cartopy_dir
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines('50m', lw=0.7, zorder=6)
        ax.add_feature(cfeature.STATES.with_scale('50m'), lw=0.4, edgecolor='0.3',
                       facecolor='none', zorder=6)
        gl = ax.gridlines(draw_labels=True, lw=0.3, color='0.6', alpha=0.6)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {'size': 7}
        return ax, ccrs.PlateCarree()
    except Exception:
        ax = fig.add_axes(rect)
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
        ax.set_aspect(1 / np.cos(np.radians(np.mean(extent[2:]))))
        ax.tick_params(labelsize=7)
        return ax, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dirs', nargs='+', required=True, help='run directories, left to right')
    ap.add_argument('--labels', nargs='+', help='panel labels (default: directory names)')
    ap.add_argument('--ncells', nargs='+', type=float, help='mean cell count per run')
    ap.add_argument('--sigmas', nargs='+', type=float, help='hierarchical sigma per run')
    ap.add_argument('--resid', nargs='+', type=float, help='travel-time residual (s) per run')
    ap.add_argument('--field', default='anom', choices=['anom', 'std', 'both'])
    ap.add_argument('--ref', default='common', choices=['common', 'each'],
                    help='anomaly reference: common = one mean velocity inside the footprint '
                         'over all runs (differences between panels are real); '
                         'each = each run vs its own footprint mean')
    ap.add_argument('--clim', type=float, help='anomaly colour limit +/- %% (default: robust, shared)')
    ap.add_argument('--std-max', type=float, help='std colour max km/s (default: robust, shared)')
    ap.add_argument('--ncolors', type=int, default=16)
    ap.add_argument('--outline', help='footprint outline: lon lat vertices (text)')
    ap.add_argument('--footprint', help='footprint npz with mask, lat2d, lon2d '
                    '(wa_array_footprint.npz); its mask edge is drawn')
    ap.add_argument('--mask-outside', type=float, default=0.0,
                    help='grey veil outside the footprint, alpha 0-1 (default 0 = none)')
    ap.add_argument('--cartopy-data', default='/data/25/yuan/mfiles/cartopy_data')
    ap.add_argument('--avg-name', default='Average.out')
    ap.add_argument('--std-name', default='Standard_deviation.out')
    ap.add_argument('--title', default='')
    ap.add_argument('--layout', default='auto',
                    choices=['auto'] + [f"{h.replace(' ', '-')}-{o}" for h, o in LAYOUTS],
                    help='grid file layout: header order of line 1 + byte order '
                         '(default auto: chosen from the data, table printed)')
    ap.add_argument('--check-reader', help='optional: path to assemble_surface2d_results.py; '
                    'compare its read_grid() with ours on the first run, then exit')
    ap.add_argument('--out', default='surface2d_compare.png')
    a = ap.parse_args()

    n = len(a.dirs)
    for name, v in (('labels', a.labels), ('ncells', a.ncells), ('sigmas', a.sigmas),
                    ('resid', a.resid)):
        if v is not None and len(v) != n:
            sys.exit(f"--{name}: {len(v)} values for {n} directories")
    labels = a.labels or [os.path.basename(os.path.normpath(d)) for d in a.dirs]

    # ---- footprint: inside-test + drawing, from outline text or npz mask
    inside_fn = draw_fp = None
    if a.outline:
        olon, olat = read_outline(a.outline)
        poly = MPath(np.column_stack([olon, olat]))
        def inside_fn(lat, lon):
            LO, LA = np.meshgrid(lon, lat)
            return poly.contains_points(np.column_stack([LO.ravel(), LA.ravel()])).reshape(LO.shape)
        def draw_fp(ax, kw):
            ax.plot(olon, olat, 'k-', lw=1.3, zorder=7, **kw)
    elif a.footprint:
        fp = np.load(a.footprint)
        flat, flon, fm = np.asarray(fp['lat2d'], float), np.asarray(fp['lon2d'], float), \
            np.asarray(fp['mask'], float)
        latc, lonc = flat[:, 0], flon[0, :]
        def inside_fn(lat, lon):
            i = np.abs(lat[:, None] - latc[None, :]).argmin(1)
            j = np.abs(lon[:, None] - lonc[None, :]).argmin(1)
            ins = fm[np.ix_(i, j)] > 0.5
            out = (lat < latc.min()) | (lat > latc.max())
            ins[out, :] = False
            ins[:, (lon < lonc.min()) | (lon > lonc.max())] = False
            return ins
        def draw_fp(ax, kw):
            ax.contour(flon, flat, fm, levels=[0.5], colors='k', linewidths=1.3,
                       zorder=7, **kw)

    # ---- grid layout: chosen once from the first run, applied to every file
    avg0 = os.path.join(a.dirs[0], a.avg_name)
    std0 = os.path.join(a.dirs[0], a.std_name)
    if a.layout == 'auto':
        layout = choose_layout(avg0, std0 if os.path.exists(std0) else None, inside_fn)
    else:
        h1, h2, o = a.layout.split('-')
        layout = (f"{h1} {h2}", o)
    if a.check_reader:
        ok = check_reader(a.check_reader, avg0, layout)
        sys.exit(0 if ok else 1)

    avg, std = [], []
    for dd in a.dirs:
        lat, lon, g = read_grid(os.path.join(dd, a.avg_name), layout)
        avg.append(g)
        sp = os.path.join(dd, a.std_name)
        if os.path.exists(sp):
            std.append(read_grid(sp, layout)[2])
        elif a.field in ('std', 'both'):
            sys.exit(f"{dd}: no {a.std_name}")
    if any(x.shape != avg[0].shape for x in avg):
        sys.exit(f"runs are on different grids: {[x.shape for x in avg]}")
    print(f"grids: {avg[0].shape} [lat, lon], layout header '{layout[0]}', {layout[1]}; "
          f"lat {lat[0]:.2f}..{lat[-1]:.2f}, lon {lon[0]:.2f}..{lon[-1]:.2f}")

    inside = np.ones((len(lat), len(lon)), bool)
    if inside_fn is not None:
        inside = inside_fn(lat, lon)
        print(f"footprint: {inside.sum()} of {inside.size} grid nodes inside")
        if len(std) == n:
            for lab_, s_ in zip(labels, std):
                footprint_orientation_check(lat, lon, s_, inside, lab_)

    rows = []
    if a.field in ('anom', 'both'):
        if a.ref == 'common':
            ref = [np.nanmean([np.nanmean(g[inside]) for g in avg])] * n
        else:
            ref = [np.nanmean(g[inside]) for g in avg]
        an = [(g / r - 1) * 100 for g, r in zip(avg, ref)]
        lim = a.clim or float(np.nanpercentile(np.abs(np.concatenate(
            [x[inside] for x in an])), 98))
        rows.append(dict(data=an, cmap=_cmap('RdBu', a.ncolors), vmin=-lim, vmax=lim,
                         extend='both', cblab=f'dV/V (%)  vs {"common" if a.ref == "common" else "own"} '
                                              f'footprint mean ({np.mean(ref):.3f} km/s)'))
        print(f"anomaly: reference {', '.join(f'{r:.4f}' for r in ref)} km/s; shared limit +/-{lim:.2f} %")
    if a.field in ('std', 'both'):
        smax = a.std_max or float(np.nanpercentile(np.concatenate([x[inside] for x in std]), 98))
        rows.append(dict(data=std, cmap=_cmap('YlOrRd', a.ncolors), vmin=0, vmax=smax,
                         extend='max', cblab='posterior std (km/s)'))
        print(f"std: shared 0-{smax:.4f} km/s")

    # layout in inches: N map columns + one colourbar per row
    extent = [lon[0] - 0.5, lon[-1] + 0.5, lat[0] - 0.5, lat[-1] + 0.5]
    pw = 3.4
    ph = pw * (extent[3] - extent[2]) / ((extent[1] - extent[0]) * np.cos(np.radians(np.mean(extent[2:]))))
    L, R, T, B, G, Gv = 0.45, 0.95, 0.75 if a.title else 0.45, 0.4, 0.25, 0.55
    nr = len(rows)
    W = L + n * pw + (n - 1) * G + R
    H = T + nr * ph + (nr - 1) * Gv + B
    fig = plt.figure(figsize=(W, H))
    for r, row in enumerate(rows):
        y0 = H - T - (r + 1) * ph - r * Gv
        for c in range(n):
            x0 = L + c * (pw + G)
            ax, tr = _map_axes(fig, [x0 / W, y0 / H, pw / W, ph / H], extent, a.cartopy_data)
            kw = {'transform': tr} if tr else {}
            norm = BoundaryNorm(np.linspace(row['vmin'], row['vmax'], a.ncolors + 1),
                                a.ncolors, clip=True)   # beyond the limits -> end colours
            im = ax.pcolormesh(lon, lat, row['data'][c], cmap=row['cmap'], norm=norm,
                               shading='nearest', zorder=1, **kw)
            if a.mask_outside and inside_fn is not None:
                ax.contourf(lon, lat, (~inside).astype(float), levels=[0.5, 1.5],
                            colors=['0.6'], alpha=a.mask_outside, zorder=2, **kw)
            if draw_fp is not None:
                draw_fp(ax, kw)
            lab = labels[c]
            if a.ncells: lab += f"\n{a.ncells[c]:.0f} cells"
            if a.sigmas: lab += f",  $\\sigma_h$ = {a.sigmas[c]:.3f}"
            if a.resid: lab += f"\nresidual {a.resid[c]:.3f} s"
            ax.text(0.03, 0.03, lab, transform=ax.transAxes, fontsize=8, va='bottom',
                    ha='left', zorder=8, bbox=dict(fc='w', ec='0.5', alpha=0.9, pad=2.5))
        cax = fig.add_axes([(W - R + 0.15) / W, y0 / H + 0.1 * ph / H, 0.14 / W, 0.8 * ph / H])
        cb = fig.colorbar(im, cax=cax, extend=row['extend'])
        lev = np.linspace(row['vmin'], row['vmax'], a.ncolors + 1)
        cb.set_ticks(lev[::max(1, a.ncolors // 8)])
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(
            '%.1f' if row['vmax'] - row['vmin'] > 1 else '%.3f'))
        cb.set_label(row['cblab'], fontsize=8)
        cb.ax.tick_params(labelsize=7)
    if a.title:
        fig.suptitle(a.title, fontsize=11, fontweight='bold', y=1 - 0.15 / H, va='top')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches='tight')
    fig.savefig(os.path.splitext(a.out)[0] + '.pdf', bbox_inches='tight')
    print(f"Saved: {a.out} (+ .pdf)")


if __name__ == '__main__':
    main()
