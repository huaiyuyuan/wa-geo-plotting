#!/usr/bin/env python3
"""
plot_xsection_compare.py - two Vs models side by side, one figure per cross-section.

  columns : model A | model B | (A - B, with --diff)
  rows    : abs  Vs (km/s), one shared colour scale for both models
            rel  dVs (%) vs each model's own whole-model mean at each depth
            err  posterior uncertainty (if both models have it), shared scale
  geology strips, boundary lines and the Moho line on every column, as in
  plot_xsection.py. Distance scale and VE identical in all columns.

  python3 plot_xsection_compare.py \\
      --models $OUT/Fvs.iter.10.Z.npz:"iter10 (no Moho)" $OUT/Fvs.iter.8.ZT.npz:"iter8" \\
      --load-sections $SEC --moho $MOHO --no-moho-mask --d-max 60 --diff \\
      --out-dir $OUT/figures/compare_iter10_iter8 --nproc 4

Model spec: PATH[:LABEL[:VEL]], VEL = auto (default) | model | vsv_true.
auto: an xi (ZT) model is converted to true Vsv = Viso/sqrt((2+Xi^2)/3) so it is
compared like-for-like with Z models; its title says so. 'model' keeps Viso.
Output: <out-dir>/compare.<section>.png (+ .pdf)
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import plot_xsection as px  # noqa: E402  (also sets up config/basemap paths)

ROWS = {'abs': 'Vs (km/s)', 'rel': 'dVs (%) vs model mean', 'err': 'uncertainty (IQR/2)'}


# ----------------------------------------------------------------------------- models
def load_model(spec, d_max):
    parts = spec.split(':')
    path = parts[0]
    label = parts[1] if len(parts) > 1 and parts[1] else None
    vel = parts[2] if len(parts) > 2 and parts[2] else 'auto'
    raw = np.load(path, allow_pickle=True)
    xi_model = 'Xi' in raw.files and np.nanstd(raw['Xi']) > 1e-4
    if vel == 'auto':
        vel = 'vsv_true' if xi_model else 'model'
    d = px._load_fvs(path, vel)
    if label is None:
        import re
        m = re.search(r'iter\.?(\d+)', os.path.basename(path))
        label = f'iter{m.group(1)}' if m else os.path.basename(path).replace('.npz', '')
    vname = 'Vsv' if (vel == 'vsv_true' or not xi_model) else 'Viso'
    note = ' (true Vsv from Viso, xi)' if vel == 'vsv_true' else ''
    z = np.asarray(d['z'], float)
    zm = z <= d_max if d_max else np.ones(len(z), bool)
    V = np.asarray(d['Vsv'], float)[:, zm]
    E = np.asarray(d['Vsv_err'], float)[:, zm] if 'Vsv_err' in d else None
    print(f"  {label}: {os.path.basename(path)} -> {vname}{note}, {len(d['Lon'])} nodes")
    return dict(label=label, vname=vname, note=note, Lon=d['Lon'], Lat=d['Lat'], z=z[zm],
                V=V, E=E, mean=np.nanmean(V, axis=0))


def sample(m, plon, plat):
    v = px._sample_profile(m['Lon'], m['Lat'], m['V'], plon, plat)
    e = px._sample_profile(m['Lon'], m['Lat'], m['E'], plon, plat) if m['E'] is not None else None
    return dict(abs=v, rel=(v / m['mean'][None, :] - 1.0) * 100.0, err=e)


def on_z(arr, z_from, z_to):
    if arr is None or (len(z_from) == len(z_to) and np.allclose(z_from, z_to)):
        return arr
    return np.array([np.interp(z_to, z_from, r, left=np.nan, right=np.nan) for r in arr])


# ----------------------------------------------------------------------------- figure
def plot_one(sec, A, B, a, scale):
    lat1, lon1, lat2, lon2, lab = sec
    plat, plon, dist = px._great_circle_path(lat1, lon1, lat2, lon2, a.ds)
    sA, sB = sample(A, plon, plat), sample(B, plon, plat)
    z = A['z']                                     # common depth axis = model A's
    for k in sB:
        sB[k] = on_z(sB[k], B['z'], z)
    rows = [r for r in a.rows if not (r == 'err' and (sA['err'] is None or sB['err'] is None))]
    moho = None
    if px.plot_section._moho_interp is not None:
        mi = px.plot_section._moho_interp
        moho = np.array([float(np.ravel(mi(la, lo))[0]) for la, lo in zip(plat, plon)])

    # shared colour limits per row
    def both(key):
        return np.concatenate([sA[key][np.isfinite(sA[key])], sB[key][np.isfinite(sB[key])]])
    lims = {}
    if 'abs' in rows:
        if a.vlim:
            lims['abs'] = tuple(a.vlim)
        else:
            f = both('abs'); med, sd = np.nanmedian(f), np.nanstd(f)
            lims['abs'] = (med - a.sigma * sd, med + a.sigma * sd)
    lims['rel'] = (-a.clim_rel, a.clim_rel)
    if 'err' in rows:
        lims['err'] = (0.0, float(np.nanpercentile(both('err'), 90)))
    cm = {'abs': px._cmap(px.CMAP_ABS, a.ncolors), 'rel': px._cmap('RdBu', a.ncolors),
          'err': px._cmap('YlOrRd', a.ncolors), 'diff': px._cmap('RdBu', a.ncolors)}
    diffs = {'abs': ((sA['abs'] / sB['abs'] - 1) * 100, a.clim_diff, f"{A['label']} vs {B['label']} (%)"),
             'rel': (sA['rel'] - sB['rel'], a.clim_rel, 'dVs difference (% points)'),
             'err': ((sA['err'] - sB['err']) if 'err' in rows else None,
                     lims.get('err', (0, 0.1))[1], 'uncertainty difference (km/s)')}

    # layout in inches: identical panel size in every column
    km_per_in, ph = scale
    pw = max(1.2, dist[-1] / km_per_in)
    ncol = 3 if a.diff else 2
    strip_h = px._strip_height_in(px._GEO)
    L, cb, G, T, B_, Gv = 0.7, 0.95, 0.35, 0.95 + strip_h, 0.55, 0.38
    W = L + ncol * (pw + cb) + (ncol - 1) * G
    H = T + len(rows) * ph + (len(rows) - 1) * Gv + B_
    fig = plt.figure(figsize=(W, H))
    D, Z = np.meshgrid(dist, z, indexing='ij')
    zbot = z[-1]
    geo = []
    cols = [(A, sA), (B, sB)] + ([(None, None)] if a.diff else [])
    for c, (M, S) in enumerate(cols):
        x0 = L + c * (pw + cb + G)
        axes = []
        for r, row in enumerate(rows):
            y0 = H - T - (r + 1) * ph - r * Gv
            ax = fig.add_axes([x0 / W, y0 / H, pw / W, ph / H],
                              sharex=axes[0] if axes else None)
            if M is not None:
                data, (vmin, vmax), cmap = S[row], lims[row], cm[row]
                ext = 'max' if row == 'err' else 'both'
                clab = {'abs': f"{M['vname']} (km/s)", 'rel': f"d{M['vname']} (%)",
                        'err': 'IQR/2 (km/s)'}[row]
                title = {'abs': f"{M['vname']} (km/s)", 'rel': f"d{M['vname']} (%) vs model mean",
                         'err': 'posterior uncertainty (IQR/2)'}[row]
            else:
                data, lim, clab = diffs[row]
                if data is None:
                    ax.set_visible(False); axes.append(ax); continue
                vmin, vmax, cmap, ext, title = -lim, lim, cm['diff'], 'both', clab
            im = ax.pcolormesh(D, Z, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            ax.set_ylim(zbot, 0); ax.set_xlim(0, dist[-1])
            ax.set_title(title, fontsize=8, loc='left')
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel('Depth (km)', fontsize=8)
            if r < len(rows) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Distance (km)', fontsize=8)
            if moho is not None:
                px._draw_moho(ax, dist, moho, zbot, a.moho_mask, lw=0.9)
            pos = ax.get_position()
            cax = fig.add_axes([pos.x1 + 0.08 / W, pos.y0, 0.13 / W, pos.height])
            cbar = fig.colorbar(im, cax=cax, extend=ext)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label(clab, fontsize=7)
            axes.append(ax)
        vis = [x for x in axes if x.get_visible()]
        g = px._add_geology(vis[0], vis, plon, plat, dist)
        if c == 0:
            geo.append(g); bottom0 = vis[-1]
        head = (f"{M['label']}{M['note']}" if M is not None else f"{A['label']} minus {B['label']}")
        fig.text((x0 + pw / 2) / W, 1 - 0.42 / H, head, ha='center', va='top',
                 fontsize=10, fontweight='bold')
    vexag = (dist[-1] / pw) / (zbot / ph)
    fig.suptitle(f"{lab}: ({lat1:.1f}°,{lon1:.1f}°) -> ({lat2:.1f}°,{lon2:.1f}°)   "
                 f"[{dist[-1]:.0f} km, VE {vexag:.1f}x]", fontsize=11, fontweight='bold',
                 y=1 - 0.06 / H, va='top')
    if geo and geo[0] is not None:
        px._geology_legend(fig, geo, bottom0, ncol=4)
    out = os.path.join(a.out_dir, f'compare.{lab}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out} (+ .pdf)")


_STATE = {}


def _worker(sec):
    plot_one(sec, _STATE['A'], _STATE['B'], _STATE['a'], _STATE['scale'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs=2, required=True, metavar='PATH[:LABEL[:VEL]]')
    ap.add_argument('--load-sections', required=True)
    ap.add_argument('--sections', nargs='+', help='only these section labels (e.g. sec9 sec11)')
    ap.add_argument('--rows', nargs='+', default=['abs', 'rel', 'err'], choices=list(ROWS))
    ap.add_argument('--diff', action='store_true', help='third column: A minus B for each row')
    ap.add_argument('--moho', default=None)
    ap.add_argument('--moho-mask-alpha', type=float, default=0.5)
    ap.add_argument('--no-moho-mask', action='store_true')
    ap.add_argument('--d-max', type=float, default=60.0)
    ap.add_argument('--ds', type=float, default=0.08)
    ap.add_argument('--ve', type=float, default=px.DEFAULT_VE)
    ap.add_argument('--col-width', type=float, default=6.5,
                    help='width (in) of the longest section in each column (default 6.5)')
    ap.add_argument('--vlim', nargs=2, type=float, default=None, metavar=('VMIN', 'VMAX'),
                    help='abs colour limits (km/s); default median +/- sigma of both models')
    ap.add_argument('--sigma', type=float, default=1.5)
    ap.add_argument('--clim-rel', type=float, default=6.0)
    ap.add_argument('--clim-diff', type=float, default=4.0)
    ap.add_argument('--ncolors', type=int, default=16)
    ap.add_argument('--strips', nargs='?', const=px._STRIPS_DEFAULT,
                    default='domain,boundaries,names',
                    help='geology strips (default domain,boundaries,names; add litho for the '
                         'lithology strip)')
    ap.add_argument('--no-strips', action='store_true')
    ap.add_argument('--domain-colours', default='map2022',
                    choices=['map2022', 'tectcolour', 'palette'])
    ap.add_argument('--bnd-lines', action='store_true',
                    help='also draw boundaries as dashed lines through the panels')
    ap.add_argument('--litho-npz'); ap.add_argument('--tect-npz'); ap.add_argument('--bnd-npz')
    ap.add_argument('--nproc', type=int, default=1)
    ap.add_argument('--out-dir', default='figures/compare')
    a = ap.parse_args()
    a.moho_mask = None if a.no_moho_mask else a.moho_mask_alpha

    print('Models:')
    A, B = (load_model(s, a.d_max) for s in a.models)
    if not a.no_strips and a.strips:
        px._GEO = px._load_geology(a.strips, a.litho_npz, a.tect_npz, a.bnd_npz,
                                   lines=a.bnd_lines, domain_colours=a.domain_colours)
    px.plot_section._moho_interp = None
    if a.moho:
        from scipy.interpolate import RectBivariateSpline
        md = np.loadtxt(a.moho, skiprows=11)
        mla, mlo = np.unique(md[:, 0]), np.unique(md[:, 1])
        px.plot_section._moho_interp = RectBivariateSpline(
            mla, mlo, md[:, 2].reshape(len(mla), len(mlo)), kx=1, ky=1)
    secs = px._load_sections(a.load_sections)
    if a.sections:
        secs = [s for s in secs if s[4] in a.sections]
        if not secs:
            sys.exit(f"none of {a.sections} in {a.load_sections}")
    maxd = max(px._great_circle_path(*s[:4], a.ds)[2][-1] for s in secs)
    km_per_in, ph = px._section_scale(maxd, min(A['z'][-1], a.d_max), a.ve, a.col_width)
    print(f"{len(secs)} sections; longest {maxd:.0f} km -> {km_per_in:.0f} km/in, "
          f"panel height {ph:.2f} in (VE {a.ve:g})")
    os.makedirs(a.out_dir, exist_ok=True)
    _STATE.update(A=A, B=B, a=a, scale=(km_per_in, ph))
    if a.nproc > 1 and len(secs) > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.nproc) as pool:
            pool.map(_worker, secs)
    else:
        for s in secs:
            _worker(s)


if __name__ == '__main__':
    main()
