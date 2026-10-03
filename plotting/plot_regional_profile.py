#!/usr/bin/env python3
"""
plot_regional_profile.py - whole-region average 1-D profiles from a TransD Fvs npz.

  Panel 1: Vsv (Viso for ZT models) vs depth
  Panel 2: xi = Vsh/Vsv (ZT) or Vp/Vs (Z/vpvs runs)

Each panel: regional mean (line), +/-1 sigma (dark band) and +/-2 sigma (light band)
of the LATERAL spread across nodes, and the mean POSTERIOR uncertainty (dotted,
mean +/- mean *_err) - two different things: how much the region varies vs how
well each node is resolved. Optional AR23 Moho: mean +/- 1 sigma at the nodes.

  python3 plot_regional_profile.py --fvs Fvs.iter.3.ZT.npz --moho AR23-moho-hmp.txt \
      --d-max 60 --out figures/regional_profile.png
  # default: xi varies -> Viso + xi (1x2); otherwise Vsv only (1 panel)
  # --second xi | vpvs | none to force
  # --spread pct  -> bands are 16-84 % and 2.5-97.5 % percentiles instead of sigma

Also writes <out>.txt: depth, mean, std, p2.5, p16, p50, p84, p97.5, mean_err per field.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', 'basemap'), os.path.join(_HERE, '..')):
    if os.path.abspath(_p) not in map(os.path.abspath, sys.path):
        sys.path.insert(0, os.path.abspath(_p))

FIELDS = {   # key: (array, error array, label, colour)
    'vsv':  ('Vsv',  'Vsv_err',  None,                   '#1f4e9c'),
    'xi':   ('Xi',   'Xi_err',   r'$\xi$',               '#b2182b'),
    'vpvs': ('Vpvs', 'Vpvs_err', r'$V_P/V_S$',           '#2a7f3f'),
}


def node_mask(d, min_ndat):
    m = np.ones(len(d['Lon']), bool)
    if min_ndat and 'Ndat' in d.files:
        m &= np.asarray(d['Ndat']) >= min_ndat
    return m


def stats(A, spread):
    """Per-depth statistics over nodes (rows). Returns dict of 1-D arrays."""
    with np.errstate(all='ignore'):
        out = dict(mean=np.nanmean(A, 0), std=np.nanstd(A, 0),
                   p2=np.nanpercentile(A, 2.5, 0), p16=np.nanpercentile(A, 16, 0),
                   p50=np.nanpercentile(A, 50, 0), p84=np.nanpercentile(A, 84, 0),
                   p97=np.nanpercentile(A, 97.5, 0), n=np.sum(np.isfinite(A), 0))
    if spread == 'pct':
        out['b1'] = (out['p16'], out['p84']); out['b2'] = (out['p2'], out['p97'])
    else:
        m, s = out['mean'], out['std']
        out['b1'] = (m - s, m + s); out['b2'] = (m - 2 * s, m + 2 * s)
    return out


def moho_at_nodes(path, lat, lon):
    from scipy.interpolate import RectBivariateSpline
    md = np.loadtxt(path, skiprows=11)
    mla, mlo = np.unique(md[:, 0]), np.unique(md[:, 1])
    f = RectBivariateSpline(mla, mlo, md[:, 2].reshape(len(mla), len(mlo)), kx=1, ky=1)
    inside = ((lat >= mla.min()) & (lat <= mla.max()) & (lon >= mlo.min()) & (lon <= mlo.max()))
    return np.array([float(f(a, o)[0, 0]) for a, o in zip(lat[inside], lon[inside])])


def draw(ax, z, st, err, colour, label, spread, moho):
    b2lab = '2.5-97.5 %' if spread == 'pct' else r'$\pm2\sigma$'
    b1lab = '16-84 %' if spread == 'pct' else r'$\pm1\sigma$'
    ax.fill_betweenx(z, *st['b2'], color=colour, alpha=0.15, lw=0, label=f'{b2lab} (lateral)')
    ax.fill_betweenx(z, *st['b1'], color=colour, alpha=0.35, lw=0, label=f'{b1lab} (lateral)')
    ax.plot(st['mean'], z, color=colour, lw=2, label='regional mean')
    if err is not None:
        ax.plot(st['mean'] - err, z, ':', color='k', lw=1, label='mean posterior error')
        ax.plot(st['mean'] + err, z, ':', color='k', lw=1)
    if moho is not None and len(moho):
        mm, ms = np.nanmean(moho), np.nanstd(moho)
        ax.axhspan(mm - ms, mm + ms, color='0.5', alpha=0.25, lw=0, zorder=0)
        ax.axhline(mm, color='0.3', ls='--', lw=1, label=f'Moho {mm:.0f}$\\pm${ms:.0f} km (AR23)')
    ax.set_xlabel(label)
    ax.grid(alpha=0.3, lw=0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs', required=True)
    ap.add_argument('--second', default='auto', choices=['auto', 'xi', 'vpvs', 'none'],
                    help='auto (default): xi varies -> Viso + xi; otherwise Vsv only. '
                         'xi / vpvs / none force panel 2.')
    ap.add_argument('--spread', default='sigma', choices=['sigma', 'pct'],
                    help='bands: +/-1,2 sigma (default) or 16-84 / 2.5-97.5 percentiles')
    ap.add_argument('--d-max', type=float, default=None, help='max depth (km)')
    ap.add_argument('--min-ndat', type=int, default=35,
                    help='only nodes with Ndat >= this (default 35, as the depth slices)')
    ap.add_argument('--moho', default=None, help='AR23 Moho file: mean +/- 1 sigma band')
    ap.add_argument('--no-err', action='store_true', help='omit posterior-uncertainty lines')
    ap.add_argument('--footprint', action='store_true',
                    help='average only nodes inside the WA Array coverage footprint (mask)')
    ap.add_argument('--footprint-npz', default=None, help='override config.FOOTPRINT_NPZ')
    ap.add_argument('--out', default='figures/regional_profile.png')
    a = ap.parse_args()

    d = np.load(a.fvs, allow_pickle=True)
    z = np.asarray(d['z'], float)
    zm = z <= a.d_max if a.d_max else np.ones(len(z), bool)
    z = z[zm]
    m = node_mask(d, a.min_ndat)
    where = ''
    if a.footprint:
        from footprint import Footprint
        fp = Footprint(a.footprint_npz)
        inside = fp.contains(d['Lon'], d['Lat'])
        print('  ' + fp.describe(inside & m) + f" (after Ndat>={a.min_ndat})")
        m &= inside
        where = ', inside array footprint'
    nn = int(m.sum())

    second = a.second
    if second == 'auto':                       # xi model -> Viso + xi ; else Vsv only
        xi_var = 'Xi' in d.files and np.nanstd(d['Xi'][m][:, zm]) > 1e-4
        second = 'xi' if xi_var else 'none'
    if second != 'none':
        key = FIELDS[second][0]
        if key not in d.files:
            sys.exit(f"{os.path.basename(a.fvs)}: no '{key}' array.")
        sd = np.nanstd(d[key][m][:, zm])
        if sd <= 1e-4:
            mean = np.nanmean(d[key][m][:, zm])
            sys.exit(f"{os.path.basename(a.fvs)}: {key} is fixed at {mean:.4f} in this model "
                     f"(spread {sd:.1e}) - nothing to plot. "
                     + ("Use the ZT (xi) Fvs, or " if second == 'xi' else "")
                     + "--second none for a Vsv-only figure.")
    panels = ('vsv',) if second == 'none' else ('vsv', second)

    zt = 'Xi' in d.files and np.nanstd(d['Xi'][m][:, zm]) > 1e-4
    vname = 'Viso' if zt else 'Vsv'
    moho = moho_at_nodes(a.moho, d['Lat'][m], d['Lon'][m]) if a.moho else None

    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels) - 0.6, 5.6),
                             sharey=True, gridspec_kw=dict(wspace=0.08), squeeze=False)
    axes = axes[0]
    table = [z]; cols = ['depth_km']
    for ax, f in zip(axes, panels):
        key, ekey, lab, colour = FIELDS[f]
        A = np.asarray(d[key], float)[m][:, zm]
        st = stats(A, a.spread)
        err = None
        if not a.no_err and ekey in d.files:
            with np.errstate(all='ignore'):
                err = np.nanmean(np.asarray(d[ekey], float)[m][:, zm], 0)
        draw(ax, z, st, err, colour, lab or f'{vname} (km/s)', a.spread, moho)
        if f == 'xi':
            ax.axvline(1.0, color='k', lw=0.6, alpha=0.6)
        for k in ('mean', 'std', 'p2', 'p16', 'p50', 'p84', 'p97'):
            table.append(st[k]); cols.append(f'{key}_{k}')
        table.append(err if err is not None else np.full(len(z), np.nan))
        cols.append(f'{ekey}_mean')

    axes[0].set_ylim(z[-1], z[0])
    axes[0].set_ylabel('Depth (km)')
    axes[0].legend(fontsize=7, loc='lower left', framealpha=0.9)
    fig.suptitle(f'Regional average - {os.path.basename(a.fvs)}  '
                 f'({nn} nodes, Ndat$\\geq${a.min_ndat}{where})', fontsize=10)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches='tight')
    fig.savefig(os.path.splitext(a.out)[0] + '.pdf', bbox_inches='tight')
    txt = os.path.splitext(a.out)[0] + '.txt'
    np.savetxt(txt, np.column_stack(table), fmt='%.5f', header=' '.join(cols))
    print(f"Saved: {a.out} (+ .pdf, {os.path.basename(txt)}) - {nn} nodes, panel 2 = {second}")


if __name__ == '__main__':
    main()
