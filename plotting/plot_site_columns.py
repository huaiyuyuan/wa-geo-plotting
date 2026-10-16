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


def from_model(a, sites):
    zq = np.arange(0, a.z_max + 1e-6, a.dz)
    moho = None
    if a.moho and os.path.isfile(a.moho):
        from plot_domain_profiles import moho_at
        moho = moho_at(a.moho, [s['lon'] for s in sites], [s['lat'] for s in sites])
    path, label = model_label(a.fvs)
    M = sample(path, label, a.field, sites, zq, a, moho)
    note = 'inverse-distance mean of %d nearest nodes' % a.idw if a.idw > 1 else 'nearest model node'
    flags = [i['flag'] for i in M['info']]
    if any(flags):
        note += f"; * hatched: node > {a.max_km:g} km away or Ndat < {a.min_ndat}"
    return dict(z=zq, P=M['P'], flag=flags, moho=[i['moho'] for i in M['info']],
                title=f"{label} {LABEL[a.field].split(' (')[0]} columns at the study sites",
                clabel=LABEL[a.field], zlabel='depth (km)', note=note)


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
    ap.add_argument('--width', type=float, default=0.62, help='column width (fraction of the spacing)')
    ap.add_argument('--figsize', nargs=2, type=float, default=(13.0, 6.2))
    ap.add_argument('--out', default='figures/sites/site_columns.png')
    a = ap.parse_args()
    shallow = a.shallow is not None
    if a.z_max is None:
        a.z_max = 650.0 if shallow else 50.0
    if a.dz is None:
        a.dz = 1.0 if shallow else 0.5
    if not shallow and not a.fvs:
        sys.exit('give --fvs (or source env_stage6.sh for $FVS2), or --shallow')

    sites = t2.load_sites(a.sites)
    if a.order == 'terrane':
        first = {}
        for s in sites:
            first.setdefault(s['terrane'], len(first))
        sites = sorted(sites, key=lambda s: (first[s['terrane']], sites.index(s)))
    R = from_shallow(a, sites) if shallow else from_model(a, sites)

    if a.bins:
        lo, hi, st = a.bins
    elif shallow:
        v = R['P'][np.isfinite(R['P'])]
        st = 250.0
        # bins from the slowest value to the 90th percentile of the profiles' maxima, so a
        # single very fast profile (Cue, ~5.9 km/s) does not wash the rest out; beyond = arrow
        mx = [np.nanmax(p) for p in R['P'] if np.isfinite(p).any()]
        lo, hi = st * np.floor(v.min() / st), st * np.ceil(np.percentile(mx, 90) / st)
    else:
        lo, hi, st = DEFAULT_BINS[a.field]
    bounds = np.round(np.arange(lo, hi + st / 2, st), 6)
    nb = len(bounds) - 1
    base = a.cmap or {'xi': 'RdBu', 'vpvs': 'viridis'}.get(a.field if not shallow else 'vs', 'turbo_r')
    cmap = make_cmap(base, nb)
    norm = BoundaryNorm(bounds, cmap.N)
    print(f"colour bins {lo:g}-{hi:g} every {st:g} ({nb} colours, {base})")

    z = R['z']; ns = len(sites); w = a.width
    ze = np.r_[z[0], 0.5 * (z[1:] + z[:-1]), z[-1]]
    fig, ax = plt.subplots(figsize=tuple(a.figsize))
    pm = None
    for k, s in enumerate(sites):
        v = R['P'][k]
        if np.isfinite(v).any():
            pm = ax.pcolormesh([k - w / 2, k + w / 2], ze, np.ma.masked_invalid(v)[:, None], cmap=cmap,
                               norm=norm, shading='flat')
            zb = z[np.isfinite(v)].max()
        else:
            zb = a.z_max
            ax.text(k, a.z_max / 2, R['sub'][k] if shallow else 'no data', rotation=90, ha='center',
                    va='center', fontsize=8, color='0.5')
        flag = R['flag'][k]
        ax.add_patch(plt.Rectangle((k - w / 2, 0), w, zb, fill=False, ec='k', lw=0.6,
                                   hatch='///' if flag else None, zorder=3))
        mo = R['moho'][k]
        if np.isfinite(mo) and mo <= a.z_max:
            ax.plot([k - w / 2 - 0.08, k + w / 2 + 0.08], [mo, mo], color='k', lw=2.4, zorder=4,
                    solid_capstyle='butt')
        col = t2.terrane_colour(s['terrane'], k)
        ax.text(k - w / 2, -0.02 * a.z_max, f"{s['no']} {s['name']}" + ('*' if flag else ''), rotation=45,
                ha='left', va='bottom', fontsize=8.5, fontweight='bold', color=col, rotation_mode='anchor')
    ax.set_xlim(-0.7, ns - 0.3)
    ax.set_ylim(a.z_max, 0)
    ax.set_xticks(range(ns))
    sub = [f"{s['terrane']}\n{R['sub'][k]}" if shallow and R['sub'][k] not in ('no data', 'Vp/Vs only')
           else s['terrane'] for k, s in enumerate(sites)]
    ax.set_xticklabels(sub, rotation=45, ha='right', fontsize=7.5)
    for tl, s in zip(ax.get_xticklabels(), sites):
        tl.set_color(t2.terrane_colour(s['terrane']))
    ax.set_ylabel(R['zlabel'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.grid(True, axis='y', lw=0.3, color='0.85')
    ax.set_axisbelow(True)
    tick_every = max(1, int(np.ceil(nb / 10)))
    cb = fig.colorbar(pm, ax=ax, orientation='vertical', fraction=0.025, pad=0.015, extend='both',
                      ticks=bounds[::tick_every])
    cb.set_label(f"{R['clabel']}  (bins of {st:g})")
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(R['title'], fontsize=12, y=0.99)
    fig.text(0.5, 0.005, R['note'] + ('; black bar: AR23 Moho' if np.isfinite(R['moho']).any() else ''),
             ha='center', va='bottom', fontsize=8, color='0.3')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if not a.shallow:
        t2.source_note(fig, [model_label(a.fvs)[0]])
    fig.savefig(a.out, dpi=220); fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")


if __name__ == '__main__':
    main()
