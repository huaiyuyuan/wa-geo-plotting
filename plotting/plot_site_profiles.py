#!/usr/bin/env python3
"""
plot_site_profiles.py - TransD 1-D Vsv profiles at study sites (e.g. the 19 Yilgarn sites
of examples/yilgarn_sites_19.txt), with values every 0.5 km to 70 km.

  source env_stage6.sh
  python3 plotting/plot_site_profiles.py --fvs $FVS2 --sites examples/yilgarn_sites_19.txt \\
      --moho $MOHO --out $FIG/iter2/sites/site_vsv.png

Profile at a site = the model node nearest the site (great-circle distance; printed and
written to the tables), or with --idw K the inverse-distance mean of the K nearest nodes
within --max-km. Sites whose nearest node is beyond --max-km (default 50) or has
Ndat < --min-ndat (35) are flagged. xi (ZT) models give true Vsv (Viso / sqrt((2+xi^2)/3));
--field viso / xi / vpvs for the other quantities.

Outputs
  <out>               one panel per site: profile, +-1 posterior std band (Vsv_err), AR23
                      Moho tick (--moho)
  <stem>.overlay.png  all sites in one panel, coloured by terrane
  <stem>.csv          depth (0, 0.5 ... --z-max) x site, km/s
  <stem>.err.csv      same for the posterior std (when the model has Vsv_err)
  <stem>.nodes.txt    the node used for each site: lon, lat, distance, Ndat, AR23 Moho
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', 'basemap'))
import tomo2d as t2                                           # noqa: E402
from domains import model_fields                              # noqa: E402

LABEL = {'vsv': 'Vsv (km/s)', 'viso': 'Viso (km/s)', 'xi': 'xi = Vsh/Vsv', 'vpvs': 'Vp/Vs'}


def gc_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return 6371.0 * np.arccos(np.clip(c, -1, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs', default=os.environ.get('FVS2'), help='Fvs npz (default $FVS2)')
    ap.add_argument('--label', default=None, help='model label (default from the file name)')
    ap.add_argument('--field', default='vsv', choices=list(LABEL))
    ap.add_argument('--sites', default=os.path.join(_HERE, '..', 'examples', 'yilgarn_sites_19.txt'))
    ap.add_argument('--dz', type=float, default=0.5)
    ap.add_argument('--z-max', type=float, default=70.0)
    ap.add_argument('--idw', type=int, default=0, help='inverse-distance mean of K nearest nodes (0 = nearest)')
    ap.add_argument('--max-km', type=float, default=50.0)
    ap.add_argument('--min-ndat', type=int, default=35)
    ap.add_argument('--moho', default=os.environ.get('MOHO'), help='AR23 Moho file (tick on each panel)')
    ap.add_argument('--xlim', nargs=2, type=float, default=None)
    ap.add_argument('--ncol', type=int, default=5)
    ap.add_argument('--out', default='figures/sites/site_vsv.png')
    a = ap.parse_args()
    if not a.fvs:
        sys.exit('give --fvs (or source env_stage6.sh for $FVS2)')

    d = dict(np.load(a.fvs, allow_pickle=True))
    import re
    m = re.search(r'iter\.?(\d+)', os.path.basename(a.fvs))
    label = a.label or (f'iter{m.group(1)}' if m else os.path.basename(a.fvs))
    F = model_fields(d)
    if a.field not in F:
        sys.exit(f"{os.path.basename(a.fvs)} has no varying '{a.field}' (has: {', '.join(F)})")
    A = np.asarray(F[a.field], float)
    E = np.asarray(d['Vsv_err'], float) if (a.field == 'vsv' and 'Vsv_err' in d
                                              and 'Xi' not in F) else None
    if a.field == 'viso' and 'Vsv_err' in d:
        E = np.asarray(d['Vsv_err'], float)                 # xi models: Vsv_err belongs to Viso
    z = np.asarray(d['z'], float)
    lon, lat = np.asarray(d['Lon'], float), np.asarray(d['Lat'], float)
    ndat = np.asarray(d['Ndat'], float) if 'Ndat' in d else np.full(len(lon), np.inf)
    zq = np.arange(0, a.z_max + 1e-6, a.dz)
    if zq[-1] > z[-1]:
        print(f"  (model stops at {z[-1]:g} km; deeper values left blank)")

    sites = t2.load_sites(a.sites)
    moho = None
    if a.moho and os.path.isfile(a.moho):
        from plot_domain_profiles import moho_at
        moho = moho_at(a.moho, [s['lon'] for s in sites], [s['lat'] for s in sites])
    P, Pe, info = [], [], []
    for k, s in enumerate(sites):
        dist = gc_km(s['lon'], s['lat'], lon, lat)
        order = np.argsort(dist)
        if a.idw > 1:
            use = [i for i in order[:a.idw] if dist[i] <= a.max_km] or [order[0]]
            w = 1.0 / np.maximum(dist[use], 1.0)
            prof = (A[use] * w[:, None]).sum(0) / w.sum()
            err = (E[use] * w[:, None]).sum(0) / w.sum() if E is not None else None
            i0 = use[0]
        else:
            i0 = order[0]
            prof, err = A[i0], (E[i0] if E is not None else None)
        P.append(np.interp(zq, z, prof, right=np.nan))
        Pe.append(np.interp(zq, z, err, right=np.nan) if err is not None else None)
        flag = []
        if dist[i0] > a.max_km:
            flag.append(f'beyond {a.max_km:g} km')
        if ndat[i0] < a.min_ndat:
            flag.append(f'Ndat {ndat[i0]:.0f}')
        info.append(dict(lon=lon[i0], lat=lat[i0], dist=dist[i0], ndat=ndat[i0], flag='; '.join(flag),
                         moho=(moho[k] if moho is not None else np.nan),
                         nuse=(len(use) if a.idw > 1 else 1)))
        print(f"  {s['no']:2d} {s['name']:<10s} {s['terrane']:<14s} node {lon[i0]:7.2f} {lat[i0]:7.2f} "
              f"{dist[i0]:5.1f} km  Ndat {ndat[i0]:4.0f}" + (f"   ** {info[-1]['flag']}" if flag else ''))
    P = np.array(P)

    xl = a.xlim or (np.floor(np.nanmin(P) * 5) / 5, np.ceil(np.nanmax(P) * 5) / 5)
    ns = len(sites); nc = a.ncol; nr = int(np.ceil(ns / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.15 * nc + 0.4, 3.0 * nr + 0.5), sharex=True, sharey=True,
                            squeeze=False)
    for k, s in enumerate(sites):
        ax = axs.flat[k]
        col = t2.terrane_colour(s['terrane'], k)
        if Pe[k] is not None:
            ax.fill_betweenx(zq, P[k] - Pe[k], P[k] + Pe[k], color=col, alpha=0.18, lw=0)
        ax.plot(P[k], zq, color=col, lw=1.5)
        if np.isfinite(info[k]['moho']):
            ax.plot([0.82, 1.0], [info[k]['moho']] * 2, color='k', lw=2.2,
                    transform=ax.get_yaxis_transform(), solid_capstyle='butt')
        ax.set_title(f"{s['no']} {s['name']}", fontsize=8.5, loc='left', pad=2, color=col, fontweight='bold')
        ax.text(0.04, 0.03, f"{s['terrane']}\nnode {info[k]['dist']:.0f} km" +
                (f"\n{info[k]['flag']}" if info[k]['flag'] else ''),
                transform=ax.transAxes, fontsize=6.3, va='bottom',
                color='#b03a2e' if info[k]['flag'] else '0.35')
        ax.grid(True, lw=0.3, color='0.85'); ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[ns:]:
        ax.axis('off')
    ax0 = axs.flat[0]
    ax0.set_ylim(a.z_max, 0); ax0.set_xlim(*xl)
    for ax in axs[:, 0]:
        ax.set_ylabel('depth (km)', fontsize=8)
    for c in range(nc):
        col_axes = [axs[r, c] for r in range(nr) if r * nc + c < ns]
        if col_axes:
            col_axes[-1].xaxis.set_tick_params(labelbottom=True)
            col_axes[-1].set_xlabel(LABEL[a.field], fontsize=8)
    how = (f'inverse-distance mean of {a.idw} nearest nodes' if a.idw > 1 else 'nearest model node')
    fig.suptitle(f"{label} {LABEL[a.field].split(' (')[0]} at the study sites ({how}; band: ±1 posterior std"
                 + ('; black tick: AR23 Moho' if moho is not None else '') + ')', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=220); fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")

    stem = os.path.splitext(a.out)[0]
    fig, ax = plt.subplots(figsize=(5.6, 6.4))
    terr = []
    for k, s in enumerate(sites):
        ax.plot(P[k], zq, color=t2.terrane_colour(s['terrane'], k), lw=1.1, alpha=0.9)
        if s['terrane'] not in terr:
            terr.append(s['terrane'])
    ax.set_ylim(a.z_max, 0); ax.set_xlim(*xl)
    ax.set_xlabel(LABEL[a.field]); ax.set_ylabel('depth (km)'); ax.grid(True, lw=0.3, color='0.85')
    ax.legend(handles=[Line2D([], [], color=t2.terrane_colour(t, i), lw=2, label=t) for i, t in enumerate(terr)],
              fontsize=7.5, loc='lower left', frameon=True, framealpha=0.9, title='terrane', title_fontsize=8)
    ax.set_title(f"{label}: {ns} study sites", fontsize=10)
    fig.tight_layout()
    fig.savefig(stem + '.overlay.png', dpi=220); fig.savefig(stem + '.overlay.pdf')
    plt.close(fig)
    print(f"Saved {stem}.overlay.png (+ .pdf)")

    head = ','.join(f"{s['no']}_{s['name']}" for s in sites)
    with open(stem + '.csv', 'w') as f:
        f.write(f"# {label} {a.field} (km/s) at study sites, {how}; model {os.path.basename(a.fvs)}\n")
        f.write('depth_km,' + head + '\n')
        for i, zz in enumerate(zq):
            f.write(f"{zz:g}," + ','.join('NaN' if not np.isfinite(P[k, i]) else f"{P[k, i]:.4f}"
                                          for k in range(ns)) + '\n')
    if all(e is not None for e in Pe):
        Pe = np.array(Pe)
        with open(stem + '.err.csv', 'w') as f:
            f.write(f"# {label} posterior std of {a.field} (km/s) at study sites\n")
            f.write('depth_km,' + head + '\n')
            for i, zz in enumerate(zq):
                f.write(f"{zz:g}," + ','.join('NaN' if not np.isfinite(Pe[k, i]) else f"{Pe[k, i]:.4f}"
                                              for k in range(ns)) + '\n')
    with open(stem + '.nodes.txt', 'w') as f:
        f.write(f"# no name terrane site_lon site_lat node_lon node_lat dist_km Ndat AR23_moho_km flag  ({how})\n")
        for s, q in zip(sites, info):
            f.write(f"{s['no']:2d} {s['name']:<10s} {s['terrane'].replace(' ', '_'):<14s} {s['lon']:8.3f} "
                    f"{s['lat']:8.3f} {q['lon']:8.3f} {q['lat']:8.3f} {q['dist']:6.1f} {q['ndat']:5.0f} "
                    f"{q['moho']:6.1f} {q['flag'].replace(' ', '_') or '-'}\n")
    print(f"Tables: {stem}.csv" + (', .err.csv' if all(e is not None for e in Pe) else '') + ', .nodes.txt')


if __name__ == '__main__':
    main()
