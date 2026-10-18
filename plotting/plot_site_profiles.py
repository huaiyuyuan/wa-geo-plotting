#!/usr/bin/env python3
"""
plot_site_profiles.py - TransD 1-D Vsv profiles at study sites (e.g. the 19 Yilgarn sites
of examples/yilgarn_sites_19.txt), with values every 0.5 km to 70 km; optionally a second
model drawn on top (e.g. iter2 with a Moho vs iter10 without one).

  source env_stage6.sh
  python3 plotting/plot_site_profiles.py --fvs $FVS2 --fvs2 $FVS10 \\
      --sites examples/yilgarn_sites_19.txt --moho $MOHO --out $FIG/sites/site_vsv.iter2_iter10.png

Profile at a site = the model node nearest the site (great-circle distance; printed and
written to the tables), or with --idw K the inverse-distance mean of the K nearest nodes
within --max-km. Sites whose nearest node is beyond --max-km (default 50) or has
Ndat < --min-ndat (35) are flagged. xi (ZT) models give true Vsv (Viso / sqrt((2+xi^2)/3));
--field viso / xi / vpvs for the other quantities.

Outputs (per model <label>, e.g. iter2, iter10)
  <out>                     one panel per site: --fvs in the terrane colour with +-1 posterior
                            std band; --fvs2 black dashed (band grey); AR23 Moho tick (--moho)
  <stem>.overlay.png        all sites in one panel, coloured by terrane (--fvs2 as a 2nd panel)
  <stem>.<label>.csv        depth (0, 0.5 ... --z-max) x site, km/s
  <stem>.<label>.err.csv    same for the posterior std (when the model has Vsv_err)
  <stem>.<label>.nodes.txt  node used for each site: lon, lat, distance, Ndat, AR23 Moho
  <stem>.diff.csv           --fvs minus --fvs2 (km/s), with --fvs2
With --ref-domain yilgarn: <stem>.dv-yilgarn.png (+ .overlay.png) = each site as dVsv (%)
from the Yilgarn mean of the same model (all its nodes in the Yilgarn with Ndat >= 35), grey
band = +-1 sigma lateral spread of those nodes; tables <stem>.<label>.dv-yilgarn.csv and
<stem>.<label>.yilgarn-mean.csv.
"""
import argparse, os, re, sys
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

LABEL = {'vsv': 'Vsv (km/s)', 'viso': 'Viso (km/s)', 'xi': 'Xi = Vsh/Vsv', 'xi2': 'xi = (Vsh/Vsv)^2',
         'vpvs': 'Vp/Vs'}


def gc_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return 6371.0 * np.arccos(np.clip(c, -1, 1))


def model_label(spec):
    path, _, label = spec.partition(':')
    if not label:
        m = re.search(r'iter\.?(\d+)', os.path.basename(path))
        label = f'iter{m.group(1)}' if m else os.path.basename(path).replace('.npz', '')
    return path, label


def sample(path, label, field, sites, zq, a, moho):
    """-> dict(label, path, P [site, depth], Pe or None, info [per site])"""
    d = dict(np.load(path, allow_pickle=True))
    F = model_fields(d)
    if field not in F:
        sys.exit(f"{os.path.basename(path)} has no varying '{field}' (has: {', '.join(F)})")
    A = np.asarray(F[field], float)
    E = None
    if 'Vsv_err' in d and ((field == 'vsv' and 'xi' not in F) or field == 'viso'):
        E = np.asarray(d['Vsv_err'], float)           # xi models: Vsv_err belongs to Viso
    z = np.asarray(d['z'], float)
    lon, lat = np.asarray(d['Lon'], float), np.asarray(d['Lat'], float)
    ndat = np.asarray(d['Ndat'], float) if 'Ndat' in d else np.full(len(lon), np.inf)
    if not np.nanmax(ndat, initial=0) > 0:              # all-zero Ndat = not filled in (v3 iter1)
        print(f"  ({label}: Ndat is all 0 - no Ndat flag for this model)")
        ndat = np.full(len(lon), np.inf)
    if zq[-1] > z[-1]:
        print(f"  ({label}: model stops at {z[-1]:g} km; deeper values left blank)")
    print(f"{label}: {os.path.basename(path)}")
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
                         moho=(moho[k] if moho is not None else np.nan)))
        print(f"  {s['no']:2d} {s['name']:<10s} {s['terrane']:<14s} node {lon[i0]:7.2f} {lat[i0]:7.2f} "
              f"{dist[i0]:5.1f} km  Ndat {ndat[i0]:4.0f}" + (f"   ** {info[-1]['flag']}" if flag else ''))
    Pe = np.array(Pe) if all(e is not None for e in Pe) else None
    return dict(label=label, path=path, P=np.array(P), Pe=Pe, info=info,
                lon=lon, lat=lat, z=z, A=A, ndat=ndat)


def write_tables(M, sites, zq, field, how, stem):
    head = ','.join(f"{s['no']}_{s['name']}" for s in sites)
    fmt = lambda x: 'NaN' if not np.isfinite(x) else f"{x:.4f}"

    def wide(fn, arr, title):
        with open(fn, 'w') as f:
            f.write(f"# {title}\n")
            f.write('depth_km,' + head + '\n')
            for i, zz in enumerate(zq):
                f.write(f"{zz:g}," + ','.join(fmt(arr[k, i]) for k in range(len(sites))) + '\n')
    L = re.sub(r'[^A-Za-z0-9._-]+', '_', M['label'])       # safe in file names
    wide(f"{stem}.{L}.csv", M['P'], f"{L} {field} (km/s) at study sites, {how}; model {os.path.basename(M['path'])}")
    if M['Pe'] is not None:
        wide(f"{stem}.{L}.err.csv", M['Pe'], f"{L} posterior std of {field} (km/s) at study sites")
    with open(f"{stem}.{L}.nodes.txt", 'w') as f:
        f.write(f"# no name terrane site_lon site_lat node_lon node_lat dist_km Ndat AR23_moho_km flag  ({how})\n")
        for s, q in zip(sites, M['info']):
            f.write(f"{s['no']:2d} {s['name']:<10s} {s['terrane'].replace(' ', '_'):<14s} {s['lon']:8.3f} "
                    f"{s['lat']:8.3f} {q['lon']:8.3f} {q['lat']:8.3f} {q['dist']:6.1f} {q['ndat']:5.0f} "
                    f"{q['moho']:6.1f} {q['flag'].replace(' ', '_') or '-'}\n")
    print(f"Tables: {stem}.{L}.csv" + (', .err.csv' if M['Pe'] is not None else '') + ', .nodes.txt')
    return wide


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs', default=os.environ.get('FVS2'), help='Fvs npz[:LABEL] (default $FVS2)')
    ap.add_argument('--fvs2', default=None, help='second model drawn on top, e.g. $FVS10 (no Moho)')
    ap.add_argument('--label', default=None, help='label of --fvs (default from the file name)')
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
    ap.add_argument('--panel-h', type=float, default=3.0, help='panel height (in); width is 2.15 in')
    ap.add_argument('--slide', action='store_true',
                    help='landscape 3 x 7 layout for a 16:9 slide (= --ncol 7 --panel-h 2.75), '
                         'axis labels on every panel (as plot_site_shallow.py --slide)')
    ap.add_argument('--ref-domain', default=None, metavar='DOMAIN',
                    help='also plot each site as d(field) %% from this domain mean (e.g. yilgarn): '
                         'mean over the model nodes in the domain with Ndat >= --min-ndat')
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--skip-flagged', action='store_true',
                    help='leave out sites whose nearest node is beyond --max-km or has Ndat < --min-ndat')
    ap.add_argument('--dlim', type=float, default=None, help='+- x limit (%%) of the --ref-domain figure')
    ap.add_argument('--out', default='figures/sites/site_vsv.png')
    a = ap.parse_args()
    if a.slide:
        a.ncol, a.panel_h = 7, 2.75
    if not a.fvs:
        sys.exit('give --fvs (or source env_stage6.sh for $FVS2)')

    zq = np.arange(0, a.z_max + 1e-6, a.dz)
    sites = t2.load_sites(a.sites)
    moho = None
    if a.moho and os.path.isfile(a.moho):
        from plot_domain_profiles import moho_at
        moho = moho_at(a.moho, [s['lon'] for s in sites], [s['lat'] for s in sites])
    p1, l1 = model_label(a.fvs)
    M1 = sample(p1, a.label or l1, a.field, sites, zq, a, moho)
    M2 = sample(*model_label(a.fvs2), a.field, sites, zq, a, moho) if a.fvs2 else None
    if M2 is not None and M2['label'] == M1['label']:
        M2['label'] += 'b'
    if a.skip_flagged:                                 # only sites with a good node in every model
        keep = [k for k in range(len(sites))
                if not M1['info'][k]['flag'] and not (M2 and M2['info'][k]['flag'])]
        drop = [sites[k]['name'] for k in range(len(sites)) if k not in keep]
        if drop:
            print(f"  --skip-flagged: leaving out {', '.join(drop)}")
        sites = [sites[k] for k in keep]
        for M in [M1] + ([M2] if M2 else []):
            M['P'] = M['P'][keep]
            M['Pe'] = M['Pe'][keep] if M['Pe'] is not None else None
            M['info'] = [M['info'][k] for k in keep]
    allP = np.concatenate([M1['P'].ravel()] + ([M2['P'].ravel()] if M2 else []))
    xl = a.xlim or ({'xi': (0.9, 1.1), 'xi2': (0.88, 1.12)}.get(a.field) or
                    (np.floor(np.nanmin(allP) * 5) / 5, np.ceil(np.nanmax(allP) * 5) / 5))

    ns = len(sites); nc = a.ncol; nr = int(np.ceil(ns / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.15 * nc + 0.4, a.panel_h * nr + 0.5), sharex=not a.slide, sharey=not a.slide,
                            squeeze=False)
    for k, s in enumerate(sites):
        ax = axs.flat[k]
        col = t2.terrane_colour(s['terrane'], k)
        P, Pe, info = M1['P'], M1['Pe'], M1['info']
        if M2 is not None and M2['Pe'] is not None:
            ax.fill_betweenx(zq, M2['P'][k] - M2['Pe'][k], M2['P'][k] + M2['Pe'][k], color='0.5',
                             alpha=0.15, lw=0)
        if Pe is not None:
            ax.fill_betweenx(zq, P[k] - Pe[k], P[k] + Pe[k], color=col, alpha=0.18, lw=0)
        ax.plot(P[k], zq, color=col, lw=1.5)
        if M2 is not None:
            ax.plot(M2['P'][k], zq, color='k', lw=1.0, ls='--')
        if np.isfinite(info[k]['moho']):
            ax.plot([0.82, 1.0], [info[k]['moho']] * 2, color='k', lw=2.2,
                    transform=ax.get_yaxis_transform(), solid_capstyle='butt')
        ax.set_title(f"{s['no']} {s['name']}", fontsize=8.5, loc='left', pad=2, color=col, fontweight='bold')
        fl = [info[k]['flag']] + ([f"{M2['label']}: {M2['info'][k]['flag']}"] if M2 and M2['info'][k]['flag']
                                   and M2['info'][k]['flag'] != info[k]['flag'] else [])
        flags = '; '.join(x for x in fl if x)
        ax.text(0.04, 0.03, f"{s['terrane']}\nnode {info[k]['dist']:.0f} km"
                + (f" / {M2['info'][k]['dist']:.0f} km" if M2 and abs(M2['info'][k]['dist'] - info[k]['dist']) > 0.5
                   else '') + (f"\n{flags}" if flags else ''),
                transform=ax.transAxes, fontsize=6.3, va='bottom', color='#b03a2e' if flags else '0.35')
        if a.field in ('xi', 'xi2'):
            ax.axvline(1.0, color='0.4', lw=0.7, ls='--')
        ax.grid(True, lw=0.3, color='0.85'); ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[ns:]:
        ax.axis('off')
    if ns < nr * nc:                                  # legend in the first empty slot
        lax = list(axs.flat)[ns]
        h = [Line2D([], [], color='0.3', lw=1.5, label=f"{M1['label']} (terrane colour" + (", ±1σ band)" if M1['Pe'] is not None else ")"))]
        if M2 is not None:
            h.append(Line2D([], [], color='k', lw=1.0, ls='--', label=f"{M2['label']} (grey band)"))
        if moho is not None:
            h.append(Line2D([], [], color='k', lw=2.2, label='AR23 Moho'))
        lax.legend(handles=h, loc='upper left', fontsize=7.5, frameon=False)
    ax0 = axs.flat[0]
    for ax in list(axs.flat)[:ns]:                    # (shared unless --slide)
        ax.set_ylim(a.z_max, 0); ax.set_xlim(*xl)
        if a.slide:
            ax.set_xlabel(LABEL[a.field], fontsize=7.5)
    for ax in axs[:, 0]:
        ax.set_ylabel('depth (km)', fontsize=8)
    for c in range(nc):
        col_axes = [axs[r, c] for r in range(nr) if r * nc + c < ns]
        if col_axes:
            col_axes[-1].xaxis.set_tick_params(labelbottom=True)
            col_axes[-1].set_xlabel(LABEL[a.field], fontsize=8)
    how = (f'inverse-distance mean of {a.idw} nearest nodes' if a.idw > 1 else 'nearest model node')
    fld = LABEL[a.field].split(' (')[0]
    fig.suptitle(f"{M1['label']}" + (f" (solid) vs {M2['label']} (dashed)" if M2 else '')
                 + f" {fld} at the study sites ({how}" + ("; band: ±1 posterior std" if M1['Pe'] is not None else '')
                 + ('; black tick: AR23 Moho' if moho is not None else '') + ')', fontsize=10.5)
    fig.tight_layout(rect=(0, 0.012, 1, 0.97))
    t2.source_note(fig, [M1['path'], M2['path'] if M2 else None])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=220); fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")

    stem = os.path.splitext(a.out)[0]
    models = [M1] + ([M2] if M2 else [])
    fig, axs = plt.subplots(1, len(models), figsize=(5.0 * len(models) + 0.6, 6.4), sharey=True, squeeze=False)
    terr = []
    for j, M in enumerate(models):
        ax = axs[0, j]
        for k, s in enumerate(sites):
            ax.plot(M['P'][k], zq, color=t2.terrane_colour(s['terrane'], k), lw=1.1, alpha=0.9)
            if s['terrane'] not in terr:
                terr.append(s['terrane'])
        ax.set_ylim(a.z_max, 0); ax.set_xlim(*xl)
        ax.set_xlabel(LABEL[a.field]); ax.grid(True, lw=0.3, color='0.85')
        ax.set_title(f"{M['label']}: {ns} study sites", fontsize=10)
    axs[0, 0].set_ylabel('depth (km)')
    axs[0, 0].legend(handles=[Line2D([], [], color=t2.terrane_colour(t, i), lw=2, label=t)
                              for i, t in enumerate(terr)],
                     fontsize=7.5, loc='lower left', frameon=True, framealpha=0.9, title='terrane',
                     title_fontsize=8)
    fig.tight_layout()
    fig.savefig(stem + '.overlay.png', dpi=220); fig.savefig(stem + '.overlay.pdf')
    plt.close(fig)
    print(f"Saved {stem}.overlay.png (+ .pdf)")

    wide = None
    for M in models:
        wide = write_tables(M, sites, zq, a.field, how, stem)
    if M2 is not None:
        wide(f"{stem}.diff.csv", M1['P'] - M2['P'],
             f"{M1['label']} minus {M2['label']}, {a.field} (km/s) at study sites, {how}")
        print(f"       {stem}.diff.csv")
    if a.ref_domain:
        ref_figures(a, sites, zq, models, stem, how, wide)


def ref_figures(a, sites, zq, models, stem, how, wide):
    """Sites as d(field) % from the domain mean of each model; band = +-1 sigma lateral spread."""
    from domains import Domains
    sys.path.insert(0, os.path.join(_HERE, '..', 'extract'))
    from export_domains import all_definitions
    D = Domains(a.tect_npz)
    D.defs = all_definitions(True)
    key = a.ref_domain
    lab = D[key]['label']
    tag = f"dv-{key}"
    for M in models:
        m = D.contains(key, M['lon'], M['lat']) & (M['ndat'] >= a.min_ndat)
        if not m.any():
            sys.exit(f"no {M['label']} nodes in '{key}' with Ndat >= {a.min_ndat}")
        mean = np.array([np.interp(zq, M['z'], col, right=np.nan) for col in [np.nanmean(M['A'][m], 0)]])[0]
        sd = np.interp(zq, M['z'], np.nanstd(M['A'][m], 0), right=np.nan)
        M['ref'] = dict(mean=mean, std=sd, n=int(m.sum()))
        M['dv'] = 100.0 * (M['P'] - mean) / mean
        M['band'] = 100.0 * sd / mean
        print(f"{M['label']}: {lab} mean over {m.sum()} nodes (Ndat >= {a.min_ndat})")
        with open(f"{stem}.{M['label']}.{key}-mean.csv", 'w') as f:
            f.write(f"# {M['label']} {a.field}: {lab} mean and lateral std over {m.sum()} nodes (Ndat >= {a.min_ndat})\n")
            f.write('depth_km,mean_kms,std_kms\n')
            for i, zz in enumerate(zq):
                f.write(f"{zz:g},{mean[i]:.4f},{sd[i]:.4f}\n")
        wide(f"{stem}.{M['label']}.{tag}.csv", M['dv'],
             f"{M['label']} d{a.field} (%) of the study sites from the {lab} mean ({m.sum()} nodes), {how}")
    M1, M2 = models[0], (models[1] if len(models) > 1 else None)
    allv = np.concatenate([M['dv'].ravel() for M in models])
    L = a.dlim or max(2.0, np.ceil(np.nanpercentile(np.abs(allv), 99)))
    fld = LABEL[a.field].split(' (')[0]
    ns = len(sites); nc = a.ncol; nr = int(np.ceil(ns / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.15 * nc + 0.4, a.panel_h * nr + 0.5), sharex=not a.slide, sharey=not a.slide,
                            squeeze=False)
    for k, s in enumerate(sites):
        ax = axs.flat[k]
        col = t2.terrane_colour(s['terrane'], k)
        ax.fill_betweenx(zq, -M1['band'], M1['band'], color='0.6', alpha=0.25, lw=0)
        ax.axvline(0, color='0.3', lw=0.7)
        ax.plot(M1['dv'][k], zq, color=col, lw=1.5)
        if M2 is not None:
            ax.plot(M2['dv'][k], zq, color='k', lw=1.0, ls='--')
        mo = M1['info'][k]['moho']
        if np.isfinite(mo):
            ax.plot([0.82, 1.0], [mo, mo], color='k', lw=2.2, transform=ax.get_yaxis_transform(),
                    solid_capstyle='butt')
        ax.set_title(f"{s['no']} {s['name']}", fontsize=8.5, loc='left', pad=2, color=col, fontweight='bold')
        fl = M1['info'][k]['flag']
        ax.text(0.04, 0.03, s['terrane'] + (f"\n{fl}" if fl else ''), transform=ax.transAxes, fontsize=6.3,
                va='bottom', color='#b03a2e' if fl else '0.35')
        ax.grid(True, lw=0.3, color='0.85'); ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[ns:]:
        ax.axis('off')
    if ns < nr * nc:
        lax = list(axs.flat)[ns]
        h = [Line2D([], [], color='0.3', lw=1.5, label=f"{M1['label']} (terrane colour)")]
        if M2 is not None:
            h.append(Line2D([], [], color='k', lw=1.0, ls='--', label=f"{M2['label']} (vs its own {key} mean)"))
        from matplotlib.patches import Patch
        h.append(Patch(color='0.6', alpha=0.35, label=f"±1σ spread of {lab}\nnodes ({M1['label']})"))
        if np.isfinite(M1['info'][0]['moho']):
            h.append(Line2D([], [], color='k', lw=2.2, label='AR23 Moho'))
        lax.legend(handles=h, loc='upper left', fontsize=7.2, frameon=False)
    for ax in list(axs.flat)[:ns]:
        ax.set_ylim(a.z_max, 0); ax.set_xlim(-L, L)
        if a.slide:
            ax.set_xlabel(f"d{fld} (%) from {key} mean", fontsize=7.5)
    for ax in axs[:, 0]:
        ax.set_ylabel('depth (km)', fontsize=8)
    for c in range(nc):
        col_axes = [axs[r, c] for r in range(nr) if r * nc + c < ns]
        if col_axes:
            col_axes[-1].xaxis.set_tick_params(labelbottom=True)
            col_axes[-1].set_xlabel(f"d{fld} (%) from {key} mean", fontsize=8)
    fig.suptitle(f"Study sites relative to the {lab} mean {fld} ({M1['label']}: {M1['ref']['n']} nodes"
                 + (f"; {M2['label']}: {M2['ref']['n']} nodes" if M2 else '') + f", Ndat >= {a.min_ndat}; {how})",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fn = f"{stem}.{tag}.png"
    fig.savefig(fn, dpi=220); fig.savefig(fn[:-4] + '.pdf')
    plt.close(fig)
    print(f"Saved {fn} (+ .pdf)")

    # overlay: one panel per model, all sites by terrane
    fig, axs = plt.subplots(1, len(models), figsize=(4.6 * len(models) + 0.6, 6.4), sharey=True, squeeze=False)
    terr = []
    for j, M in enumerate(models):
        ax = axs[0, j]
        ax.fill_betweenx(zq, -M['band'], M['band'], color='0.6', alpha=0.25, lw=0)
        ax.axvline(0, color='0.3', lw=0.7)
        for k, s in enumerate(sites):
            ax.plot(M['dv'][k], zq, color=t2.terrane_colour(s['terrane'], k), lw=1.1, alpha=0.9)
            if s['terrane'] not in terr:
                terr.append(s['terrane'])
        ax.set_ylim(a.z_max, 0); ax.set_xlim(-L, L); ax.grid(True, lw=0.3, color='0.85')
        ax.set_xlabel(f"d{fld} (%) from {key} mean"); ax.set_title(f"{M['label']} ({M['ref']['n']} {key} nodes)",
                                                                  fontsize=10)
    axs[0, 0].set_ylabel('depth (km)')
    axs[0, 0].legend(handles=[Line2D([], [], color=t2.terrane_colour(t, i), lw=2, label=t) for i, t in enumerate(terr)],
                     fontsize=7.5, loc='lower left', frameon=True, framealpha=0.9, title='terrane', title_fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{stem}.{tag}.overlay.png", dpi=220); fig.savefig(f"{stem}.{tag}.overlay.pdf")
    plt.close(fig)
    print(f"Saved {stem}.{tag}.overlay.png (+ .pdf); tables {stem}.<model>.{tag}.csv, .<model>.{key}-mean.csv")


if __name__ == '__main__':
    main()
