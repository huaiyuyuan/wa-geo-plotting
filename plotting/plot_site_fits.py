#!/usr/bin/env python3
"""
plot_site_fits.py - observed dispersion data and TransD synthetics at the study sites, for
two models on the same panels (e.g. v2 iter2 against v3): one panel per site, 3 x 7 slide
layout like plot_site_profiles.py --slide.

  python3 plotting/plot_site_fits.py \\
      --model "v2 iter2" "<v2 run>/IN/Data_SWD.in.*" "<v2 run>/OUT/**/data_bestd.out*" \\
      --model "v3 sm"    "<v3 run>/IN/Data_SWD.in.*" "<v3 run>/OUT/**/data_bestd.out*" \\
      --out $FIG/compare_v3sm_iter2/site_fits.png

--model LABEL DATA_GLOB SYN_GLOB (repeat; quote the globs, ** recurses). Each file path must
carry the node coordinates as in the TransD names '...x.<lon>.y.<lat>...' (e.g.
Data_SWD.in.pt.0123.x.118.25.y.-30.03); the node nearest each site (within --max-km) is used,
its data file and the synthetic file whose path has the same coordinates.
File formats (RJ_MCMC_2D):
  Data_SWD.in   period  cR(Rayleigh phase)  uR(Rayleigh group)  cL  uL   (-1 = no data;
                lines that do not hold 5 numbers are skipped)
  data_bestd.out  cR uR cL uL period   (period LAST, best-fit synthetic)   (--syn-order
                to change: e.g. 'p cR uR cL uL' if your synthetic file has period first)
Style: model 1 data blue, synthetics red; model 2 data green, synthetics magenta; data thick
with dots, synthetics thin; phase solid, group dashed; Love curves when present (lighter).
Each panel lists the RMS (synthetic - data) per model. Writes <out> (+ .pdf) and
<stem>.rms.txt.
"""
import argparse, glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import tomo2d as t2                                           # noqa: E402

XY = re.compile(r'x\.(-?\d+(?:\.\d+)?)\.y\.(-?\d+(?:\.\d+)?)')
COLS = ('cR', 'uR', 'cL', 'uL')
NAME = {'cR': 'R phase', 'uR': 'R group', 'cL': 'L phase', 'uL': 'L group'}
STYLE = [dict(data='#1f4fd1', syn='#d62728'), dict(data='#1a9641', syn='#c51b7d')]
LS = {'cR': '-', 'uR': '--', 'cL': '-', 'uL': '--'}


def gc_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.radians(lon2 - lon1))
    return 6371.0 * np.arccos(np.clip(c, -1, 1))


def index(pattern):
    """{(lon, lat): path} for files whose path holds x.<lon>.y.<lat>."""
    out = {}
    for p in glob.glob(pattern, recursive=True):
        m = XY.findall(p)
        if m:
            lo, la = m[-1]
            out[(round(float(lo), 4), round(float(la), 4))] = p
    return out


def read_table(path, order):
    """-> dict(p=periods, cR=..., uR=..., cL=..., uL=...) with NaN for missing (-1/<=0)."""
    rows = []
    for line in open(path):
        try:
            v = [float(x) for x in line.replace(',', ' ').split()]
        except ValueError:
            continue
        if len(v) >= 5:
            rows.append(v[:5])
    if not rows:
        return None
    a = np.array(rows)
    out = {k: a[:, i] for i, k in enumerate(order)}
    for k in COLS:
        out[k] = np.where(out[k] > 0, out[k], np.nan)
    o = np.argsort(out['p'])
    return {k: v[o] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', nargs=3, action='append', required=True, metavar=('LABEL', 'DATA_GLOB', 'SYN_GLOB'))
    ap.add_argument('--sites', default=os.path.join(_HERE, '..', 'examples', 'yilgarn_sites_19.txt'))
    ap.add_argument('--max-km', type=float, default=50.0)
    ap.add_argument('--data-order', default='p cR uR cL uL')
    ap.add_argument('--syn-order', default='cR uR cL uL p')
    ap.add_argument('--no-love', action='store_true')
    ap.add_argument('--ylim', nargs=2, type=float, default=None)
    ap.add_argument('--ncol', type=int, default=7)
    ap.add_argument('--panel-h', type=float, default=2.75)
    ap.add_argument('--out', default='figures/sites/site_fits.png')
    a = ap.parse_args()
    dord, sord = a.data_order.split(), a.syn_order.split()

    sites = t2.load_sites(a.sites)
    models = []
    for label, dglob, sglob in a.model[:2]:
        D, S = index(dglob), index(sglob)
        print(f"{label}: {len(D)} data files ({dglob}), {len(S)} synthetic files ({sglob})")
        if not D:
            sys.exit(f"  no data file with 'x.<lon>.y.<lat>' in its path matches {dglob}")
        keys = np.array(list(D))
        per = []
        for s in sites:
            d = gc_km(s['lon'], s['lat'], keys[:, 0], keys[:, 1])
            i = int(np.argmin(d))
            k = tuple(keys[i])
            rec = dict(dist=d[i], key=k, data=None, syn=None)
            if d[i] <= a.max_km:
                rec['data'] = read_table(D[k], dord)
                sp = S.get(k)
                if sp is None:                         # tolerate rounding in the names
                    sk = np.array(list(S)) if S else np.zeros((0, 2))
                    if len(sk):
                        j = int(np.argmin(np.hypot(sk[:, 0] - k[0], sk[:, 1] - k[1])))
                        if np.hypot(*(sk[j] - np.array(k))) < 0.01:
                            sp = S[tuple(sk[j])]
                rec['syn'] = read_table(sp, sord) if sp else None
            per.append(rec)
            print(f"  {s['no']:2d} {s['name']:<10s} node {k[0]:.2f} {k[1]:.2f}  {d[i]:5.1f} km"
                  + ('' if rec['data'] is not None else '  ** no data (too far)')
                  + ('' if rec['syn'] is not None or rec['data'] is None else '  ** no synthetic file'))
        models.append(dict(label=label, per=per))

    comps = [c for c in COLS if not (a.no_love and c in ('cL', 'uL'))]
    allv = [r[t][c] for m in models for r in m['per'] for t in ('data', 'syn') if r[t] is not None for c in comps]
    allv = np.concatenate([v[np.isfinite(v)] for v in allv]) if allv else np.array([3, 4.5])
    yl = a.ylim or (np.floor(np.nanmin(allv) * 5) / 5, np.ceil(np.nanmax(allv) * 5) / 5)
    pall = np.concatenate([r['data']['p'] for m in models for r in m['per'] if r['data'] is not None])
    xl = (pall.min() * 0.9, pall.max() * 1.1)

    ns = len(sites); nc = a.ncol; nr = int(np.ceil(ns / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.15 * nc + 0.4, a.panel_h * nr + 0.5), squeeze=False)
    rms_rows = []
    for k, s in enumerate(sites):
        ax = axs.flat[k]
        col = t2.terrane_colour(s['terrane'], k)
        txt = []
        rms_row = [s['no'], s['name']]
        for mi, m in enumerate(models):
            r = m['per'][k]
            st = STYLE[mi % 2]
            if r['data'] is None:
                txt.append(f"{m['label']}: no node")
                rms_row.append(np.nan)
                continue
            dd, sy = r['data'], r['syn']
            res = []
            for c in comps:
                if not np.isfinite(dd[c]).any():
                    continue
                al = 1.0 if c in ('cR', 'uR') else 0.5
                ax.plot(dd['p'], dd[c], LS[c], color=st['data'], lw=2.2, alpha=al, marker='o', ms=2.2)
                if sy is not None and np.isfinite(sy[c]).any():
                    ax.plot(sy['p'], sy[c], LS[c], color=st['syn'], lw=1.0, alpha=al)
                    sv = np.interp(dd['p'], sy['p'], sy[c], left=np.nan, right=np.nan)
                    ok = np.isfinite(sv) & np.isfinite(dd[c])
                    res.append(sv[ok] - dd[c][ok])
            rms = np.sqrt(np.mean(np.concatenate(res) ** 2)) if res and sum(len(x) for x in res) else np.nan
            rms_row.append(rms)
            txt.append(f"{m['label']}: rms {rms:.3f}" if np.isfinite(rms) else f"{m['label']}: no syn")
        rms_rows.append(rms_row)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(mticker.FixedLocator([2, 3, 5, 10, 20, 30, 50, 80, 120]))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.set_title(f"{s['no']} {s['name']}", fontsize=8.5, loc='left', pad=2, color=col, fontweight='bold')
        ax.text(0.97, 0.03, '\n'.join(txt), transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3,
                color='0.3', bbox=dict(fc='w', ec='none', alpha=0.8, pad=1))
        ax.grid(True, lw=0.3, color='0.85'); ax.tick_params(labelsize=7)
        ax.set_xlabel('period (s)', fontsize=7.5)
        if k % nc == 0:
            ax.set_ylabel('velocity (km/s)', fontsize=8)
    for ax in list(axs.flat)[ns:]:
        ax.axis('off')
    if ns < nr * nc:
        lax = list(axs.flat)[ns]
        h = []
        for mi, m in enumerate(models):
            st = STYLE[mi % 2]
            h += [Line2D([], [], color=st['data'], lw=2.2, marker='o', ms=3, label=f"{m['label']} data"),
                  Line2D([], [], color=st['syn'], lw=1.0, label=f"{m['label']} synthetic")]
        h += [Line2D([], [], color='0.3', ls='-', lw=1.2, label='phase'),
              Line2D([], [], color='0.3', ls='--', lw=1.2, label='group')]
        lax.legend(handles=h, loc='upper left', fontsize=7.2, frameon=False)
    fig.suptitle('Dispersion data and TransD best-fit synthetics at the study sites (nearest node of each model)',
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=220); fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")
    stem = os.path.splitext(a.out)[0]
    with open(stem + '.rms.txt', 'w') as f:
        f.write('# RMS (synthetic - data) over all components present, km/s\n# no name '
                + ' '.join(m['label'].replace(' ', '_') for m in models) + '\n')
        for r in rms_rows:
            f.write(f"{r[0]:2d} {r[1]:<10s} " + ' '.join(f"{x:.4f}" if np.isfinite(x) else 'nan' for x in r[2:]) + '\n')
    print(f"Table: {stem}.rms.txt")


if __name__ == '__main__':
    main()
