#!/usr/bin/env python3
"""
plot_site_shallow.py - shallow (top few hundred metres) Vs profiles at the study sites,
laid out like plot_site_profiles.py: one panel per site in the terrane colour, plus an
overlay of all sites and tables.

Input: a folder of '<no>_<name>_median_<vs|vpvs>_station_<STATION>.dat' files (columns
lon lat depth_m value; Vs in m/s, depth in m, any depth step), matched to the sites of
--sites by <no>. A site without a file gets an empty 'no profile' panel; a Vp/Vs file is
drawn on its own (top) axis and left out of the Vs tables.

  python3 plotting/plot_site_shallow.py --dir station_files --sites examples/yilgarn_sites_19.txt \\
      --out $FIG/sites/site_shallow_vs.png

Per panel: profile, station name, station-site distance (flagged beyond --max-km, 50),
Vs30 (time-averaged: 30 / sum(h / Vs) over the top 30 m).
Outputs: <out> (+ .pdf), <stem>.overlay.png (all sites to --overlay-zmax m),
<stem>.csv (depth every --dz m x site, m/s), <stem>.summary.txt (station, distance,
depth range, Vs at 0 / 30 / 100 m, Vs30, max Vs).
"""
import argparse, glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import tomo2d as t2                                           # noqa: E402

PAT = re.compile(r'^(\d+)_(.+?)_median_(vs|vpvs)_station_([^.]+)\.dat$', re.I)


def gc_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.radians(lon2 - lon1))
    return 6371.0 * np.arccos(np.clip(c, -1, 1))


def vs30(z, v):
    """Time-averaged Vs over the top 30 m (None if the profile is shallower)."""
    if z.max() < 30:
        return np.nan
    zz = np.linspace(0, 30, 3001)
    vv = np.interp(zz, z, v)
    return 30.0 / np.trapezoid(1.0 / vv, zz)


def load_dir(d):
    """Profiles from a folder (searched recursively) or straight from a .zip of the .dat files."""
    import zipfile, tempfile
    if os.path.isfile(d) and zipfile.is_zipfile(d):
        tmp = tempfile.mkdtemp(prefix='shallow_')
        with zipfile.ZipFile(d) as z:
            z.extractall(tmp)
        d = tmp
    if not os.path.isdir(d):
        sys.exit(f"--dir {d}: no such folder or zip file")
    files = sorted(glob.glob(os.path.join(d, '**', '*.dat'), recursive=True))
    if not files:
        sys.exit(f"no .dat files under {d} (give the folder holding "
                 f"<no>_<name>_median_vs_station_<sta>.dat, or the .zip itself)")
    out = {}
    for f in files:
        m = PAT.match(os.path.basename(f))
        if not m:
            print(f"  (skipped {os.path.basename(f)}: name not <no>_<name>_median_<vs|vpvs>_station_<sta>.dat)")
            continue
        a = np.loadtxt(f, ndmin=2)
        o = np.argsort(a[:, 2])
        out[int(m.group(1))] = dict(file=os.path.basename(f), name=m.group(2), kind=m.group(3).lower(),
                                    sta=m.group(4), lon=a[0, 0], lat=a[0, 1], z=a[o, 2], v=a[o, 3])
    print(f"{len(out)} profiles from {len(files)} .dat files under {d}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True,
                    help='folder (searched recursively) or .zip with the <no>_<name>_median_vs_station_<sta>.dat files')
    ap.add_argument('--sites', default=os.path.join(_HERE, '..', 'examples', 'yilgarn_sites_19.txt'))
    ap.add_argument('--max-km', type=float, default=50.0, help='flag stations further than this from the site')
    ap.add_argument('--zmax', type=float, default=None, help='common depth limit (m); default each panel its own')
    ap.add_argument('--xlim', nargs=2, type=float, default=None, help='Vs limits (m/s)')
    ap.add_argument('--overlay-zmax', type=float, default=400.0)
    ap.add_argument('--dz', type=float, default=1.0, help='depth step of the csv table (m)')
    ap.add_argument('--ncol', type=int, default=5)
    ap.add_argument('--panel-h', type=float, default=3.0, help='panel height (in); width is 2.15 in')
    ap.add_argument('--slide', action='store_true',
                    help='landscape 3 x 7 layout for a 16:9 slide (= --ncol 7 --panel-h 2.75)')
    ap.add_argument('--out', default='figures/sites/site_shallow_vs.png')
    a = ap.parse_args()
    if a.slide:
        a.ncol, a.panel_h = 7, 2.75

    sites = t2.load_sites(a.sites)
    P = load_dir(a.dir)
    rows = []
    for s in sites:
        p = P.get(s['no'])
        if p is None:
            print(f"  {s['no']:2d} {s['name']:<10s} no file")
            continue
        p['dist'] = gc_km(s['lon'], s['lat'], p['lon'], p['lat'])
        flags = []
        if p['dist'] > a.max_km:
            flags.append(f"station {p['dist']:.0f} km from site")
        if p['kind'] != 'vs':
            flags.append('file is Vp/Vs, not Vs')
        if p['name'].lower() != s['name'].lower():
            flags.append(f"file name '{p['name']}'")
        p['flag'] = '; '.join(flags)
        p['vs30'] = vs30(p['z'], p['v']) if p['kind'] == 'vs' else np.nan
        print(f"  {s['no']:2d} {s['name']:<10s} {p['sta']:<6s} {p['dist']:6.1f} km  0-{p['z'].max():g} m  "
              + (f"Vs30 {p['vs30']:.0f} m/s" if np.isfinite(p['vs30']) else '') + (f"  ** {p['flag']}" if flags else ''))
    vsv = [p for p in P.values() if p['kind'] == 'vs']
    if not vsv:
        sys.exit('no Vs profiles matched the sites (check the <no>_ prefix of the file names)')
    xl = a.xlim or (0, 500 * np.ceil(max(p['v'].max() for p in vsv) / 500))

    ns = len(sites); nc = a.ncol; nr = int(np.ceil(ns / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.15 * nc + 0.4, a.panel_h * nr + 0.5), squeeze=False)
    for k, s in enumerate(sites):
        ax = axs.flat[k]
        col = t2.terrane_colour(s['terrane'], k)
        ax.set_title(f"{s['no']} {s['name']}", fontsize=8.5, loc='left', pad=2, color=col, fontweight='bold')
        ax.grid(True, lw=0.3, color='0.85'); ax.tick_params(labelsize=7)
        p = P.get(s['no'])
        if p is None:
            ax.set_xlim(*xl); ax.set_ylim(a.zmax or 300, 0)
            ax.text(0.5, 0.5, 'no profile', transform=ax.transAxes, ha='center', va='center',
                    fontsize=8, color='0.5')
            ax.text(0.04, 0.03, s['terrane'], transform=ax.transAxes, fontsize=6.3, va='bottom', color='0.35')
            continue
        zt = a.zmax or (50 * np.ceil(p['z'].max() / 50))
        if p['kind'] == 'vs':
            ax.plot(p['v'], p['z'], color=col, lw=1.5)
            ax.set_xlim(*xl)
        else:                                          # Vp/Vs: own axis, no Vs scale
            ax.plot(p['v'], p['z'], color=col, lw=1.5, ls='--')
            ax.set_xlabel('Vp/Vs', fontsize=7.5)
            ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
        ax.set_ylim(zt, 0)
        txt = [s['terrane'], f"{p['sta']}, {p['dist']:.0f} km"]
        if np.isfinite(p['vs30']):
            txt.append(f"Vs30 {p['vs30']:.0f} m/s")
        if p['flag']:
            txt.append(p['flag'])
        ax.text(0.96, 0.03, '\n'.join(txt), transform=ax.transAxes, fontsize=6.3, va='bottom', ha='right',
                color='#b03a2e' if p['flag'] else '0.35',
                bbox=dict(fc='w', ec='none', alpha=0.75, pad=1))
    for ax in list(axs.flat)[ns:]:
        ax.axis('off')
    for r in range(nr):
        axs[r, 0].set_ylabel('depth (m)', fontsize=8)
    for k in range(ns):
        ax = axs.flat[k]
        p = P.get(sites[k]['no'])
        if p is None or p['kind'] == 'vs':
            ax.set_xlabel('Vs (m/s)', fontsize=7.5)
    fig.suptitle(f"Shallow Vs at the study sites (median profiles at the nearest station; "
                 f"own depth scale per panel{'' if not a.zmax else ''})" if not a.zmax else
                 f"Shallow Vs at the study sites (median profiles at the nearest station, top {a.zmax:g} m)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=220); fig.savefig(os.path.splitext(a.out)[0] + '.pdf')
    plt.close(fig)
    print(f"Saved {a.out} (+ .pdf)")

    stem = os.path.splitext(a.out)[0]
    fig, ax = plt.subplots(figsize=(5.8, 6.4))
    terr = []
    for k, s in enumerate(sites):
        p = P.get(s['no'])
        if p is None or p['kind'] != 'vs':
            continue
        c = t2.terrane_colour(s['terrane'], k)
        ax.plot(p['v'], p['z'], color=c, lw=1.1, alpha=0.9)
        j = np.searchsorted(p['z'], min(a.overlay_zmax, p['z'].max())) - 1
        ax.annotate(str(s['no']), (p['v'][j], p['z'][j]), fontsize=6.3, color=c, fontweight='bold',
                    xytext=(2, 0), textcoords='offset points', clip_on=True)
        if s['terrane'] not in terr:
            terr.append(s['terrane'])
    ax.set_ylim(a.overlay_zmax, 0); ax.set_xlim(*xl)
    ax.set_xlabel('Vs (m/s)'); ax.set_ylabel('depth (m)'); ax.grid(True, lw=0.3, color='0.85')
    ax.legend(handles=[Line2D([], [], color=t2.terrane_colour(t, i), lw=2, label=t) for i, t in enumerate(terr)],
              fontsize=7.5, loc='lower left', frameon=True, framealpha=0.9, title='terrane', title_fontsize=8)
    ax.set_title(f"Shallow Vs, {len(vsv)} sites (numbers at the profile ends)", fontsize=10)
    fig.tight_layout()
    fig.savefig(stem + '.overlay.png', dpi=220); fig.savefig(stem + '.overlay.pdf')
    plt.close(fig)
    print(f"Saved {stem}.overlay.png (+ .pdf)")

    zmax = max(p['z'].max() for p in vsv)
    zq = np.arange(0, zmax + 1e-6, a.dz)
    cols = [(s, P[s['no']]) for s in sites if s['no'] in P and P[s['no']]['kind'] == 'vs']
    with open(stem + '.csv', 'w') as f:
        f.write(f"# shallow Vs (m/s) at study sites from {os.path.basename(os.path.abspath(a.dir))}; "
                f"NaN below a profile's base\n")
        f.write('depth_m,' + ','.join(f"{s['no']}_{s['name']}_{p['sta']}" for s, p in cols) + '\n')
        for zz in zq:
            f.write(f"{zz:g}," + ','.join('NaN' if zz > p['z'].max() + 1e-6 else f"{np.interp(zz, p['z'], p['v']):.1f}"
                                          for s, p in cols) + '\n')
    with open(stem + '.summary.txt', 'w') as f:
        f.write('# no name terrane station sta_lon sta_lat dist_km zmax_m vs0 vs30m vs100m Vs30 vsmax  flag\n')
        for s in sites:
            p = P.get(s['no'])
            if p is None:
                f.write(f"{s['no']:2d} {s['name']:<10s} {s['terrane'].replace(' ', '_'):<14s} -  no_file\n")
                continue
            at = lambda d: (np.interp(d, p['z'], p['v']) if d <= p['z'].max() else np.nan)
            f.write(f"{s['no']:2d} {s['name']:<10s} {s['terrane'].replace(' ', '_'):<14s} {p['sta']:<6s} "
                    f"{p['lon']:8.3f} {p['lat']:8.3f} {p['dist']:6.1f} {p['z'].max():7.1f} {p['v'][0]:7.1f} "
                    f"{at(30):7.1f} {at(100):7.1f} {p['vs30']:7.1f} {p['v'].max():7.1f}  "
                    f"{p['flag'].replace(' ', '_') or '-'}\n")
    print(f"Tables: {stem}.csv, {stem}.summary.txt")


if __name__ == '__main__':
    main()
