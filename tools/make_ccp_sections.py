#!/usr/bin/env python3
"""make_ccp_sections.py - ONE source for the CCP-landscape profiles.

Reads ccp_profiles.txt (the SAME file the MATLAB plot_ccp_landscape_WA.m reads) and
writes the Python-side profile files:
    <stem>.sections.txt    endpoints: lat1 lon1 lat2 lon2 short
    <stem>.polylines.npz   full paths: short/name/group/colour/order/lat/lon
so the Python plotters, map and tectonic strips all follow the same definitions as
the MATLAB figure. Re-run tools/export_profile_strips.py on the new polylines.npz
afterwards (for the fine units + boundary crossings).

  python3 tools/make_ccp_sections.py \
      --def     /workspace/CCP/CCP.WA.Array.2027.phase.3/ccp_profiles.txt \
      --mat-dir /workspace/CCP/CCP.WA.Array.2027.phase.3 \
      --out     $FIG/profiles/profiles.latlon.phases.1.2.3

ccp_profiles.txt columns:  short  group(senw|ns|we)  type(gc|file)  args
  gc   -> args = lat1 lon1 lat2 lon2   (great circle, 100 pts; START pair then END pair)
  file -> args = latlon .mat (var 'latlon' = [lat lon]); basename resolved vs --mat-dir
Great circles use the same slerp as MATLAB gcwaypts (even spacing in central angle),
so the lines match the MATLAB figure to plotting precision.
"""
import argparse
import os
import numpy as np
from scipy.io import loadmat

GROUP_COL = {'we': (0.80, 0.20, 0.20), 'ns': (0.20, 0.35, 0.80), 'senw': (0.20, 0.55, 0.25)}


def _unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def orient(grp, lat, lon):
    """Flip to the canonical direction so row/click order never matters:
    we -> W to E (start min lon); ns -> N to S, senw -> NW to SE (start max lat)."""
    if grp == 'we':
        if lon[0] > lon[-1]:
            lat, lon = lat[::-1], lon[::-1]
    else:
        if lat[0] < lat[-1]:
            lat, lon = lat[::-1], lon[::-1]
    return lat, lon


def great_circle(lat1, lon1, lat2, lon2, n=100):
    """n points along the great circle, evenly spaced in central angle (= gcwaypts)."""
    p1, p2 = _unit(lat1, lon1), _unit(lat2, lon2)
    omega = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    t = np.linspace(0.0, 1.0, n)
    if omega < 1e-9:
        pts = np.tile(p1, (n, 1))
    else:
        a = np.sin((1 - t) * omega) / np.sin(omega)
        b = np.sin(t * omega) / np.sin(omega)
        pts = a[:, None] * p1 + b[:, None] * p2
    lat = np.degrees(np.arcsin(np.clip(pts[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return lat, lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--def', dest='deffile', required=True, help='ccp_profiles.txt')
    ap.add_argument('--mat-dir', default=None,
                    help='base dir for file-type latlon .mat (default: def-file dir)')
    ap.add_argument('--out', required=True, help='output stem (.sections.txt + .polylines.npz)')
    ap.add_argument('--npts', type=int, default=100)
    a = ap.parse_args()
    mat_dir = a.mat_dir or os.path.dirname(os.path.abspath(a.deffile))

    shorts, groups, lats, lons, cols = [], [], [], [], []
    with open(a.deffile) as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            tok = s.split()
            if len(tok) < 4:
                continue
            short, grp, typ = tok[0], tok[1].lower(), tok[2].lower()
            if typ == 'gc':
                v = list(map(float, tok[3:7]))
                lat, lon = great_circle(v[0], v[1], v[2], v[3], a.npts)
            elif typ == 'file':
                fp = tok[3] if os.path.isfile(tok[3]) else os.path.join(mat_dir, tok[3])
                if not os.path.isfile(fp):
                    raise SystemExit(f"{short}: latlon file not found: {tok[3]}")
                ll = np.asarray(loadmat(fp)['latlon'], float)
                if ll.shape[1] != 2 and ll.shape[0] == 2:
                    ll = ll.T
                lat, lon = ll[:, 0], ll[:, 1]
            else:
                raise SystemExit(f"{short}: unknown type {typ!r}")
            lat, lon = np.asarray(lat, float), np.asarray(lon, float)
            lat, lon = orient(grp, lat, lon)        # canonical: N->S, W->E, NW->SE
            shorts.append(short)
            groups.append(grp)
            lats.append(lat)
            lons.append(lon)
            cols.append(GROUP_COL.get(grp, (0.3, 0.3, 0.3)))
            print(f"  {short:8s} {grp:4s} {typ:4s}  {len(lat)} pts")

    n = len(shorts)
    if n == 0:
        raise SystemExit(f"no profiles parsed from {a.deffile}")

    sec = a.out + '.sections.txt'
    with open(sec, 'w') as f:
        for i in range(n):
            f.write(f"{lats[i][0]:.4f} {lons[i][0]:.4f} "
                    f"{lats[i][-1]:.4f} {lons[i][-1]:.4f} {shorts[i]}\n")

    npz = a.out + '.polylines.npz'
    np.savez(npz,
             short=np.array(shorts, dtype=object),
             name=np.array(shorts, dtype=object),
             group=np.array(groups, dtype=object),
             colour=np.array(cols, float),
             order=np.arange(n),
             lat=np.array(lats, dtype=object),
             lon=np.array(lons, dtype=object))
    print(f"wrote {sec}\n      {npz}  ({n} profiles: {', '.join(shorts)})")
    print("next: python3 tools/export_profile_strips.py --profiles "
          f"{npz} --out <FIG>/profiles/profiles.strips.json")


if __name__ == '__main__':
    main()
