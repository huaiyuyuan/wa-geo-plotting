#!/usr/bin/env python3
"""ccp_profiles_to_mat.py - write the current profiles (from ccp_profiles.txt) back
into the MATLAB `prof` struct .mat, so profiles.latlon.phases.1.2.3.mat carries the
up-to-date coordinates (re-picks, AF, N->S/W->E/NW->SE orientation).

  python3 tools/ccp_profiles_to_mat.py \
      --def     /workspace/CCP/CCP.WA.Array.2027.phase.3/ccp_profiles.txt \
      --mat-dir /workspace/CCP/CCP.WA.Array.2027.phase.3 \
      --out     /workspace/CCP/CCP.WA.Array.2027.phase.3/profiles.latlon.phases.1.2.3.mat

prof is a 1xN struct array with fields:
  name (char), short (char), latlon (Mx2 = [lat lon]), colour (1x3).
name/colour are preserved from the existing --out .mat where `short` matches; new
profiles get name=short and a palette colour. Coordinates are the oriented current
picks (gc -> great circle; file -> the latlon .mat). Orientation matches the toolkit:
we -> W to E, ns -> N to S, senw -> NW to SE.
"""
import argparse
import os
import numpy as np
from scipy.io import loadmat, savemat

GROUP_COL = {'we': (0.80, 0.20, 0.20), 'ns': (0.20, 0.35, 0.80), 'senw': (0.20, 0.55, 0.25)}


def _unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def great_circle(lat1, lon1, lat2, lon2, n=100):
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


def orient(grp, lat, lon):
    if grp == 'we':
        if lon[0] > lon[-1]:
            lat, lon = lat[::-1], lon[::-1]
    else:
        if lat[0] < lat[-1]:
            lat, lon = lat[::-1], lon[::-1]
    return lat, lon


def existing_meta(out_mat, var='prof'):
    """name/colour per short from the current .mat, if present."""
    meta = {}
    if not os.path.isfile(out_mat):
        return meta
    m = loadmat(out_mat, squeeze_me=True, struct_as_record=False)
    prof = m.get(var)
    if prof is None:
        return meta
    prof = np.atleast_1d(prof)
    for p in prof:
        try:
            short = str(p.short)
            col = np.asarray(p.colour, float).ravel()[:3]
            meta[short] = (str(p.name), col if col.size == 3 else None)
        except Exception:
            pass
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--def', dest='deffile', required=True)
    ap.add_argument('--mat-dir', default=None)
    ap.add_argument('--out', required=True, help='prof struct .mat to write')
    ap.add_argument('--var', default='prof')
    ap.add_argument('--npts', type=int, default=100)
    a = ap.parse_args()
    mat_dir = a.mat_dir or os.path.dirname(os.path.abspath(a.deffile))
    meta = existing_meta(a.out, a.var)

    recs = []
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
            lat, lon = orient(grp, np.asarray(lat, float), np.asarray(lon, float))
            name, col = meta.get(short, (None, None))
            if name is None:
                name = short
            if col is None:
                col = np.array(GROUP_COL.get(grp, (0.3, 0.3, 0.3)), float)
            latlon = np.column_stack([lat, lon])          # Mx2 = [lat lon]
            recs.append((name, short, latlon, np.asarray(col, float).reshape(1, 3)))
            print(f"  {short:8s} {grp:4s}  {len(lat)} pts  name='{name}'")

    n = len(recs)
    if n == 0:
        raise SystemExit(f"no profiles parsed from {a.deffile}")
    dt = np.dtype([('name', 'O'), ('short', 'O'), ('latlon', 'O'), ('colour', 'O')])
    prof = np.zeros((1, n), dtype=dt)
    for i, (name, short, latlon, col) in enumerate(recs):
        prof[0, i]['name'] = name
        prof[0, i]['short'] = short
        prof[0, i]['latlon'] = latlon
        prof[0, i]['colour'] = col
    savemat(a.out, {a.var: prof})
    print(f"wrote {n} profiles -> {a.out}  (var '{a.var}')")


if __name__ == '__main__':
    main()
