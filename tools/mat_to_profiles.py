#!/usr/bin/env python3
"""
mat_to_profiles.py — turn a MATLAB `prof` struct of picked transects into the
inputs the plotting code reads.

Expected .mat (as in CCP.WA.Array.2027.phase.3/profiles.latlon.phases.1.2.3.mat):

    prof  1xN struct array with fields
        name    char   e.g. 'SE-NW transect 1'
        short   char   e.g. 'SENW1'
        latlon  Mx2 double   the picked polyline vertices
        colour  1x3 double   the line colour used on the map

Writes two files next to --out (default: beside the .mat), stem from --stem:

    <stem>.sections.txt    stock 5-column lines  "lat1 lon1 lat2 lon2 short"
                           (first & last vertex, oriented) — works unchanged with
                           plot_xsection.py --load-sections and the index map.
    <stem>.polylines.npz   full polylines + metadata for plot_profiles_models.py:
                           short[], name[], group[], colour[Nx3], order[],
                           lon (object array of 1-D), lat (object array of 1-D).

Column order in `latlon` ([lat lon] vs [lon lat]) is auto-detected from the WA
value ranges, so it does not matter which way round it was saved.

Group per profile, from `short`:
    WE*            -> 'we'    (west-east transects)
    SENW*, WEST    -> 'senw'  (SE-NW transects, incl. the coastal WEST line)
    everything else-> 'ns'    (116.5E, 117.5E, NS120, ...)

Orientation (canonical "start" vertex, so the left of every panel is the same
geographic end and the stock endpoints come out consistent):
    we   -> start at the WEST end  (min lon)
    ns   -> start at the NORTH end (max lat)
    senw -> start at the NW end    (min lon)

Usage:
    python3 tools/mat_to_profiles.py profiles.latlon.phases.1.2.3.mat
    python3 tools/mat_to_profiles.py in.mat --var prof --stem profiles --out-dir .
"""
import argparse, os, sys
import numpy as np


# WA value ranges used to tell a latitude column from a longitude column.
LAT_RANGE = (-40.0, -15.0)
LON_RANGE = (108.0, 132.0)


def _cellstr(x):
    """A MATLAB char field from loadmat comes back as an ndarray; make it a str."""
    a = np.asarray(x).ravel()
    if a.size == 0:
        return ''
    if a.dtype.kind in ('U', 'S'):
        return str(a[0]) if a.size == 1 else ''.join(str(v) for v in a)
    return str(np.asarray(x).squeeze())


def _latlon_cols(arr):
    """Return (lat, lon) 1-D arrays from an Mx2 block, auto-detecting which
    column is which from the WA ranges. Falls back to [lat, lon] order."""
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or 2 not in arr.shape:
        raise ValueError(f"latlon is not Mx2 (got shape {arr.shape})")
    if arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    c0, c1 = arr[:, 0], arr[:, 1]

    def score(col, rng):
        m = np.nanmedian(col)
        return rng[0] <= m <= rng[1]

    c0_is_lat = score(c0, LAT_RANGE)
    c1_is_lat = score(c1, LAT_RANGE)
    if c0_is_lat and not c1_is_lat:
        lat, lon = c0, c1
    elif c1_is_lat and not c0_is_lat:
        lat, lon = c1, c0
    else:
        # ambiguous: use the more-negative-median column as latitude
        lat, lon = (c0, c1) if np.nanmedian(c0) < np.nanmedian(c1) else (c1, c0)
        print(f"    (lat/lon order ambiguous; assumed lat = the more-negative column)")
    return np.asarray(lat, float), np.asarray(lon, float)


def _group(short):
    s = short.strip().upper()
    if s.startswith('SENW') or s == 'WEST':   # check before the generic 'WE' prefix
        return 'senw'                          # (WEST is the coastal SE-NW line, not a W-E one)
    if s.startswith('WE'):
        return 'we'
    return 'ns'


def _orient(lat, lon, group):
    """Flip the polyline so vertex 0 is the canonical start for its group.
    Convention (vertex 0 = the preferred start): W-E -> west, N-S -> south,
    SE-NW -> south-east end. So distance increases W->E, S->N and SE->NW."""
    if group == 'we':
        start_is_first = lon[0] <= lon[-1]          # start at the west end
    elif group == 'ns':
        start_is_first = lat[0] <= lat[-1]          # start at the south end
    else:                                           # senw: start at the SE (southern) end
        start_is_first = lat[0] <= lat[-1]
    if not start_is_first:
        lat, lon = lat[::-1].copy(), lon[::-1].copy()
    return lat, lon


def load_prof(mat_path, var='prof'):
    from scipy.io import loadmat
    m = loadmat(mat_path, squeeze_me=False, struct_as_record=True)
    if var not in m:
        cands = [k for k in m if not k.startswith('__')]
        sys.exit(f"variable '{var}' not in {mat_path}; found: {cands}")
    S = m[var]
    S = S.ravel(order='F')
    out = []
    for k in range(S.size):
        rec = S[k]
        short = _cellstr(rec['short'])
        name = _cellstr(rec['name'])
        lat, lon = _latlon_cols(rec['latlon'])
        try:
            colour = np.asarray(rec['colour'], float).ravel()[:3]
            if colour.size < 3:
                colour = np.array([0.2, 0.2, 0.2])
        except Exception:
            colour = np.array([0.2, 0.2, 0.2])
        grp = _group(short)
        lat, lon = _orient(lat, lon, grp)
        out.append(dict(short=short, name=name, group=grp,
                        colour=colour, lat=lat, lon=lon))
    return out


# sort key so profiles stack in map order within each group
def _sort_key(p):
    g = p['group']
    la, lo = p['lat'], p['lon']
    if g == 'we':                         # north to south
        return -0.5 * (la[0] + la[-1])
    if g == 'ns':                         # west to east
        return 0.5 * (lo[0] + lo[-1])
    return 0.5 * (lo[0] + lo[-1])         # senw: west to east


def write_outputs(profs, stem, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    profs = sorted(profs, key=_sort_key)
    for i, p in enumerate(profs):
        p['order'] = i

    # 1) stock endpoints txt (first & last vertex) — plot_xsection.py compatible
    sec_path = os.path.join(out_dir, f'{stem}.sections.txt')
    with open(sec_path, 'w') as f:
        f.write("# lat1 lon1 lat2 lon2 label   (from mat_to_profiles.py; "
                "endpoints of each picked polyline)\n")
        for p in profs:
            f.write(f"{p['lat'][0]:.4f} {p['lon'][0]:.4f} "
                    f"{p['lat'][-1]:.4f} {p['lon'][-1]:.4f} {p['short']}\n")

    # 2) full polylines + metadata npz
    npz_path = os.path.join(out_dir, f'{stem}.polylines.npz')
    np.savez(
        npz_path,
        short=np.array([p['short'] for p in profs], dtype=object),
        name=np.array([p['name'] for p in profs], dtype=object),
        group=np.array([p['group'] for p in profs], dtype=object),
        order=np.array([p['order'] for p in profs], dtype=int),
        colour=np.array([p['colour'] for p in profs], dtype=float),
        lon=np.array([p['lon'] for p in profs], dtype=object),
        lat=np.array([p['lat'] for p in profs], dtype=object),
    )

    # console summary
    print(f"Read {len(profs)} profiles:")
    for g in ('we', 'ns', 'senw'):
        gp = [p for p in profs if p['group'] == g]
        if gp:
            names = ', '.join(p['short'] for p in gp)
            print(f"  {g:4s} ({len(gp)}): {names}")
    print(f"Wrote:\n  {sec_path}\n  {npz_path}")
    return sec_path, npz_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mat', help='the .mat file holding the prof struct')
    ap.add_argument('--var', default='prof', help="struct variable name (default prof)")
    ap.add_argument('--stem', default=None,
                    help='output stem (default: the .mat basename without extension)')
    ap.add_argument('--out-dir', default=None,
                    help='where to write (default: beside the .mat)')
    args = ap.parse_args()

    if not os.path.isfile(args.mat):
        sys.exit(f"no such file: {args.mat}")
    stem = args.stem or os.path.splitext(os.path.basename(args.mat))[0]
    out_dir = args.out_dir or (os.path.dirname(os.path.abspath(args.mat)) or '.')

    profs = load_prof(args.mat, args.var)
    write_outputs(profs, stem, out_dir)


if __name__ == '__main__':
    main()
