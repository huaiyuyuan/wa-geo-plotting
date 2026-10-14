#!/usr/bin/env python3
"""
export_domains.py - the tectonic domains of basemap/domains.csv (Yilgarn, West Australian
Craton, Perth Basin, Albany-Fraser, Capricorn, ...) as a stand-alone GeoJSON, so other
projects (e.g. per-domain dispersion curves in noise_asdf) classify points with exactly
the polygons used here - no wa-geo-plotting needed on that side.

Source: GSWA 1:10M tectonic units (WA_Tectonic_Units_10M_2021, GDA2020, CC BY 4.0) via
the wa_tectonics.npz cache (fields TECTNAME -> unit name, PARENTNAME -> parent).

  python3 extract/export_domains.py --out wa_domains.geojson            # domains.csv
  python3 extract/export_domains.py --all --out wa_domains.geojson      # + commented ones
                                                                        #   (pilbara, canning ...)
Output: one Feature per (domain, 10M unit): properties domain, label, colour, unit;
geometry Polygon (exterior anticlockwise, holes clockwise, lon/lat degrees). A point is in
a domain when ANY feature with that domain contains it. Domains overlap on purpose (the
West Australian Craton contains the Yilgarn etc.); classify per domain, not exclusively.
"""
import argparse, json, os, sys
import numpy as np
from matplotlib.path import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'basemap'))
from domains import Domains, load_domains, DOMAINS_CSV, _norm      # noqa: E402


def area(r):
    return 0.5 * (np.dot(r[:, 0], np.roll(r[:, 1], -1)) - np.dot(r[:, 1], np.roll(r[:, 0], -1)))


def polygons(rings):
    """Rings of one unit -> [[exterior, hole, ...], ...]: rings nested an odd number of times
    are holes, each attached to the smallest exterior containing it."""
    paths = [Path(r) for r in rings]
    depth = [sum(paths[j].contains_point(r[0]) for j in range(len(rings)) if j != i)
             for i, r in enumerate(rings)]
    ext = [i for i, dp in enumerate(depth) if dp % 2 == 0]
    polys = {i: [rings[i] if area(rings[i]) > 0 else rings[i][::-1]] for i in ext}
    for i, dp in enumerate(depth):
        if dp % 2 == 0:
            continue
        hosts = [e for e in ext if paths[e].contains_point(rings[i][0])]
        if hosts:
            e = min(hosts, key=lambda e: abs(area(rings[e])))
            polys[e].append(rings[i] if area(rings[i]) < 0 else rings[i][::-1])
    return list(polys.values())


def all_definitions(include_commented):
    defs = load_domains()
    if include_commented:                      # '# key,label,colour,units' lines in the csv
        import csv, io
        for line in open(DOMAINS_CSV, encoding='utf-8'):
            body = line.lstrip('#').strip()
            if line.startswith('#') and body.count(',') >= 3 and ':' in body.split(',', 3)[3]:
                r = next(csv.reader(io.StringIO(body)))
                if r[0] not in defs:
                    units = [(u.split(':', 1)[0].strip().lower(), _norm(u.split(':', 1)[1]))
                             for u in r[3].split(';') if ':' in u]
                    defs[r[0]] = dict(key=r[0], label=r[1], colour=r[2], units=units)
    return defs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='wa_domains.geojson')
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--all', action='store_true', help='also the commented-out domains of domains.csv')
    ap.add_argument('--domains', nargs='+', default=None, help='only these keys')
    a = ap.parse_args()
    D = Domains(a.tect_npz)
    D.defs = all_definitions(a.all)
    keys = a.domains or list(D.defs)
    feats = []
    for k in keys:
        dom = D[k]
        units = D.unit_rings(k)
        nring = 0
        for unit, idx in units.items():
            rings = [D.rings[i] for i in idx if len(D.rings[i]) >= 3]
            for poly in polygons(rings):
                coords = [[[round(float(x), 6), round(float(y), 6)] for x, y in
                           (r if np.allclose(r[0], r[-1]) else np.vstack([r, r[:1]]))] for r in poly]
                feats.append(dict(type='Feature', geometry=dict(type='Polygon', coordinates=coords),
                                  properties=dict(domain=k, label=dom['label'], colour=dom['colour'],
                                                  unit=D.raw_names.get(unit, unit))))
                nring += len(poly)
        print(f"  {k:<14s} {dom['label']:<24s} {len(units):3d} units, {nring:4d} rings")
    fc = dict(type='FeatureCollection', features=feats,
              source='GSWA 1:10M tectonic units of Western Australia (2021), GDA2020, CC BY 4.0; '
                     'domains: wa-geo-plotting basemap/domains.csv',
              usage='point is in domain D if any feature with properties.domain == D contains it; '
                    'domains overlap (WAC contains Yilgarn, Capricorn, Albany-Fraser)')
    with open(a.out, 'w') as f:
        json.dump(fc, f)
    print(f"Saved {a.out}: {len(feats)} features, {os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
