#!/usr/bin/env python3
"""
export_profile_strips.py - precompute the GSWA tectonic-unit strips along each prof
transect and write them to JSON, so the MATLAB CCP landscape can draw the SAME fine
units as the Python sections (plot_xsection / plot_profiles_models).

The point-in-polygon logic (parity + smallest-area) lives in xsection_strips.PolygonIndex;
this just samples it along each profile, collapses to contiguous coloured runs with
runs_from_samples, and dumps them. One JSON the MATLAB reads with jsondecode.

  source env_stage6.sh
  python3 tools/export_profile_strips.py \
      --profiles $FIG/profiles/profiles.latlon.phases.1.2.3.polylines.npz \
      --out $FIG/profiles/profiles.strips.json

Output JSON:
  {"profiles": [
     {"short": "WE1",
      "spans": [{"d0": 0.0, "d1": 37.2, "name": "Southern Carnarvon Basin",
                 "rgb": [0.70, 0.85, 1.0]}, ...]},
     ... ]}
d0/d1 are along-profile distance (km); rgb is 0-1.
"""
import argparse, os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLOT = os.path.join(_HERE, '..', 'plotting')
sys.path.insert(0, _PLOT)
sys.path.insert(0, os.path.join(_HERE, '..'))
import plot_xsection as px
import xsection_strips as xs


def polyline_path(lons, lats, ds_deg):
    lons = np.asarray(lons, float); lats = np.asarray(lats, float)
    if len(lons) < 2:
        raise ValueError("profile has < 2 vertices")
    PL, PO, DS, base = [], [], [], 0.0
    for i in range(len(lons) - 1):
        la, lo, ds = px._great_circle_path(lats[i], lons[i], lats[i + 1], lons[i + 1], ds_deg)
        if i > 0:
            la, lo, ds = la[1:], lo[1:], ds[1:]
        PL.append(la); PO.append(lo); DS.append(ds + base)
        base = DS[-1][-1] if len(DS[-1]) else base
    return np.concatenate(PL), np.concatenate(PO), np.concatenate(DS)


def _hex_to_rgb(c):
    from matplotlib.colors import to_rgb
    try:
        return [round(float(v), 4) for v in to_rgb(c)]
    except Exception:
        return [1.0, 1.0, 1.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', required=True, help='polylines npz (from mat_to_profiles.py)')
    ap.add_argument('--tect-npz', default=None, help='override wa_tectonics.npz')
    ap.add_argument('--class-key', default='names',
                    help="unit class key in the npz (default 'names' = TECTNAME, fine units)")
    ap.add_argument('--colours', default='map2022', help='colour scheme (default map2022)')
    ap.add_argument('--ds', type=float, default=0.03, help='sampling step (deg)')
    ap.add_argument('--out', required=True, help='output JSON path')
    args = ap.parse_args()

    tect = px._resolve_npz('TECTONICS_NPZ', 'wa_tectonics.npz', args.tect_npz)
    if not tect or not os.path.isfile(tect):
        sys.exit(f"tectonics npz not found ({tect}); set config.TECTONICS_NPZ or --tect-npz")
    index = xs.PolygonIndex.from_npz(tect, class_key=args.class_key, colours=args.colours)

    d = np.load(args.profiles, allow_pickle=True)
    out = {'profiles': []}
    for i in range(len(d['short'])):
        short = str(d['short'][i])
        plat, plon, dist = polyline_path(np.asarray(d['lon'][i], float),
                                         np.asarray(d['lat'][i], float), args.ds)
        idx = index.sample(plon, plat)
        runs = xs.runs_from_samples(dist, idx, index)
        spans = [dict(d0=round(float(r.d0), 3), d1=round(float(r.d1), 3),
                      name=r.name, rgb=_hex_to_rgb(r.color)) for r in runs]
        out['profiles'].append(dict(short=short, spans=spans))
        print(f"  {short:8s}: {len(spans)} spans over {dist[-1]:.0f} km")

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=0)
    print(f"Wrote {args.out} ({len(out['profiles'])} profiles)")


if __name__ == '__main__':
    main()
