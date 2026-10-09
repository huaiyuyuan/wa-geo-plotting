#!/usr/bin/env python3
"""check_section_nodes.py - why does a section look odd?  For each named section, every
~50 km: distance to the nearest model node (deg; > 0.5 is blanked as extrapolation),
that node's Ndat and its Vp/Vs and Vsv range over 0-60 km. Also lists nodes whose
Vp/Vs reaches the model maximum (a prior bound?).

  python3 tests/check_section_nodes.py $FVS11 $XS EW7 EW8
"""
import sys, numpy as np
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'plotting')]
import plot_xsection as px
from scipy.spatial import cKDTree

fvs, xs, names = sys.argv[1], sys.argv[2], sys.argv[3:]
d = np.load(fvs)
lon, lat, z = d['Lon'], d['Lat'], d['z']
zm = z <= 60
P = d['Vpvs'][:, zm] if 'Vpvs' in d.files else None
V = d['Vsv'][:, zm]
N = d['Ndat'] if 'Ndat' in d.files else np.full(len(lon), np.nan)
tree = cKDTree(np.column_stack([lon, lat]))
print(f"{Path(fvs).name}: {len(lon)} nodes, lon {lon.min():.2f}-{lon.max():.2f}, "
      f"lat {lat.min():.2f}-{lat.max():.2f}")
if P is not None:
    pmax = np.nanmax(P)
    hi = np.nanmax(P, axis=1) >= pmax - 0.01
    print(f"Vp/Vs {np.nanmin(P):.3f}-{pmax:.3f}; {hi.sum()} nodes reach >= {pmax - 0.01:.3f} "
          f"(lon {lon[hi].min():.2f}-{lon[hi].max():.2f}, lat {lat[hi].min():.2f}-{lat[hi].max():.2f})"
          if hi.any() else "")
for s in px._load_sections(xs):
    if names and s[4] not in names:
        continue
    plat, plon, dist = px._great_circle_path(*s[:4])
    dd, ii = tree.query(np.column_stack([plon, plat]))
    print(f"\n{s[4]}  ({s[0]:.2f},{s[1]:.2f}) -> ({s[2]:.2f},{s[3]:.2f}), {dist[-1]:.0f} km")
    print("   km    lon     lat   node_dist  Ndat   Vp/Vs(0-60)    Vsv(0-60)")
    for k in range(0, len(dist), max(1, int(round(50 / np.median(np.diff(dist)))))):
        j = ii[k]
        pv = f"{np.nanmin(P[j]):.3f}-{np.nanmax(P[j]):.3f}" if P is not None else "-"
        flag = "  <- beyond 0.5 deg: blanked" if dd[k] > 0.5 else ""
        print(f"{dist[k]:5.0f} {plon[k]:7.2f} {plat[k]:7.2f} {dd[k]:8.2f} {N[j]:6.0f}   {pv}  "
              f"{np.nanmin(V[j]):.2f}-{np.nanmax(V[j]):.2f}{flag}")
