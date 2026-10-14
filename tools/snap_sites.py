#!/usr/bin/env python3
"""
snap_sites.py - move study sites onto their snapped stations: a table to send round, and a
sites file at the station positions that every site script takes (--sites).

  python3 tools/snap_sites.py --sites examples/yilgarn_sites_19.txt \\
      --snap examples/yilgarn_sites_19.snap.txt --stations $ST/stations.phases.1.2.3.txt \\
      --out examples/yilgarn_sites_19.stations

  --snap      site_no site_name station network phase dist_km   ('nan' = no station);
              without --snap each site takes its nearest station (--max-km, default 50)
  --stations  'name lat lon [elev]' (e.g. stations.phases.1.2.3.txt, 958 stations)

Writes
  <out>.table.txt  one row per site: site and station coordinates, network, phase, elevation,
                   distance (recomputed; flagged when it differs from --snap by > 1 km)
  <out>.short.txt  the snap table with two more columns: station Lat, Lon (tab-separated)
  <out>.txt        sites file at the station positions (no name terrane lat lon); a site
                   without a station keeps its own position (marked in the table), so the
                   site plots run unchanged:  --sites examples/yilgarn_sites_19.stations.txt
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'plotting'))
import tomo2d as t2                                           # noqa: E402


def gc_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.radians(lon2 - lon1))
    return 6371.0 * np.arccos(np.clip(c, -1, 1))


def load_stations(path):
    st = {}
    for line in open(path):
        p = line.replace(',', ' ').split()
        if not p or p[0].startswith('#'):
            continue
        try:
            st[p[0].split('.')[-1]] = (float(p[1]), float(p[2]), float(p[3]) if len(p) > 3 else np.nan)
        except (ValueError, IndexError):
            continue
    return st                                                 # code -> (lat, lon, elev)


def load_snap(path):
    out = {}
    for line in open(path, encoding='utf-8'):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        p = s.replace('–', '-').split()
        if not p[0].isdigit():
            continue
        sta = p[2] if len(p) > 2 and p[2].lower() != 'nan' else None
        out[int(p[0])] = dict(sta=sta, net=(p[3] if len(p) > 3 and sta else '-'),
                              phase=(p[4] if len(p) > 4 and sta else '-'),
                              dist=(float(p[5]) if len(p) > 5 and sta else np.nan))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sites', default=os.path.join(_HERE, '..', 'examples', 'yilgarn_sites_19.txt'))
    ap.add_argument('--snap', default=None, help='site_no name station network phase dist_km table')
    ap.add_argument('--stations', required=True, help="'name lat lon [elev]' station file")
    ap.add_argument('--max-km', type=float, default=50.0, help='nearest-station search radius without --snap')
    ap.add_argument('--out', default=os.path.join(_HERE, '..', 'examples', 'yilgarn_sites_19.stations'))
    a = ap.parse_args()

    sites = t2.load_sites(a.sites)
    st = load_stations(a.stations)
    snap = load_snap(a.snap) if a.snap else None
    codes = np.array(list(st)); slat = np.array([st[c][0] for c in codes]); slon = np.array([st[c][1] for c in codes])
    rows, new = [], []
    for s in sites:
        if snap is not None:
            sn = snap.get(s['no'], dict(sta=None, net='-', phase='-', dist=np.nan))
        else:
            d = gc_km(s['lon'], s['lat'], slon, slat)
            i = int(np.argmin(d))
            sn = dict(sta=codes[i] if d[i] <= a.max_km else None, net='-', phase='-', dist=np.nan)
        note = ''
        if sn['sta'] and sn['sta'] in st:
            la, lo, el = st[sn['sta']]
            dist = gc_km(s['lon'], s['lat'], lo, la)
            if np.isfinite(sn['dist']) and abs(dist - sn['dist']) > 1.0:
                note = f"dist {dist:.1f} km in station file vs {sn['dist']:g} in snap table"
            new.append(dict(s, lat=la, lon=lo))
        else:
            la = lo = el = dist = np.nan
            note = 'no station - site position kept' if not sn['sta'] else f"{sn['sta']} not in station file"
            new.append(dict(s))
        rows.append((s, sn, la, lo, el, dist, note))
        print(f"  {s['no']:2d} {s['name']:<10s} {sn['sta'] or '-':<6s} "
              + (f"{dist:6.1f} km" if np.isfinite(dist) else '     -   ') + (f"   ** {note}" if note else ''))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + '.table.txt', 'w') as f:
        f.write(f"# Study sites snapped to stations ({os.path.basename(a.sites)}; stations: "
                f"{os.path.basename(a.stations)})\n")
        f.write('# dist_km = great-circle site-station distance; lat/lon decimal degrees (GDA/WGS84)\n')
        f.write(f"{'no':>2s}  {'site':<10s} {'terrane':<14s} {'site_lat':>9s} {'site_lon':>9s}  "
                f"{'station':<7s} {'net':<3s} {'phase':<5s} {'sta_lat':>9s} {'sta_lon':>9s} {'elev_m':>7s} "
                f"{'dist_km':>7s}  note\n")
        for s, sn, la, lo, el, dist, note in rows:
            fmt = lambda x, w, p: f"{x:{w}.{p}f}" if np.isfinite(x) else f"{'-':>{w}s}"
            f.write(f"{s['no']:>2d}  {s['name']:<10s} {s['terrane']:<14s} {s['lat']:9.4f} {s['lon']:9.4f}  "
                    f"{sn['sta'] or '-':<7s} {sn['net']:<3s} {sn['phase']:<5s} {fmt(la, 9, 4)} {fmt(lo, 9, 4)} "
                    f"{fmt(el, 7, 1)} {fmt(dist, 7, 1)}  {note}\n")
    with open(a.out + '.short.txt', 'w') as f:          # the snap table + station lat/lon
        f.write('#\tName\tStation\tNetwork\tPhase\tDist (km)\tLat\tLon\n')
        for s_, sn, la, lo, el, dist, note in rows:
            nan = 'nan'
            f.write(f"{s_['no']}\t{s_['name']}\t{sn['sta'] or nan}\t{sn['net'] if sn['sta'] else nan}\t"
                    f"{sn['phase'] if sn['sta'] else nan}\t"
                    + (f"{sn['dist']:.1f}" if np.isfinite(sn['dist']) else (f"{dist:.1f}" if np.isfinite(dist) else nan))
                    + '\t' + (f"{la:.4f}\t{lo:.4f}" if np.isfinite(la) else 'nan\tnan') + '\n')
    with open(a.out + '.txt', 'w') as f:
        f.write('#\tName\tTerrane\tLAT\tLONG\t(at the snapped station; sites without one keep their own position)\n')
        for s in new:
            f.write(f"{s['no']}\t{s['name']}\t{s['terrane']}\t{s['lat']:.4f}\t{s['lon']:.4f}\n")
    print(f"Saved {a.out}.table.txt, {a.out}.short.txt (snap table + station lat/lon) and {a.out}.txt")


if __name__ == '__main__':
    main()
