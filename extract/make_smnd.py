#!/usr/bin/env python3
"""extract/make_smnd.py - GSWA Sm-Nd isotope shapefile -> wa_smnd_tdm2.npz.

Reads the point shapefile WA_isotope_neodymium.shp (+ .dbf) from the GSWA
GEOCHRON_Samarium_Neodymium_GDA2020_SHP package (GDA2020 = lon/lat degrees) and
pulls out, per sample, the two-stage depleted-mantle model age TDM2 (Ma) at its
lon/lat. Pure-Python reader (numpy only), same style as extract/make_litho.py.

  python3 extract/make_smnd.py --shp <...>/WA_isotope_neodymium \
      --out data/geology/wa_smnd_tdm2.npz [--field TDM2_MA]

Output npz: lon, lat, tdm2 (Ma) + sym_tdm2 (GSWA bin label), tect_unit; plus
`field`/`units`/`source` metadata. CC-BY-4.0 (c) Geological Survey of Western Australia.
"""
import argparse, struct, os
import numpy as np


def read_points(path):
    """(N,2) lon/lat from a Point/PointZ shapefile (shapeType 1 or 11)."""
    with open(path, 'rb') as f:
        d = f.read()
    pts, pos, n = [], 100, len(d)
    while pos < n:
        clen = struct.unpack('>I', d[pos + 4:pos + 8])[0]
        rs = pos + 8
        st = struct.unpack('<I', d[rs:rs + 4])[0]
        if st in (1, 11):
            x, y = struct.unpack('<2d', d[rs + 4:rs + 20])
            pts.append((x, y))
        else:
            pts.append((np.nan, np.nan))
        pos = rs + 2 * clen
    return np.asarray(pts, float)


def read_dbf(path):
    """(field names, list of {field: str}) from a .dbf (latin-1, stripped)."""
    with open(path, 'rb') as f:
        d = f.read()
    nrec = struct.unpack('<I', d[4:8])[0]
    hlen = struct.unpack('<H', d[8:10])[0]
    rlen = struct.unpack('<H', d[10:12])[0]
    pos, fields = 32, []
    while d[pos] != 0x0D:
        nm = d[pos:pos + 11].split(b'\x00')[0].decode('latin1')
        fl = d[pos + 16]
        fields.append((nm, fl)); pos += 32
    recs, p = [], hlen
    for _ in range(nrec):
        off, row = p + 1, {}
        for nm, fl in fields:
            row[nm] = d[off:off + fl].decode('latin1').strip(); off += fl
        recs.append(row); p += rlen
    return [f[0] for f in fields], recs


def _num(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def _bin_edges(syms):
    """Official GSWA TDM2 bin edges from the SYM_TDM2 labels ('1250 to 1905', ...).
    Returns the sorted unique edge list, ignoring the 'Non-felsic Igneous Rocks' label."""
    import re
    edges = set()
    for s in syms:
        mm = re.findall(r'-?\d+', str(s))
        if len(mm) == 2:                       # a numeric 'LO to HI' felsic bin
            edges.add(int(mm[0])); edges.add(int(mm[1]))
    return np.array(sorted(edges), float)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shp', required=True, help='shapefile base path (no extension)')
    ap.add_argument('--out', default='data/geology/wa_smnd_tdm2.npz')
    ap.add_argument('--field', default='TDM2_MA', help='model-age field (default TDM2_MA)')
    ap.add_argument('--all-rocks', action='store_true',
                    help='keep every sample; default keeps only felsic igneous rocks '
                         '(SYM_TDM2 is a numeric bin, not "Non-felsic Igneous Rocks"), '
                         'as in the GSWA/Smithies TDM2 model-age map')
    args = ap.parse_args()

    base = args.shp[:-4] if args.shp.lower().endswith('.shp') else args.shp
    pts = read_points(base + '.shp')
    fields, recs = read_dbf(base + '.dbf')
    if args.field not in fields:
        raise SystemExit(f"field '{args.field}' not in {fields}")

    # prefer the geometry lon/lat; fall back to LATITUDE/LONGITUDE columns
    lon, lat = pts[:, 0], pts[:, 1]
    bad = ~np.isfinite(lon) | ~np.isfinite(lat)
    if bad.any() and {'LONGITUDE', 'LATITUDE'} <= set(fields):
        lon = np.where(bad, [_num(r['LONGITUDE']) for r in recs], lon)
        lat = np.where(bad, [_num(r['LATITUDE']) for r in recs], lat)

    tdm2 = np.array([_num(r[args.field]) for r in recs])
    sym = np.array([r.get('SYM_TDM2', '') for r in recs])
    tect = np.array([r.get('TECT_UNIT', '') for r in recs])

    # 'felsic' = SYM_TDM2 is a numeric age bin (not 'Non-felsic Igneous Rocks'); the
    # GSWA/Smithies TDM2 model-age map is felsic-only (non-felsic TDM2 isn't meaningful).
    felsic = np.array([len(__import__('re').findall(r'-?\d+', str(s))) == 2 for s in sym])
    edges = _bin_edges(sym)                                   # official GSWA bin edges

    keep = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(tdm2) & (tdm2 > 0)
    if not args.all_rocks:
        keep &= felsic
    lon, lat, tdm2, sym, tect = lon[keep], lat[keep], tdm2[keep], sym[keep], tect[keep]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out, lon=lon, lat=lat, tdm2=tdm2, sym_tdm2=sym, tect_unit=tect,
             bins=edges, field=args.field, units='Ma',
             rocks=('all' if args.all_rocks else 'felsic'),
             source='GSWA GEOCHRON Sm-Nd (GDA2020); two-stage depleted-mantle model age '
                    'TDM2; classified SYM_TDM2 bins' + ('' if args.all_rocks else ', felsic only'))
    print(f"  {keep.sum()}/{len(recs)} samples kept "
          f"({'all rocks' if args.all_rocks else 'felsic only'}; "
          f"{int(felsic.sum())} felsic of {len(recs)})")
    print(f"  TDM2 range: {tdm2.min():.0f}-{tdm2.max():.0f} Ma (median {np.median(tdm2):.0f})")
    if edges.size:
        print(f"  official bins (Ma): {[int(e) for e in edges]}")
    print(f"  lon {lon.min():.2f}-{lon.max():.2f}  lat {lat.min():.2f}-{lat.max():.2f}")
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
