#!/usr/bin/env python3
"""
make_tectonic_basemap.py — Extract GSWA tectonic polygons from a shapefile into
(a) a GeoJSON (universal: ArcGIS/QGIS/web) and (b) an npz (fast Python cache for
wa_basemap.py). Pure-Python shapefile reader — no geopandas/fiona/pyshp needed.

Usage:
  python3 make_tectonic_basemap.py \
      --shp /workspace/shape.files/WA_Tectonic_Units_10M_2021/<name>.shp \
      --out-npz  /workspace/shape.files/derived/wa_tectonics.npz \
      --out-json /workspace/shape.files/derived/wa_tectonics.geojson \
      --color-field PARENTNAME

Color scheme is keyed on --color-field (default PARENTNAME = major domains).
Source data: GSWA GEOLOGY 10M Tectonics, GDA2020, CC-BY-4.0. Attribute GSWA.
"""
import argparse, struct, json, os
import numpy as np

def _objarr(items):
    """1-D object array of arrays. np.array(list, dtype=object) silently stacks
    equal-length arrays into an N-D object array; this never does."""
    out = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        out[i] = x
    return out


# --- Pure-Python ESRI shapefile + DBF readers ---
def read_shp_polygons(path):
    with open(path,'rb') as f: data=f.read()
    shapes=[]; pos=100; n=len(data)
    while pos < n:
        clen = struct.unpack('>I', data[pos+4:pos+8])[0]
        rstart = pos+8
        shptype = struct.unpack('<I', data[rstart:rstart+4])[0]
        if shptype == 5:  # Polygon
            off = rstart+4+32
            nparts = struct.unpack('<I', data[off:off+4])[0]
            npts   = struct.unpack('<I', data[off+4:off+8])[0]
            off += 8
            parts = struct.unpack(f'<{nparts}I', data[off:off+4*nparts]); off += 4*nparts
            pts = struct.unpack(f'<{2*npts}d', data[off:off+16*npts])
            xy = np.array(pts).reshape(npts,2)
            rings=[xy[parts[k]:(parts[k+1] if k+1<nparts else npts)] for k in range(nparts)]
            shapes.append(rings)
        else:
            shapes.append([])
        pos = rstart + 2*clen
    return shapes

def _dec(b):
    """DBF text: UTF-8 (GSWA shapefiles) with Latin-1 fallback; en/em dashes -> '-'
    so names match the colour tables ('Albany-Fraser Orogen')."""
    try:
        s = b.decode('utf-8')
    except UnicodeDecodeError:
        s = b.decode('latin1')
    return s.strip().replace('\u2013', '-').replace('\u2014', '-')


def read_dbf(path):
    with open(path,'rb') as f: d=f.read()
    nrec=struct.unpack('<I',d[4:8])[0]; hlen=struct.unpack('<H',d[8:10])[0]; rlen=struct.unpack('<H',d[10:12])[0]
    fields=[]; pos=32
    while d[pos]!=0x0D:
        nm=d[pos:pos+11].split(b'\x00')[0].decode('latin1'); fl=d[pos+16]
        fields.append((nm,fl)); pos+=32
    recs=[]; pos=hlen
    for _ in range(nrec):
        off=pos+1; row={}
        for nm,fl in fields:
            row[nm]=_dec(d[off:off+fl]); off+=fl
        recs.append(row); pos+=rlen
    return recs

# Domain color scheme (GSWA-inspired, by PARENTNAME)
PARENT_COLORS = {
    'Yilgarn Craton':'#2ca02c','West Australian Craton':'#98df8a','Pilbara Craton':'#1a9641',
    'North Australian Craton':'#d9ef8b','Capricorn Orogen':'#b07aa1','Albany-Fraser Orogen':'#9467bd',
    'Pinjarra Orogen':'#e377c2','Paterson Orogen':'#c49c94','Halls Creek Orogen':'#8c564b',
    'Granites-Tanami Orogen':'#bcbd22','Kepa Kurl Booya Province':'#aec7e8','Coompana Province':'#c5b0d5',
    'Centralian Superbasin':'#ffdd88','Westralian Superbasin':'#add8e6','Phanerozoic basins':'#add8e6',
    'Neoproterozoic basins':'#f0e68c','Barren Basin':'#f5deb3','STATE':'#dfefb0',
    'Fortescue Basin':'#e6a157',
}
# PARENTNAME groups that are super-units: their rings are grouped/coloured by their
# own unit name instead (WAC children = Yilgarn, Pilbara, Capricorn, Albany-Fraser,
# Fortescue Basin), so e.g. the Pilbara and Fortescue are not one 'craton' colour.
SPLIT_PARENTS = {'West Australian Craton'}
DEFAULT_COLOR='#e8e8e8'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shp', required=True)
    ap.add_argument('--out-npz', default=None)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--color-field', default='PARENTNAME')
    ap.add_argument('--name-field', default='TECTNAME',
                    help='per-ring unit name saved as "names" (terrane strip on sections)')
    ap.add_argument('--colour-field', default='TECTCOLOUR',
                    help='GSWA map colour per unit, saved as "tect_colors" (domain strip colours)')
    args=ap.parse_args()

    dbf = args.shp.replace('.shp','.dbf')
    shapes = read_shp_polygons(args.shp)
    recs   = read_dbf(dbf)
    print(f"{len(shapes)} polygons, {len(recs)} records")
    if len(shapes) != len(recs):
        print("  WARNING: polygon/record count mismatch - names may be misaligned")

    # npz: flat rings + colors
    ring_coords=[]; ring_color=[]; ring_parent=[]; ring_name=[]; ring_tc=[]
    from make_litho import parse_colour          # same parser as the 500k lithology colours
    fmt_count={}; examples={}
    # geojson: FeatureCollection
    features=[]
    for rings, rec in zip(shapes, recs):
        key = rec.get(args.color_field,'')
        if key in SPLIT_PARENTS and rec.get(args.name_field,''):
            key = rec.get(args.name_field)          # e.g. 'Pilbara Craton', 'Fortescue Basin'
        col = PARENT_COLORS.get(key, DEFAULT_COLOR)
        poly_rings=[]
        for r in rings:
            if len(r)>=3:
                ring_coords.append(np.asarray(r,dtype=np.float32))
                ring_color.append(col); ring_parent.append(key)
                ring_name.append(rec.get(args.name_field,''))
                raw=rec.get(args.colour_field,''); tc,fmt=parse_colour(raw); ring_tc.append(tc)
                fmt_count[fmt]=fmt_count.get(fmt,0)+1
                if len(examples.setdefault(fmt,[]))<4 and raw not in examples[fmt]:
                    examples[fmt].append(raw)
                poly_rings.append([[float(x),float(y)] for x,y in r])
        if poly_rings:
            features.append({"type":"Feature",
                "properties":{k:rec.get(k,'') for k in
                    ['TECTNAME','TECTCODE','PARENTNAME','TECTTYPE','CRATON',
                     'OROGEN','PROVINCE','BASIN','ERA_FROM','ERA_TO',
                     'MAX_AGE_MA','MIN_AGE_MA','TECTCOLOUR']},
                "geometry":{"type":"Polygon","coordinates":poly_rings}})

    if args.out_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
        miss = sorted({p for p in ring_parent if p not in PARENT_COLORS})
        if miss:
            print(f"  groups with no colour (default grey): {miss}")
        np.savez_compressed(args.out_npz,
            rings=_objarr(ring_coords),
            colors=np.array(ring_color), parents=np.array(ring_parent),
            names=np.array(ring_name), tect_colors=np.array(ring_tc),
            parent_color_keys=np.array(list(PARENT_COLORS.keys())),
            parent_color_vals=np.array(list(PARENT_COLORS.values())))
        print(f"npz: {args.out_npz} ({len(ring_coords)} rings)")
        print(f"  {args.colour_field} formats (rings): {fmt_count}")
        for fmt,ex in examples.items():
            print(f"    {fmt:8s} e.g. {ex}")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        fc={"type":"FeatureCollection",
            "crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "metadata":{"source":"GSWA GEOLOGY 10M Tectonics GDA2020",
                        "licence":"CC-BY-4.0","attribution":"Geological Survey of Western Australia"},
            "features":features}
        with open(args.out_json,'w') as f: json.dump(fc,f)
        print(f"geojson: {args.out_json} ({len(features)} features)")

if __name__=='__main__':
    main()
