#!/usr/bin/env python3
"""Extract GSWA 500k polygons colored by LITHOLOGY → npz + geojson overlay.
Fast rocks (mafic/greenstone) green, slow (granite/sediment) pink/yellow —
a velocity-oriented scheme to compare against Vs models."""
import argparse, struct, json, os
import numpy as np

def _objarr(items):
    """1-D object array of arrays. np.array(list, dtype=object) silently stacks
    equal-length arrays into an N-D object array; this never does."""
    out = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        out[i] = x
    return out


def read_shp_polygons(path):
    with open(path,'rb') as f: data=f.read()
    shapes=[]; pos=100; n=len(data)
    while pos<n:
        clen=struct.unpack('>I',data[pos+4:pos+8])[0]; rstart=pos+8
        if struct.unpack('<I',data[rstart:rstart+4])[0]==5:
            off=rstart+4+32
            nparts=struct.unpack('<I',data[off:off+4])[0]; npts=struct.unpack('<I',data[off+4:off+8])[0]; off+=8
            parts=struct.unpack(f'<{nparts}I',data[off:off+4*nparts]); off+=4*nparts
            pts=np.array(struct.unpack(f'<{2*npts}d',data[off:off+16*npts])).reshape(npts,2)
            shapes.append([pts[parts[k]:(parts[k+1] if k+1<nparts else npts)] for k in range(nparts)])
        else: shapes.append([])
        pos=rstart+2*clen
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
    pos=32; fields=[]
    while d[pos]!=0x0D:
        nm=d[pos:pos+11].split(b'\x00')[0].decode('latin1'); fl=d[pos+16]; fields.append((nm,fl)); pos+=32
    recs=[]; p=hlen
    for _ in range(nrec):
        off=p+1; row={}
        for nm,fl in fields: row[nm]=_dec(d[off:off+fl]); off+=fl
        recs.append(row); p+=rlen
    return recs

# Velocity-oriented lithology colors
LITHO_COLORS = {
    'mafic/ultramafic intrusive/extrusive rocks': '#1b7837',  # dark green (fast)
    'greenstones':                                 '#4daf4a',  # green (fast)
    'granite-greenstones':                         '#a6d96a',  # light green (fast-ish)
    'granitic rocks':                              '#f4a7b9',  # pink (slow)
    'igneous and metamorphic rocks':               '#d8b365',  # tan
    'igneous, sedimentary, and metamorphic rocks': '#c7c7c7',  # grey
    'sedimentary rocks':                           '#fee08b',  # yellow (slow)
    'sedimentary and volcanic rocks':              '#fdddaa',  # pale (slow)
}
DEFAULT='#e8e8e8'

def parse_colour(raw):
    """GSWA colour field -> '#rrggbb' (or '' if unrecognised), plus the format seen.
    Handles '#rrggbb' / 'rrggbb', 'R G B' or 'R,G,B' (0-255), CMYK as four numbers
    0-100 ('C M Y K', optionally with C/M/Y/K letters), and an Esri packed integer
    (R + 256*G + 65536*B)."""
    import re
    s = (raw or '').strip()
    if not s:
        return '', 'empty'
    m = re.fullmatch(r'#?([0-9a-fA-F]{6})', s)
    if m:
        return '#' + m.group(1).lower(), 'hex'
    nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', s)]
    clamp = lambda v: int(round(min(max(v, 0), 255)))
    if len(nums) == 3 and all(0 <= v <= 255 for v in nums):
        r, g, b = nums
        return '#%02x%02x%02x' % (clamp(r), clamp(g), clamp(b)), 'rgb'
    if len(nums) == 4 and all(0 <= v <= 100 for v in nums):
        c, m_, y, k = [v / 100 for v in nums]
        rgb = [255 * (1 - x) * (1 - k) for x in (c, m_, y)]
        return '#%02x%02x%02x' % tuple(clamp(v) for v in rgb), 'cmyk'
    if len(nums) == 1 and nums[0] == int(nums[0]) and 0 <= nums[0] < 2 ** 24:
        v = int(nums[0])
        return '#%02x%02x%02x' % (v & 255, (v >> 8) & 255, (v >> 16) & 255), 'packed'
    return '', 'unknown'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shp', required=True)
    ap.add_argument('--out-npz', required=True)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--field', default='LITHOLOGY')
    ap.add_argument('--name-field', default='TECTNAME',
                    help='per-ring unit name saved as "names" (e.g. greenstone belt names)')
    ap.add_argument('--colour-field', default='TECTCOLOUR',
                    help='GSWA map colour per unit, saved as "tect_colors" (index-map style)')
    args=ap.parse_args()
    shapes=read_shp_polygons(args.shp); recs=read_dbf(args.shp.replace('.shp','.dbf'))
    print(f"{len(shapes)} polygons")
    rc=[]; rcol=[]; rlab=[]; rname=[]; rtc=[]; rraw=[]; rrec=[]; features=[]
    fmt_count={}; examples={}
    for irec,(rings,rec) in enumerate(zip(shapes,recs)):
        key=rec.get(args.field,''); col=LITHO_COLORS.get(key,DEFAULT)
        raw=rec.get(args.colour_field,''); tc,fmt=parse_colour(raw)
        fmt_count[fmt]=fmt_count.get(fmt,0)+1
        examples.setdefault(fmt,[]).append(raw) if len(examples.get(fmt,[]))<4 else None
        pr=[]
        for r in rings:
            if len(r)>=3:
                rc.append(np.asarray(r,np.float32)); rcol.append(col); rlab.append(key)
                rname.append(rec.get(args.name_field,''))
                rtc.append(tc); rraw.append(raw); rrec.append(irec)
                pr.append([[float(x),float(y)] for x,y in r])
        if pr:
            features.append({"type":"Feature",
                "properties":{k:rec.get(k,'') for k in ['TECTNAME','LITHOLOGY','TECTCODE','PARENTNAME','TECTCOLOUR']},
                "geometry":{"type":"Polygon","coordinates":pr}})
    np.savez_compressed(args.out_npz,
        rings=_objarr(rc), colors=np.array(rcol), parents=np.array(rlab),
        names=np.array(rname), tect_colors=np.array(rtc),
        tect_colour_raw=np.array(rraw), rec_idx=np.array(rrec),
        parent_color_keys=np.array(list(LITHO_COLORS.keys())),
        parent_color_vals=np.array(list(LITHO_COLORS.values())))
    print(f"npz: {args.out_npz} ({len(rc)} rings)")
    print(f"  {args.colour_field} formats: {fmt_count}")
    for fmt,ex in examples.items():
        print(f"    {fmt:8s} e.g. {ex}")
    if fmt_count.get('unknown') or fmt_count.get('empty',0)==sum(fmt_count.values()):
        print(f"  NOTE: some/all {args.colour_field} values not understood - those units "
              f"fall back to the lithology colour")
    if args.out_json:
        fc={"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "metadata":{"source":"GSWA GEOLOGY 500k Tectonics GDA2020","licence":"CC-BY-4.0",
                        "attribution":"Geological Survey of Western Australia"},"features":features}
        json.dump(fc,open(args.out_json,'w'))
        print(f"geojson: {args.out_json} ({len(features)} features)")

if __name__=='__main__': main()
