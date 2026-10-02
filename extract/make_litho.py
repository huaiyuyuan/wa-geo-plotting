#!/usr/bin/env python3
"""Extract GSWA 500k polygons colored by LITHOLOGY → npz + geojson overlay.
Fast rocks (mafic/greenstone) green, slow (granite/sediment) pink/yellow —
a velocity-oriented scheme to compare against Vs models."""
import argparse, struct, json, os
import numpy as np

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

def read_dbf(path):
    with open(path,'rb') as f: d=f.read()
    nrec=struct.unpack('<I',d[4:8])[0]; hlen=struct.unpack('<H',d[8:10])[0]; rlen=struct.unpack('<H',d[10:12])[0]
    pos=32; fields=[]
    while d[pos]!=0x0D:
        nm=d[pos:pos+11].split(b'\x00')[0].decode('latin1'); fl=d[pos+16]; fields.append((nm,fl)); pos+=32
    recs=[]; p=hlen
    for _ in range(nrec):
        off=p+1; row={}
        for nm,fl in fields: row[nm]=d[off:off+fl].decode('latin1').strip(); off+=fl
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shp', required=True)
    ap.add_argument('--out-npz', required=True)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--field', default='LITHOLOGY')
    args=ap.parse_args()
    shapes=read_shp_polygons(args.shp); recs=read_dbf(args.shp.replace('.shp','.dbf'))
    print(f"{len(shapes)} polygons")
    rc=[]; rcol=[]; rlab=[]; features=[]
    for rings,rec in zip(shapes,recs):
        key=rec.get(args.field,''); col=LITHO_COLORS.get(key,DEFAULT)
        pr=[]
        for r in rings:
            if len(r)>=3:
                rc.append(np.asarray(r,np.float32)); rcol.append(col); rlab.append(key)
                pr.append([[float(x),float(y)] for x,y in r])
        if pr:
            features.append({"type":"Feature",
                "properties":{k:rec.get(k,'') for k in ['TECTNAME','LITHOLOGY','TECTCODE','PARENTNAME','TECTCOLOUR']},
                "geometry":{"type":"Polygon","coordinates":pr}})
    np.savez_compressed(args.out_npz,
        rings=np.array(rc,dtype=object), colors=np.array(rcol), parents=np.array(rlab),
        parent_color_keys=np.array(list(LITHO_COLORS.keys())),
        parent_color_vals=np.array(list(LITHO_COLORS.values())), allow_pickle=True)
    print(f"npz: {args.out_npz} ({len(rc)} rings)")
    if args.out_json:
        fc={"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "metadata":{"source":"GSWA GEOLOGY 500k Tectonics GDA2020","licence":"CC-BY-4.0",
                        "attribution":"Geological Survey of Western Australia"},"features":features}
        json.dump(fc,open(args.out_json,'w'))
        print(f"geojson: {args.out_json} ({len(features)} features)")

if __name__=='__main__': main()
