#!/usr/bin/env python3
"""Extract GSWA Major Crustal Boundaries (polylines) → npz for overlay.
These are the terrane/suture boundary LINES, drawn over maps and marked where
cross-sections cross them."""
import argparse, struct, json, os
import numpy as np

def _objarr(items):
    """1-D object array of arrays. np.array(list, dtype=object) silently stacks
    equal-length arrays into an N-D object array; this never does."""
    out = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        out[i] = x
    return out


def read_shp_polylines(path):
    """Returns (lines, rec_idx): every polyline part, and the DBF record each
    part belongs to (multipart records give several parts; null shapes none)."""
    with open(path,'rb') as f: data=f.read()
    lines=[]; rec_idx=[]; pos=100; n=len(data); irec=0
    while pos<n:
        clen=struct.unpack('>I',data[pos+4:pos+8])[0]; rstart=pos+8
        st=struct.unpack('<I',data[rstart:rstart+4])[0]
        if st==3:  # PolyLine
            off=rstart+4+32
            nparts=struct.unpack('<I',data[off:off+4])[0]; npts=struct.unpack('<I',data[off+4:off+8])[0]; off+=8
            parts=struct.unpack(f'<{nparts}I',data[off:off+4*nparts]); off+=4*nparts
            pts=np.array(struct.unpack(f'<{2*npts}d',data[off:off+16*npts])).reshape(npts,2)
            for k in range(nparts):
                a=parts[k]; b=parts[k+1] if k+1<nparts else npts
                lines.append(pts[a:b]); rec_idx.append(irec)
        pos=rstart+2*clen; irec+=1
    return lines, rec_idx

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
    return recs, [f[0] for f in fields]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shp', required=True)
    ap.add_argument('--out-npz', required=True)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--name-field', default=None, help='DBF field for boundary name (auto if omitted)')
    args=ap.parse_args()
    lines,rec_idx=read_shp_polylines(args.shp)
    recs,flds=read_dbf(args.shp.replace('.shp','.dbf'))
    print(f"{len(lines)} polyline parts, {len(recs)} records, fields: {flds}")
    # name field: first text field that looks like a name
    nf=args.name_field
    if nf is None:
        for cand in ['NAME','BOUNDARY','BNDNAME','FEATURE','TYPE','CLASS','Name']:
            if cand in flds: nf=cand; break
    rec_names=[r.get(nf,'') for r in recs] if nf else ['']*len(recs)
    names=[rec_names[i] if i<len(rec_names) else '' for i in rec_idx]   # one per part
    # attributes for map styling (both the 2021 2.5M and 2025 versions carry these;
    # the type field is 'TYPE' in one and 'TYPE_' in the other)
    def per_part(*cands):
        f=next((c for c in cands if c in flds), None)
        vals=[r.get(f,'') for r in recs] if f else ['']*len(recs)
        return np.array([vals[i] if i<len(vals) else '' for i in rec_idx]), f
    scale,f_s=per_part('FEAT_SCALE'); conf,f_c=per_part('FEAT_CONF'); btype,f_t=per_part('TYPE','TYPE_')
    print(f"  attributes: scale<-{f_s} conf<-{f_c} type<-{f_t}")
    np.savez_compressed(args.out_npz,
        lines=_objarr([np.asarray(l,np.float32) for l in lines]),
        names=np.array(names), rec_idx=np.array(rec_idx),
        scale=scale, conf=conf, btype=btype,
        name_field=np.array([nf or '']))
    print(f"  {sum(1 for x in names if x)} of {len(names)} parts named, "
          f"{len(set(x for x in names if x))} distinct names")
    print(f"npz: {args.out_npz} ({len(lines)} lines, name field '{nf}')")
    if args.out_json:
        feats=[{"type":"Feature","properties":{"NAME":nm},"geometry":{"type":"LineString",
                "coordinates":[[float(x),float(y)] for x,y in l]}}
               for l,nm in zip(lines,names) if len(l)>=2]
        json.dump({"type":"FeatureCollection","metadata":{"source":"GSWA Major Crustal Boundaries 2025",
            "licence":"CC-BY-4.0","attribution":"Geological Survey of Western Australia"},
            "features":feats}, open(args.out_json,'w'))
        print(f"geojson: {args.out_json}")

if __name__=='__main__': main()
