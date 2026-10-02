#!/usr/bin/env python3
"""
plot_fig1.py — WA Array Figure 1: lithology (500k) + tectonic-domain boundaries
(10M) + stations + optional array footprint / section lines.

  python3 plot_fig1.py \
      --litho /workspace/shape.files/derived/wa_lithology.npz \
      --tectonic /workspace/shape.files/derived/wa_tectonics.npz \
      --stations stations.txt \
      --out fig1.png

Layering: lithology fill (base) → tectonic-domain outlines (thick) → stations.
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection, LineCollection

_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_HERE)
try:
    from plot_depth_slice import _make_ax, _add_states_ocean, _HAS_CARTOPY
    import cartopy.crs as ccrs
except Exception:
    _HAS_CARTOPY=False

def _load_npz(p):
    d=np.load(p, allow_pickle=True)
    return d['rings'], d['colors'], d.get('parents',None), \
           d.get('parent_color_keys',None), d.get('parent_color_vals',None)

def _fill(ax, rings, colors, tr, clip=None, alpha=0.8, zorder=0):
    patches=[]; pc=[]
    for r,c in zip(rings,colors):
        if len(r)<3: continue
        if clip:
            lo0,lo1,la0,la1=clip
            if r[:,0].max()<lo0 or r[:,0].min()>lo1 or r[:,1].max()<la0 or r[:,1].min()>la1: continue
        patches.append(MplPoly(r,closed=True)); pc.append(c)
    kw={'transform':tr} if tr else {}
    coll=PatchCollection(patches,facecolor=pc,edgecolor='0.6',linewidths=0.05,
                         alpha=alpha,zorder=zorder,**kw)
    ax.add_collection(coll)

def _outlines(ax, rings, parents, tr, major, clip=None, lw=1.2, zorder=6):
    segs=[]
    for r,p in zip(rings,parents if parents is not None else ['']*len(rings)):
        if len(r)<3: continue
        if major and p not in major: continue
        if clip:
            lo0,lo1,la0,la1=clip
            if r[:,0].max()<lo0 or r[:,0].min()>lo1 or r[:,1].max()<la0 or r[:,1].min()>la1: continue
        segs.append(np.asarray(r))
    kw={'transform':tr} if tr else {}
    ax.add_collection(LineCollection(segs,colors='k',linewidths=lw,zorder=zorder,**kw))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--litho', required=True)
    ap.add_argument('--tectonic', default=None)
    ap.add_argument('--stations', default=None, help='text file, cols: lon lat [code]')
    ap.add_argument('--sections', default=None, help='section file to overlay lines')
    ap.add_argument('--extent', nargs=4, type=float, default=[112,130,-36,-13])
    ap.add_argument('--mode', default='litho', choices=['litho','tectonic'],
                    help='litho=lithology fill; tectonic=tectonic fill + greenstone overlay')
    ap.add_argument('--station-cols', nargs=2, type=int, default=[0,1],
                    help='columns for lon lat in stations file (default 0 1)')
    ap.add_argument('--out', default='fig1.png')
    args=ap.parse_args()

    lo0,lo1,la0,la1=args.extent; clip=(lo0-1,lo1+1,la0-1,la1+1)
    tr=ccrs.PlateCarree() if _HAS_CARTOPY else None
    fig=plt.figure(figsize=(11,13))
    ax=_make_ax(fig,[0.06,0.05,0.90,0.90]) if _HAS_CARTOPY else fig.add_subplot(111)

    lr,lc,lp,lkeys,lvals=_load_npz(args.litho)
    if args.mode=='litho':
        # Lithology fill (base) + tectonic-domain outlines
        _fill(ax, lr, lc, tr, clip=clip, alpha=0.85, zorder=0)
        if args.tectonic:
            tr_,tc,tp,_,_=_load_npz(args.tectonic)
            major={'Yilgarn Craton','West Australian Craton','Pilbara Craton',
                   'North Australian Craton','Capricorn Orogen','Albany-Fraser Orogen',
                   'Pinjarra Orogen','Paterson Orogen','Gascoyne Province'}
            _outlines(ax, tr_, tp, tr, major, clip=clip, lw=1.3, zorder=6)
    else:
        # Tectonic fill (base) + greenstone/mafic lithology overlay
        if args.tectonic:
            tr_,tc,tp,_,_=_load_npz(args.tectonic)
            _fill(ax, tr_, tc, tr, clip=clip, alpha=0.55, zorder=0)
        # overlay only the fast rocks (greens) from lithology
        fast={'#1b7837','#4daf4a','#a6d96a'}
        gr=[r for r,c in zip(lr,lc) if c in fast]
        gc=[c for c in lc if c in fast]
        _fill(ax, gr, gc, tr, clip=clip, alpha=0.9, zorder=3)

    # Ocean mask + coastline
    if _HAS_CARTOPY: _add_states_ocean(ax); ax.set_extent(args.extent, crs=tr)
    else: ax.set_xlim(lo0,lo1); ax.set_ylim(la0,la1); ax.set_aspect('equal')

    # Stations
    if args.stations and os.path.isfile(args.stations):
        # auto-detect delimiter (comma or whitespace)
        with open(args.stations) as _f: _first=_f.readline()
        _delim=',' if ',' in _first else None
        st=np.loadtxt(args.stations, usecols=tuple(args.station_cols),
                      delimiter=_delim)
        skw={'transform':tr} if tr else {}
        ax.scatter(st[:,0], st[:,1], marker='^', s=22, c='k', edgecolor='w',
                   linewidths=0.4, zorder=10, **skw)

    # Section lines
    if args.sections and os.path.isfile(args.sections):
        with open(args.sections) as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                p=line.split(); la1_,lo1_,la2_,lo2_=map(float,p[:4])
                lkw={'transform':tr} if tr else {}
                ax.plot([lo1_,lo2_],[la1_,la2_],'r-',lw=1.5,zorder=8,**lkw)

    # Lithology legend
    _,_,_,keys,vals=_load_npz(args.litho)
    if keys is not None:
        from matplotlib.patches import Patch
        short={'mafic/ultramafic intrusive/extrusive rocks':'Mafic/ultramafic',
               'granite-greenstones':'Granite-greenstone','greenstones':'Greenstone',
               'granitic rocks':'Granite','igneous and metamorphic rocks':'Igneous+metamorphic',
               'igneous, sedimentary, and metamorphic rocks':'Mixed',
               'sedimentary rocks':'Sedimentary','sedimentary and volcanic rocks':'Sed+volcanic'}
        h=[Patch(facecolor=v,edgecolor='k',lw=0.3,label=short.get(k,k)) for k,v in zip(keys,vals)]
        ax.legend(handles=h, loc='upper left', fontsize=8, framealpha=0.92,
                  title='Lithology (GSWA 500k)', title_fontsize=9)

    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    fig.savefig(args.out.replace('.png','.pdf'), bbox_inches='tight')
    print('Saved:', args.out, '(+ .pdf)')

if __name__=='__main__': main()
