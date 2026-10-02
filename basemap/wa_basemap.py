#!/usr/bin/env python3
"""
wa_basemap.py — WA tectonic background for Fig 1 and map overlays.

Loads the GSWA tectonic polygons (from wa_tectonics.npz, extracted from
GEOLOGY_10M_Tectonics_GDA2020) and paints them colored by major tectonic
domain (PARENTNAME). Use add_tectonic_background(ax) on any lon/lat axes.

  from wa_basemap import add_tectonic_background, tectonic_legend
  add_tectonic_background(ax, alpha=0.6, edgecolor='k', lw=0.2)
  tectonic_legend(ax)              # optional domain legend

Data file resolved from (in order): $WA_TECTONICS_NPZ, alongside this file,
or ./wa_tectonics.npz.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

def _find_npz():
    for p in [os.environ.get('WA_TECTONICS_NPZ',''),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wa_tectonics.npz'),
              'wa_tectonics.npz']:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError("wa_tectonics.npz not found (set $WA_TECTONICS_NPZ)")

_CACHE = None
def _load():
    global _CACHE
    if _CACHE is None:
        d = np.load(_find_npz(), allow_pickle=True)
        _CACHE = d
    return _CACHE

def add_tectonic_background(ax, alpha=0.55, edgecolor='0.25', lw=0.15,
                            zorder=0, transform=None, clip_box=None):
    """Paint WA tectonic domains on ax. clip_box=(lonmin,lonmax,latmin,latmax)
    to only draw rings overlapping that extent (speeds up small maps)."""
    d = _load()
    rings = d['rings']; colors = d['colors']
    patches=[]; pcolors=[]
    for r, c in zip(rings, colors):
        if len(r) < 3: continue
        if clip_box is not None:
            lo0,lo1,la0,la1 = clip_box
            if (r[:,0].max()<lo0 or r[:,0].min()>lo1 or
                r[:,1].max()<la0 or r[:,1].min()>la1): continue
        patches.append(MplPoly(r, closed=True))
        pcolors.append(c)
    kw = {}
    if transform is not None: kw['transform'] = transform
    pc = PatchCollection(patches, facecolor=pcolors, edgecolor=edgecolor,
                         linewidths=lw, alpha=alpha, zorder=zorder, **kw)
    ax.add_collection(pc)
    return pc

def add_tectonic_outlines(ax, edgecolor='k', lw=0.4, alpha=0.6, zorder=8,
                          transform=None, clip_box=None, major_only=False):
    """Draw ONLY the tectonic boundary outlines (no fill) — for overlaying on
    velocity/xi maps so anomalies can be read against domain boundaries.
    major_only=True draws only the big domains (craton/orogen outer edges)."""
    d = _load()
    rings = d['rings']; parents = d['parents']
    from matplotlib.collections import LineCollection
    segs=[]
    major = {'Yilgarn Craton','West Australian Craton','Pilbara Craton',
             'North Australian Craton','Capricorn Orogen','Albany-Fraser Orogen',
             'Pinjarra Orogen','Paterson Orogen','Halls Creek Orogen'}
    for r, p in zip(rings, parents):
        if len(r) < 3: continue
        if major_only and p not in major: continue
        if clip_box is not None:
            lo0,lo1,la0,la1 = clip_box
            if (r[:,0].max()<lo0 or r[:,0].min()>lo1 or
                r[:,1].max()<la0 or r[:,1].min()>la1): continue
        segs.append(np.asarray(r))
    kw = {}
    if transform is not None: kw['transform'] = transform
    lc = LineCollection(segs, colors=edgecolor, linewidths=lw, alpha=alpha,
                        zorder=zorder, **kw)
    ax.add_collection(lc)
    return lc


def tectonic_legend(ax, loc='lower left', fontsize=7, ncol=1):
    """Add a legend of the major tectonic domains actually colored."""
    d = _load()
    keys = d['parent_color_keys']; vals = d['parent_color_vals']
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=v, edgecolor='k', lw=0.3, label=k)
               for k,v in zip(keys, vals)]
    ax.legend(handles=handles, loc=loc, fontsize=fontsize, ncol=ncol,
              framealpha=0.92, title='Tectonic domain', title_fontsize=fontsize+2,
              handlelength=1.6, handleheight=1.3, labelspacing=0.5,
              borderpad=0.7)

if __name__ == '__main__':
    # Test render: Fig-1 style base map over WA
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='wa_fig1_basemap.png')
    ap.add_argument('--extent', nargs=4, type=float, default=[112,130,-36,-13],
                    help='lonmin lonmax latmin latmax')
    ap.add_argument('--stations', default=None, help='optional lon,lat file (2 cols)')
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(10,11))
    add_tectonic_background(ax, alpha=0.65)
    lo0,lo1,la0,la1 = args.extent
    ax.set_xlim(lo0,lo1); ax.set_ylim(la0,la1)
    ax.set_aspect('equal'); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('WA tectonic domains (GSWA 10M Tectonics)', fontsize=11)
    if args.stations and os.path.isfile(args.stations):
        st = np.loadtxt(args.stations)
        ax.scatter(st[:,0], st[:,1], marker='^', s=30, c='k',
                   edgecolor='w', lw=0.4, zorder=10, label='Stations')
    tectonic_legend(ax)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print('Saved:', args.out)
