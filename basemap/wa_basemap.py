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


CRATONS = ('Yilgarn Craton', 'Pilbara Craton')   # granite-greenstone cratons only
CRATON_LITHO = {                          # make_litho key -> short label (draw order)
    'granitic rocks': 'Granite',
    'granite-greenstones': 'Granite-greenstone',
    'greenstones': 'Greenstone',
    'mafic/ultramafic intrusive/extrusive rocks': 'Mafic/ultramafic',
}


def _ring_path(rings):
    """One compound Path from many rings (for clipping)."""
    from matplotlib.path import Path
    verts, codes = [], []
    for r in rings:
        r = np.asarray(r, float)[:, :2]
        if len(r) < 3:
            continue
        verts += list(r) + [r[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 1) + [Path.CLOSEPOLY]
    return Path(np.array(verts), codes)


def _in_box(r, clip_box):
    if clip_box is None:
        return True
    lo0, lo1, la0, la1 = clip_box
    return not (r[:, 0].max() < lo0 or r[:, 0].min() > lo1 or
                r[:, 1].max() < la0 or r[:, 1].min() > la1)


def add_craton_geology(ax, litho_npz, cratons=CRATONS, classes=CRATON_LITHO,
                       alpha=0.95, zorder=1, transform=None, clip_box=None,
                       clip_to_cratons=True):
    """GSWA 500k granite + mafic/greenstone/granite-greenstone polygons, drawn
    only inside the given cratons (default Yilgarn + Pilbara), as in plot_fig1.
    These are the real 500k polygons, not a solid fill: everything else (cover
    basins, orogens, other lithologies) keeps its tectonic-domain colour.
    Returns [(label, colour)] actually drawn, for the legend."""
    d = _load()
    names = d['names'] if 'names' in d.files else [''] * len(d['rings'])
    craton_rings = [np.asarray(r) for r, p, n in zip(d['rings'], d['parents'], names)
                    if (p in cratons or n in cratons) and len(r) >= 3
                    and _in_box(np.asarray(r), clip_box)]
    kw = {'transform': transform} if transform is not None else {}
    L = np.load(litho_npz, allow_pickle=True)
    order = {k: i for i, k in enumerate(classes)}
    items = []
    for r, c, p in zip(L['rings'], L['colors'], L['parents']):
        p = str(p)
        if p not in classes:
            continue
        r = np.asarray(r)
        if len(r) < 3 or not _in_box(r, clip_box):
            continue
        items.append((order[p], r, c, p))
    items.sort(key=lambda t: t[0])           # granite first, greens on top
    if not items:
        return []
    pc = PatchCollection([MplPoly(r, closed=True) for _, r, _, _ in items],
                         facecolor=[c for _, _, c, _ in items], edgecolor='0.45',
                         linewidths=0.08, alpha=alpha, zorder=zorder, **kw)
    ax.add_collection(pc)
    if clip_to_cratons and craton_rings:
        from matplotlib.patches import PathPatch
        t = transform._as_mpl_transform(ax) if transform is not None else ax.transData
        pc.set_clip_path(PathPatch(_ring_path(craton_rings), transform=t))
    seen = {p: c for _, _, c, p in items}
    return [(classes[k], seen[k]) for k in classes if k in seen]


def has_gswa_colours(litho_npz):
    """True if the 500k npz carries parsed GSWA unit colours (tect_colors)."""
    try:
        d = np.load(litho_npz, allow_pickle=True)
        return 'tect_colors' in d.files and any(str(c) for c in d['tect_colors'])
    except Exception:
        return False


def add_gswa_units(ax, litho_npz, transform=None, clip_box=None, zorder=1,
                   alpha=1.0, edgecolor='k', lw=0.12):
    """GSWA 1:500k tectonic units in GSWA's own map colours (TECTCOLOUR), with
    thin black unit outlines, like GeoVIEW. Each record is one compound path, so
    holes are real holes and inliers show through. Units with no parsable colour
    use their lithology colour."""
    from matplotlib.path import Path
    from matplotlib.collections import PathCollection
    d = np.load(litho_npz, allow_pickle=True)
    rings = d['rings']
    tcol = d['tect_colors'] if 'tect_colors' in d.files else [''] * len(rings)
    fcol = d['colors']
    rec = d['rec_idx'] if 'rec_idx' in d.files else np.arange(len(rings))
    groups = {}
    for i, r in enumerate(rings):
        r = np.asarray(r, float)[:, :2]
        if len(r) < 3 or not _in_box(r, clip_box):
            continue
        groups.setdefault(int(rec[i]), []).append(i)
    paths, cols, areas = [], [], []
    for _, idx in groups.items():
        verts, codes = [], []
        for i in idx:
            r = np.asarray(rings[i], float)[:, :2]
            verts += list(r) + [r[0]]
            codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 1) + [Path.CLOSEPOLY]
        paths.append(Path(np.array(verts), codes))
        c = str(tcol[idx[0]]) or str(fcol[idx[0]])
        cols.append(c)
        r0 = np.asarray(rings[idx[0]], float)
        areas.append(np.ptp(r0[:, 0]) * np.ptp(r0[:, 1]))
    order = np.argsort(areas)[::-1]          # big first: any residual overlap -> small on top
    kw = {'transform': transform} if transform is not None else {}
    pc = PathCollection([paths[i] for i in order], facecolors=[cols[i] for i in order],
                        edgecolors=edgecolor, linewidths=lw, alpha=alpha, zorder=zorder, **kw)
    ax.add_collection(pc)
    return pc


def _boundaries_npz(path=None):
    cands = [path, os.environ.get('WA_BOUNDARIES_NPZ', '')]
    try:
        import config
        cands.append(getattr(config, 'BOUNDARIES_NPZ', None))
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    cands.append(os.path.join(here, '..', 'data', 'wa_crustal_boundaries.npz'))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError("wa_crustal_boundaries.npz not found (config.BOUNDARIES_NPZ, "
                            "$WA_BOUNDARIES_NPZ or data/); run extract/make_boundaries.py")


LOW_CONF = ('low', 'very low', 'none')


def add_crustal_boundaries(ax, npz=None, transform=None, clip_box=None, color='k',
                           lw_litho=1.3, lw_crust=0.7, zorder=6.5, alpha=0.9):
    """GSWA Major Crustal Boundaries: lithospheric-scale lines thick, crustal-scale thin,
    low / very low / no confidence dashed. Older npz without attributes -> one style.
    Returns the npz path used."""
    from matplotlib.collections import LineCollection
    path = _boundaries_npz(npz)
    d = np.load(path, allow_pickle=True)
    n = len(d['lines'])
    scale = d['scale'] if 'scale' in d.files else np.array(['lithospheric'] * n)
    conf = d['conf'] if 'conf' in d.files else np.array(['high'] * n)
    kw = {'transform': transform} if transform is not None else {}
    groups = {}
    for ln, sc, cf in zip(d['lines'], scale, conf):
        ln = np.asarray(ln, float)
        if len(ln) < 2:
            continue
        if clip_box is not None:
            lo0, lo1, la0, la1 = clip_box
            if (ln[:, 0].max() < lo0 or ln[:, 0].min() > lo1 or
                    ln[:, 1].max() < la0 or ln[:, 1].min() > la1):
                continue
        key = (str(sc).lower() == 'crustal', str(cf).lower() in LOW_CONF)
        groups.setdefault(key, []).append(ln)
    for (crustal, low), segs in groups.items():
        ax.add_collection(LineCollection(
            segs, colors=color, linewidths=lw_crust if crustal else lw_litho,
            linestyles=(0, (4, 2.5)) if low else 'solid', alpha=alpha, zorder=zorder, **kw))
    return path


def boundaries_legend(ax, loc='lower left', fontsize=7, color='k', lw_litho=1.3, lw_crust=0.7):
    """Key for add_crustal_boundaries line styles."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=color, lw=lw_litho, label='lithospheric boundary'),
         Line2D([], [], color=color, lw=lw_crust, label='crustal boundary'),
         Line2D([], [], color=color, lw=lw_crust, ls=(0, (4, 2.5)), label='low confidence')]
    leg = ax.legend(handles=h, loc=loc, fontsize=fontsize, framealpha=0.9,
                    title='Major crustal boundaries (GSWA)', title_fontsize=fontsize,
                    handlelength=2.2, borderpad=0.4, labelspacing=0.3)
    leg.set_zorder(21)
    return leg


def tectonic_legend(ax, loc='lower left', fontsize=7, ncol=1, clip_box=None,
                    exclude=('STATE',), litho=None):
    """Legend of the major tectonic domains.
    clip_box=(lonmin,lonmax,latmin,latmax): list only domains that have a polygon
    in that extent (i.e. what is actually visible on the map). exclude: keys never
    listed (STATE is the state outline, not a domain)."""
    d = _load()
    keys = list(d['parent_color_keys']); vals = list(d['parent_color_vals'])
    if clip_box is not None:
        lo0, lo1, la0, la1 = clip_box
        seen = set()
        for r, p in zip(d['rings'], d['parents']):
            if len(r) < 3 or p in seen:
                continue
            if not (r[:, 0].max() < lo0 or r[:, 0].min() > lo1 or
                    r[:, 1].max() < la0 or r[:, 1].min() > la1):
                seen.add(str(p))
        keep = [i for i, k in enumerate(keys) if k in seen]
    else:
        keep = range(len(keys))
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=vals[i], edgecolor='k', lw=0.3, label=keys[i])
               for i in keep if keys[i] not in exclude]
    headers = []
    if litho:   # [(label, colour)] from add_craton_geology: two titled groups
        blank = lambda t: Patch(visible=False, label=t)
        handles = ([blank('Yilgarn & Pilbara - lithology (GSWA 500k)')]
                   + [Patch(facecolor=c, edgecolor='k', lw=0.3, label=l) for l, c in litho]
                   + [blank('Tectonic domain')] + handles)
        headers = [0, len(litho) + 1]
    leg = ax.legend(handles=handles, loc=loc, fontsize=fontsize, ncol=ncol,
                    framealpha=0.92, title=None if litho else 'Tectonic domain',
                    title_fontsize=fontsize + 1,
                    handlelength=1.4, handleheight=1.0, labelspacing=0.35,
                    borderpad=0.5, handletextpad=0.5)
    for i in headers:   # group titles: bold, pulled left over the (hidden) swatch
        t = leg.get_texts()[i]
        t.set_fontweight('bold'); t.set_fontsize(fontsize + 0.5)
        t.set_position((-(1.4 + 0.5) * fontsize, 0))
    return leg


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
