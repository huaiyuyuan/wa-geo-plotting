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


MAP2022_CREDIT = ('Domain colours after GSWA (2022) 1:10 000 000 Simplified tectonic map of WA;\n'
                  'units © State of Western Australia (GSWA), CC BY 4.0')

_MAP2022 = None


def map2022_table():
    """{lower-case unit name: '#rrggbb'} from tectonic_map_2022_colours.csv (next to this file)."""
    global _MAP2022
    if _MAP2022 is None:
        import csv
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'tectonic_map_2022_colours.csv')
        _MAP2022 = {}
        if os.path.isfile(path):
            with open(path) as f:
                rows = csv.reader(l for l in f if not l.startswith('#'))
                next(rows, None)
                for r in rows:
                    if len(r) == 2 and (r[1].strip().startswith('#') or r[1].strip() == 'none'):
                        _MAP2022[r[0].strip().lower()] = r[1].strip()   # 'none' = hidden
    return _MAP2022


def _signed_area(r):
    return 0.5 * (np.dot(r[:, 0], np.roll(r[:, 1], -1)) - np.dot(r[:, 1], np.roll(r[:, 0], -1)))


def _compound(rings):
    """One Path from the rings of one unit, filled correctly under matplotlib's
    nonzero rule: rings nested an odd number of times in the others are holes
    (wound clockwise), the rest outer rings (anticlockwise)."""
    from matplotlib.path import Path
    paths = [Path(r) for r in rings]
    verts, codes = [], []
    for i, r in enumerate(rings):
        depth = sum(paths[j].contains_point(r[0]) for j in range(len(rings)) if j != i)
        ccw = _signed_area(r) > 0
        if ccw == bool(depth % 2):          # hole must be CW, outer CCW
            r = r[::-1]
        verts += list(r) + [r[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 1) + [Path.CLOSEPOLY]
    return Path(np.array(verts), codes)


def add_map2022_units(ax, npz=None, transform=None, clip_box=None, zorder=0,
                      edgecolor='0.3', lw=0.2, alpha=1.0, verbose=True):
    """GSWA 10M tectonic units filled like the 2022 Simplified Tectonic Map of WA:
    colour by unit name (TECTNAME), then its parent, from tectonic_map_2022_colours.csv;
    then the shapefile TECTCOLOUR; then the parent-domain colour. Rings of one unit
    form one compound path (holes are holes); units are drawn big-first so inliers
    sit on top. Returns {source: n_units} for the colour sources used."""
    from matplotlib.collections import PathCollection
    d = np.load(npz, allow_pickle=True) if npz else _load()
    rings = d['rings']
    names = d['names'].astype(str) if 'names' in d.files else d['parents'].astype(str)
    parents = d['parents'].astype(str)
    tcol = d['tect_colors'].astype(str) if 'tect_colors' in d.files else [''] * len(rings)
    key = d['rec_idx'] if 'rec_idx' in d.files else names     # one record / unit
    table = map2022_table()
    groups = {}
    for i, r in enumerate(rings):
        r = np.asarray(r, float)[:, :2]
        if len(r) < 3 or not _in_box(r, clip_box):
            continue
        groups.setdefault(key[i], []).append(i)
    items, src = [], {}
    proj = getattr(ax, 'projection', None)
    pre = proj is not None and hasattr(transform, 'transform_points')
    if pre:   # project vertices ourselves: keeps the hole winding, avoids cartopy's path re-noding
        P = lambda r: proj.transform_points(transform, r[:, 0], r[:, 1])[:, :2]
    for idx in groups.values():
        i0 = idx[0]
        c, s = (table.get(names[i0].strip().lower()), 'map2022')
        if not c:
            c, s = table.get(parents[i0].strip().lower()), 'map2022 (parent)'
        if not c:
            c, s = (tcol[i0], 'TECTCOLOUR') if tcol[i0] else (str(d['colors'][i0]), 'domain')
        if c == 'none':                     # concealed on the 2022 map (e.g. under Eucla Basin)
            src['hidden'] = src.get('hidden', 0) + 1
            continue
        src[s] = src.get(s, 0) + 1
        rr = [np.asarray(rings[i], float)[:, :2] for i in idx]
        area = max(abs(_signed_area(r)) for r in rr)
        if pre:
            rr = [P(r) for r in rr]
        items.append((area, _compound(rr), c))
    items.sort(key=lambda t: -t[0])
    kw = {'transform': ax.transData if pre else transform} if transform is not None else {}
    ax.add_collection(PathCollection([p for _, p, _ in items], facecolors=[c for *_, c in items],
                                     edgecolors=edgecolor, linewidths=lw, alpha=alpha,
                                     zorder=zorder, **kw))
    if verbose:
        print('  map2022 unit colours: ' + ', '.join(f'{v} {k}' for k, v in src.items()))
    return src


_HERE = os.path.dirname(os.path.abspath(__file__))
_NARROW = None


def _narrow_font():
    """A condensed sans like the GSWA map's Arial Narrow if installed, else DejaVu Sans
    (text is fitted to the original lengths either way)."""
    global _NARROW
    if _NARROW is None:
        from matplotlib import font_manager
        have = {f.name for f in font_manager.fontManager.ttflist}
        _NARROW = next((f for f in ('Arial Narrow', 'Liberation Sans Narrow', 'Nimbus Sans Narrow',
                                    'Arial') if f in have), 'DejaVu Sans')
    return _NARROW


def add_map2022_legend(ax, loc='upper left', width_in=3.2, anchor=None, fontsize=7,
                       title='Tectonic units: rock type and age (GSWA 2022, CC BY 4.0)', zorder=22):
    """The legend chart of the GSWA 2022 simplified tectonic map (rock type x age), redrawn
    as vectors from tectonic_map_2022_legend.json (extract/make_map2022_assets.py), on a
    white panel width_in wide. loc = corner; anchor = (x, y) axes fraction of that corner.
    Call after the map extent is set. Returns the inset axes."""
    import json
    from matplotlib.patches import Polygon as Poly
    path = os.path.join(_HERE, 'tectonic_map_2022_legend.json')
    if not os.path.isfile(path):
        print(f"  (no {path}: map legend not drawn)")
        return None
    L = json.load(open(path))
    W, H = L['width'], L['height']
    pad, top = 4.0, (13.0 if title else 4.0)                   # PDF pt
    fig = ax.figure
    ax.apply_aspect()
    pos = ax.get_position()
    fw, fh = fig.get_size_inches()
    k = width_in / (W + 2 * pad)                               # inches per PDF pt
    w_ax = width_in / (pos.width * fw)
    h_ax = k * (H + pad + top) / (pos.height * fh)
    if anchor is None:
        anchor = {'upper left': (0.01, 0.99), 'upper right': (0.99, 0.99),
                  'lower left': (0.01, 0.01), 'lower right': (0.99, 0.01)}[loc]
    x0 = anchor[0] - (w_ax if 'right' in loc else 0)
    y0 = anchor[1] - (h_ax if 'upper' in loc else 0)
    ia = ax.inset_axes([x0, y0, w_ax, h_ax], zorder=zorder)
    ia.set_xlim(-pad, W + pad); ia.set_ylim(-pad, H + top)
    ia.set_xticks([]); ia.set_yticks([])
    ia.set_facecolor('white'); ia.patch.set_alpha(0.95)
    for sp in ia.spines.values():
        sp.set_edgecolor('0.6'); sp.set_linewidth(0.6)
    pt = k * 72.0                                              # 1 PDF pt -> points on the figure
    for sh in L['shapes']:
        xy = np.asarray(sh['pts'], float)
        if sh['fill'] or sh['closed']:
            ia.add_patch(Poly(xy, closed=True, facecolor=sh['fill'] or 'none',
                              edgecolor=sh['stroke'] or 'none',
                              linewidth=(sh['lw'] * pt) if sh['stroke'] else 0))
        else:
            ia.plot(xy[:, 0], xy[:, 1], color=sh['stroke'] or 'k',
                    lw=max(sh['lw'] * pt, 0.3), solid_capstyle='butt')
    fam = _narrow_font()
    renderer = fig.canvas.get_renderer()
    for t in L['texts']:
        if t['rot']:
            xy, length = ((t['x0'] + t['x1']) / 2, (t['y0'] + t['y1']) / 2), t['y1'] - t['y0']
        else:
            xy, length = ((t['x0'] + t['x1']) / 2, t['y'] + 0.42 * t['size']), t['x1'] - t['x0']
        txt = ia.text(*xy, t['s'], rotation=t['rot'], ha='center', va='center',
                      fontsize=t['size'] * pt, family=fam)
        bb = txt.get_window_extent(renderer)
        got = (bb.height if t['rot'] else bb.width) * 72.0 / fig.dpi
        want = length * pt
        if got > 0 and abs(got / want - 1) > 0.03:             # fit to the original length
            txt.set_fontsize(t['size'] * pt * want / got)
    if title:
        ia.text(-pad + 3, H + top - 2.5, title, ha='left', va='top', fontsize=fontsize,
                weight='bold')
    return ia


_LABEL_STYLE = {      # class -> colour, weight; letter-spaced = orogens (as on the GSWA map)
    'craton':  ('#ed1c24', 'bold', False),
    'terrane': ('#ed1c24', 'normal', False),
    'orogen':  ('#717774', 'normal', True),
    'basin':   ('#000000', 'normal', False),
}


def add_map2022_labels(ax, transform=None, level=1, scale=1.4, clip_box=None, zorder=6.8,
                       halo=True):
    """Unit names at the GSWA 2022 map's own label positions (tectonic_map_2022_labels.csv):
    cratons red bold, terranes/inliers red, orogens grey letter-spaced, basins and provinces
    black. level 1 = major units only, 2 = all. scale multiplies the A4 map's font sizes.
    Angles are the map's (Albers screen angles). Returns the number drawn."""
    import csv
    import matplotlib.patheffects as pe
    path = os.path.join(_HERE, 'tectonic_map_2022_labels.csv')
    if not os.path.isfile(path):
        print(f"  (no {path}: unit labels not drawn)")
        return 0
    rows = csv.DictReader(l for l in open(path, encoding='utf-8') if not l.startswith('#'))
    fam = _narrow_font()
    eff = [pe.withStroke(linewidth=1.8, foreground='white', alpha=0.75)] if halo else None
    kw = {'transform': transform} if transform is not None else {}
    n = 0
    for r in rows:
        if int(r['level']) > level:
            continue
        lon, lat = float(r['lon']), float(r['lat'])
        if clip_box is not None and not (clip_box[0] <= lon <= clip_box[1]
                                         and clip_box[2] <= lat <= clip_box[3]):
            continue
        col, wt, spaced = _LABEL_STYLE.get(r['class'], ('k', 'normal', False))
        text = r['text'].replace('\\n', '\n')
        if spaced:
            # letter-spaced with plain spaces (every font has them; thin/en spaces are
            # missing from e.g. Liberation Sans Narrow): 1 between letters, 3 between words
            text = '\n'.join('   '.join(' '.join(w) for w in line.split(' '))
                             for line in text.split('\n'))
        ax.text(lon, lat, text, color=col, fontsize=float(r['size_pt']) * scale, family=fam,
                weight='bold' if int(r['bold']) else wt, rotation=float(r['angle']),
                rotation_mode='anchor', ha='center', va='center', multialignment='center',
                linespacing=1.0, zorder=zorder, path_effects=eff, clip_on=True, **kw)
        n += 1
    return n


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
