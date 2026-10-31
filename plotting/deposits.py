#!/usr/bin/env python3
"""
deposits.py - shared mineral-deposit / greenstone overlay for ALL WA map plots.

ONE copy of the recipe from the deposit-overlay handoff, imported by every Python
map tool (plot_vsv_tectonic.py, plot_depth_slice.py, plot_moho_map.py, ...) so the
datasets, sizing and colours stay identical across Moho / Vsv / xi / Viso maps.

Marker AREA (pts^2), not diameter, carries the magnitude (contained metal, kg):
  lv = log10(size); frac = (lv-min)/(max-min); area = AMIN + (AMAX-AMIN)*frac^GAMMA
GAMMA>1 keeps small/medium deposits small so the giants stand out.

Datasets live in the repo matlab/ dir (default) or --deposit-dir:
  gold/copper/lithium/nickel_deposits.mat  (struct: lon lat size name unit)
  greenstones_sw_yilgarn.mat               (struct greenstones: [lon lat] NaN rings)

Typical use in a map tool:
    from deposits import add_deposit_args, overlay_deposits
    add_deposit_args(ap)                 # adds --deposits / --greenstones / --deposit-dir
    ...
    overlay_deposits(ax, args, transform=tr)   # AFTER the field + basemap are drawn
"""
import os
import numpy as np
from scipy.io import loadmat

DEP_AREA = (3.0, 260.0)     # marker area range (pts^2); half of the original [6 520]
DEP_GAMMA = 2.0             # >1 lets the giants stand out

# name: (matfile basename, struct var, marker, facecolor, filled)
DEPOSITS = {
    'gold':    ('gold_deposits.mat',    'gold',    'o', (1.00, 0.90, 0.00), True),
    'copper':  ('copper_deposits.mat',  'copper',  '^', (0.85, 0.00, 0.00), True),
    'lithium': ('lithium_deposits.mat', 'lithium', 'h', (0.60, 0.20, 0.80), True),
    'nickel':  ('nickel_deposits.mat',  'nickel',  'h', (0.20, 0.70, 0.20), True),
}
GREENSTONE_FILE = 'greenstones_sw_yilgarn.mat'
GREENSTONE_VAR = 'greenstones'
GREENSTONE_COLOR = (0.0, 0.35, 0.0)

_HERE = os.path.dirname(os.path.abspath(__file__))
# bundled under data/deposits/ ($WA_DEPOSIT_DIR overrides); the repo matlab/ was the old home
DEFAULT_DIR = os.environ.get('WA_DEPOSIT_DIR',
                             os.path.abspath(os.path.join(_HERE, '..', 'data', 'deposits')))


def _xy(lon, lat, xform):
    """Projected coords for a plain axes; lon/lat untouched for a cartopy transform."""
    if callable(xform):           # a lon/lat -> (x,y) projection function (e.g. wa_albers)
        return xform(lon, lat)
    return lon, lat               # None, or a cartopy transform passed via scatter kwargs


def draw_one(ax, matfile, var, marker, facecolor, filled=True,
             area=DEP_AREA, gamma=DEP_GAMMA, transform=None, projfn=None):
    """Draw one commodity. transform = a cartopy transform (kept lon/lat); projfn =
    a lon/lat->(x,y) function for a plain projected axes. Returns the PathCollection
    or None if the file is missing."""
    if not matfile or not os.path.isfile(matfile):
        print(f"  ({var}: file not found, skipped: {matfile})")
        return None
    g = loadmat(matfile, squeeze_me=True, struct_as_record=False)[var]
    lon = np.asarray(g.lon, float).ravel()
    lat = np.asarray(g.lat, float).ravel()
    v = np.asarray(g.size, float).ravel()
    v[(v <= 0) | ~np.isfinite(v)] = np.nan
    lv = np.log10(v)
    lo, hi = np.nanmin(lv), np.nanmax(lv)
    frac = (lv - lo) / max(hi - lo, 1e-12)
    sz = area[0] + (area[1] - area[0]) * frac ** gamma
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(sz)
    x, y = _xy(lon, lat, projfn)
    kw = dict(s=sz[ok], marker=marker, linewidths=0.3, zorder=6)
    if transform is not None:
        kw['transform'] = transform
    if filled:
        h = ax.scatter(np.asarray(x)[ok], np.asarray(y)[ok], facecolors=[facecolor],
                       edgecolors='k', alpha=0.7, **kw)
    else:
        h = ax.scatter(np.asarray(x)[ok], np.asarray(y)[ok], facecolors='none',
                       edgecolors=[facecolor], **kw)
    print(f"{var}: {int(np.count_nonzero(ok))} deposits (size {np.nanmin(v):.3g}-{np.nanmax(v):.3g})")
    return h


def draw_greenstones(ax, matfile, color=GREENSTONE_COLOR, alpha=0.5,
                     transform=None, projfn=None):
    """Fill greenstone-belt rings (NaN-separated [lon lat]) under the deposits."""
    if not matfile or not os.path.isfile(matfile):
        print(f"  (greenstones: file not found, skipped: {matfile})")
        return
    g = loadmat(matfile, squeeze_me=True, struct_as_record=False)[GREENSTONE_VAR]
    ll = np.asarray(g, float) if not hasattr(g, GREENSTONE_VAR) else np.asarray(getattr(g, GREENSTONE_VAR), float)
    ll = np.atleast_2d(ll)
    if ll.shape[1] != 2 and ll.shape[0] == 2:
        ll = ll.T
    brk = np.where(~np.isfinite(ll[:, 0]))[0]           # NaN rows split rings
    bounds = np.r_[-1, brk, len(ll)]
    n = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        ring = ll[a + 1:b]
        if ring.shape[0] < 3:
            continue
        x, y = _xy(ring[:, 0], ring[:, 1], projfn)
        kw = dict(facecolor=color, edgecolor='none', alpha=alpha, zorder=2)
        if transform is not None:
            kw['transform'] = transform
        ax.fill(np.asarray(x), np.asarray(y), **kw)
        n += 1
    print(f"greenstones: {n} rings")


def legend_handles(names, size=8):
    """Line2D proxy handles for a legend, one per active commodity (fixed marker size
    — the on-map size encodes magnitude, the legend just identifies the commodity)."""
    from matplotlib.lines import Line2D
    if isinstance(names, str):
        names = [s.strip() for s in names.split(',') if s.strip()]
    hs = []
    for nm in names:
        if nm not in DEPOSITS:
            continue
        _, _, marker, col, filled = DEPOSITS[nm]
        if filled:
            hs.append(Line2D([], [], marker=marker, color='none', markerfacecolor=col,
                             markeredgecolor='k', markersize=size, label=nm.capitalize()))
        else:
            hs.append(Line2D([], [], marker=marker, color='none', markerfacecolor='none',
                             markeredgecolor=col, markersize=size, label=nm.capitalize()))
    return hs


def add_deposit_args(ap):
    """Register the shared overlay CLI args on an argparse parser (default: none)."""
    ap.add_argument('--deposits', default='',
                    help="comma list of commodity overlays: gold,copper,lithium,nickel (default none)")
    ap.add_argument('--greenstones', action='store_true', help="fill SW-Yilgarn greenstone belts")
    ap.add_argument('--deposit-dir', default=DEFAULT_DIR,
                    help="dir holding the deposit/greenstone .mat files (default: repo matlab/)")
    ap.add_argument('--deposit-gamma', type=float, default=DEP_GAMMA)


def overlay_deposits(ax, args, transform=None, projfn=None):
    """Draw whatever --deposits / --greenstones ask for, AFTER the field + basemap.
    Pass transform for a cartopy axes, or projfn (lon/lat->x,y) for a projected one."""
    mindir = getattr(args, 'deposit_dir', DEFAULT_DIR)
    gamma = getattr(args, 'deposit_gamma', DEP_GAMMA)
    if getattr(args, 'greenstones', False):
        draw_greenstones(ax, os.path.join(mindir, GREENSTONE_FILE),
                         transform=transform, projfn=projfn)
    names = [s.strip() for s in getattr(args, 'deposits', '').split(',') if s.strip()]
    for nm in names:
        if nm not in DEPOSITS:
            print(f"  (unknown deposit '{nm}'; known: {', '.join(DEPOSITS)})")
            continue
        fn, var, marker, col, filled = DEPOSITS[nm]
        draw_one(ax, os.path.join(mindir, fn), var, marker, col, filled,
                 gamma=gamma, transform=transform, projfn=projfn)
