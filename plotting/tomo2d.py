#!/usr/bin/env python3
"""
tomo2d.py - surface-wave 2-D tomography maps (TransD.region.{group,phase}.npz, v3b):
loading, the WA Array coverage mask, stations, and an Albers map frame drawn without
cartopy's GeoAxes (coastline / states from Natural Earth when cartopy is installed).

Shared by plot_dispersion_maps.py (single-period maps) and plot_domain_dispersion.py
(per-domain dispersion curves).

npz keys (rj_tomo, v3b): fk (periods, s), vall2 [nper, 70, 60] (km/s), std2 (posterior
std), maski (NaN outside posterior std <= 0.80), lat2d, lon2d. Orientation comes from
lat2d/lon2d only (never re-derived from Average.out).

Coverage mask (all WA Array maps): footprint npz key 'density' = 10 s path samples per
cell, paths with both ends in the deployed arrays (W????, C[WESC]??, F???; not FORT,
FITZ, KIM*, CA*) -> density >= 10, interior holes filled, largest connected region, no
dilation. (Key 'mask' in the same file is the old v2 mask: threshold 40, dilated 3.)
Figure mask = coverage AND posterior std <= 0.80 (finite maski) AND finite velocity.
"""
import os, re

# Natural Earth data (no internet on d204403): set CARTOPY_DATA_DIR before cartopy is imported,
# exactly as plot_depth_slice.py does - cartopy reads it at import as its pre-downloaded data dir
for _c in ('/data/25/yuan/mfiles/cartopy_data', '/mnt/data64/yuan/mfiles/cartopy_data',
           os.path.expanduser('~/.local/share/cartopy')):
    if os.path.isdir(_c) and not os.environ.get('CARTOPY_DATA_DIR'):
        os.environ['CARTOPY_DATA_DIR'] = _c
        break
import numpy as np

SETS = (('Z', 'phase'), ('T', 'phase'), ('Z', 'group'), ('T', 'group'))
WAVE = {'Z': 'Rayleigh', 'T': 'Love'}

# Caption caveats for v3b iter-2 (band120); (component, type): [(pmin, pmax, text)]
CAVEATS = {
    ('Z', 'group'): [(32.0, 40.0, 'Z group 32-40 s: low confidence (residual RMS 21-34 s)')],
    ('T', 'group'): [(30.0, 32.0, 'T group ~30-32 s: low confidence')],
    ('Z', 'phase'): [(52.0, 60.0, 'Z phase 52-60 s: at the 50-cell floor (1.3-2.3k rays), '
                                   'smooth by construction')],
}
DROPPED = 'long-period group deliberately dropped (Z > 40 s, T > 32 s)'

ARRAY_RE = re.compile(r'^(W....|C[WESC]..|F...)$')
ARRAY_EXCLUDE = {'FORT', 'FITZ'}


def caveats_for(comp, typ, period):
    return [t for lo, hi, t in CAVEATS.get((comp, typ), []) if lo - 1e-6 <= period <= hi + 1e-6]


# ----------------------------------------------------------------------------- loading
def _orient(a, lat2d, lon2d):
    """Arrays [..., nlat, nlon] with latitude and longitude ascending (from lat2d/lon2d)."""
    lat2d, lon2d = np.asarray(lat2d, float), np.asarray(lon2d, float)
    tr = np.ptp(lat2d[:, 0]) == 0                 # latitude varies along axis 1
    if tr:
        lat2d, lon2d = lat2d.T, lon2d.T
    flip_lat = lat2d[0, 0] > lat2d[-1, 0]
    flip_lon = lon2d[0, 0] > lon2d[0, -1]

    def f(g):
        g = np.asarray(g)
        if tr:
            g = np.swapaxes(g, -1, -2)
        if flip_lat:
            g = g[..., ::-1, :]
        if flip_lon:
            g = g[..., :, ::-1]
        return g
    lat = np.sort(lat2d[:, 0])
    lon = np.sort(lon2d[0, :])
    return f, lat, lon


def set_from_path(path):
    """('Z'|'T', 'phase'|'group') from .../v3b/Z/phase_v_tomo/TransD.region.phase.npz."""
    p = os.path.abspath(path).replace('\\', '/')
    m = re.search(r'/([ZTR])/(group|phase)_v_tomo/', p)
    comp = m.group(1) if m else ('T' if '/T/' in p else 'Z')
    typ = m.group(2) if m else ('group' if 'group' in os.path.basename(p) else 'phase')
    return comp, typ


def load_tomo(path, comp=None, typ=None):
    """-> dict(per, v[nper,nlat,nlon], std, maski (bool), lat, lon, LAT, LON, comp, typ, path)."""
    d = np.load(path, allow_pickle=True)
    f, lat, lon = _orient(None, d['lat2d'], d['lon2d'])
    v = f(np.asarray(d['vall2'], float))
    std = f(np.asarray(d['std2'], float)) if 'std2' in d.files else np.full_like(v, np.nan)
    mi = (np.isfinite(f(np.asarray(d['maski'], float))) if 'maski' in d.files
          else np.ones_like(v, bool))
    per = np.atleast_1d(np.asarray(d['fk'], float)).ravel()
    if v.ndim == 2:
        v, std, mi = v[None], std[None], mi[None]
    if v.shape[0] != len(per):
        raise ValueError(f"{path}: vall2 has {v.shape[0]} periods, fk has {len(per)}")
    c, t = set_from_path(path)
    LON, LAT = np.meshgrid(lon, lat)
    return dict(per=per, v=v, std=std, maski=mi, lat=lat, lon=lon, LAT=LAT, LON=LON,
                comp=comp or c, typ=typ or t, path=path)


def find_sets(root, sets=None):
    """[(comp, typ, path)] under a v3b root: <root>/{Z,T}/{group,phase}_v_tomo/TransD.region.*.npz"""
    out = []
    for comp, typ in (sets or SETS):
        p = os.path.join(root, comp, f'{typ}_v_tomo', f'TransD.region.{typ}.npz')
        if os.path.isfile(p):
            out.append((comp, typ, p))
        else:
            print(f"  (missing {p})")
    return out


def parse_sets(tokens):
    """['Z.phase', 'T.group', 'all'] -> [(comp, typ)]"""
    if not tokens or 'all' in tokens:
        return list(SETS)
    out = []
    for t in tokens:
        c, ty = t.replace('/', '.').split('.')
        out.append((c.upper(), ty.lower()))
    return out


def pick_periods(per, want, tol=0.05):
    """Indices of the periods nearest to `want` (list of s, or None/'all' = all)."""
    if want is None or (len(want) == 1 and str(want[0]).lower() == 'all'):
        return list(range(len(per)))
    idx = []
    for w in map(float, want):
        i = int(np.argmin(np.abs(per - w)))
        if abs(per[i] - w) > tol * w:              # not in this set (e.g. 60 s group)
            print(f"   ({w:g} s not in this set; nearest {per[i]:g} s)")
            continue
        if i not in idx:
            idx.append(i)
    return idx


# ----------------------------------------------------------------------------- coverage
def density_mask(density, thr=10.0):
    """density >= thr, interior holes filled, largest connected region kept (no dilation)."""
    from scipy import ndimage
    m = np.asarray(density, float) >= thr
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


def _default_footprint():
    try:
        import config
        return config.FOOTPRINT_NPZ
    except Exception:
        return None


def load_coverage(lat, lon, fp_npz=None, key='density', thr=10.0):
    """Coverage mask on the tomography grid (lat, lon ascending) from the footprint npz.
    -> (mask [nlat, nlon], description)"""
    fp_npz = fp_npz or _default_footprint()
    if not fp_npz or not os.path.isfile(fp_npz):
        raise FileNotFoundError(f"footprint npz not found: {fp_npz} (--footprint)")
    d = np.load(fp_npz, allow_pickle=True)
    if key not in d.files:
        raise KeyError(f"{fp_npz} has no '{key}' (keys: {', '.join(d.files)}); the v3b mask "
                       f"needs 'density' - the 'mask' key is the old v2 mask")
    f, flat, flon = _orient(None, d['lat2d'], d['lon2d'])
    g = f(np.asarray(d[key], float))
    m = density_mask(g, thr) if key == 'density' else g.astype(bool)
    same = (len(flat) == len(lat) and len(flon) == len(lon)
            and np.allclose(flat, lat, atol=1e-3) and np.allclose(flon, lon, atol=1e-3))
    if not same:                                   # nearest footprint cell
        dla, dlo = np.median(np.diff(flat)), np.median(np.diff(flon))
        i = np.rint((lat - flat[0]) / dla).astype(int)
        j = np.rint((lon - flon[0]) / dlo).astype(int)
        ok_i = (i >= 0) & (i < len(flat)); ok_j = (j >= 0) & (j < len(flon))
        mm = np.zeros((len(lat), len(lon)), bool)
        mm[np.ix_(ok_i, ok_j)] = m[np.ix_(i[ok_i], j[ok_j])]
        m = mm
    desc = (f"{os.path.basename(fp_npz)} {key}>={thr:g}, holes filled, largest region"
            if key == 'density' else f"{os.path.basename(fp_npz)} {key}")
    return m, desc


def figure_masks(T, cov, std_max=None):
    """[nper, nlat, nlon] = coverage & std <= std_max (default: finite maski) & finite v."""
    if std_max is None:
        s = T['maski']
    else:
        with np.errstate(invalid='ignore'):
            s = T['std'] <= std_max
    return cov[None] & s & np.isfinite(T['v'])


# ----------------------------------------------------------------------------- stations
def load_stations(path=None, which='array'):
    """-> (names, lon, lat, is_array). 'name lat lon' lines (config.STATION_COLS)."""
    if path is None:
        try:
            import config
            path = config.STATIONS
        except Exception:
            path = None
    if not path or not os.path.isfile(path):
        print(f"  (stations not found: {path})")
        return np.array([]), np.array([]), np.array([]), np.array([], bool)
    names, lon, lat = [], [], []
    for line in open(path):
        p = line.replace(',', ' ').split()
        if not p or p[0].startswith('#'):
            continue
        try:
            la, lo = float(p[1]), float(p[2])
        except (ValueError, IndexError):
            continue
        names.append(p[0]); lat.append(la); lon.append(lo)
    names = np.array(names)
    code = np.array([n.split('.')[-1] for n in names])
    arr = np.array([bool(ARRAY_RE.match(c)) and c not in ARRAY_EXCLUDE for c in code], bool)
    return names, np.array(lon), np.array(lat), arr


# ----------------------------------------------------------------------------- projection
class Albers:
    """Spherical Albers equal-area, as the GSWA 2022 1:10M tectonic map (central meridian
    121E, standard parallels 17.5S and 31.5S). x, y in km."""

    def __init__(self, lon0=121.0, lat1=-17.5, lat2=-31.5, lat0=-25.0, R=6371.0):
        p1, p2, p0 = np.radians([lat1, lat2, lat0])
        self.n = 0.5 * (np.sin(p1) + np.sin(p2))
        self.C = np.cos(p1) ** 2 + 2 * self.n * np.sin(p1)
        self.R, self.lon0 = R, lon0
        self.rho0 = R * np.sqrt(self.C - 2 * self.n * np.sin(p0)) / self.n

    def __call__(self, lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        rho = self.R * np.sqrt(self.C - 2 * self.n * np.sin(np.radians(lat))) / self.n
        th = self.n * np.radians(lon - self.lon0)
        return rho * np.sin(th), self.rho0 - rho * np.cos(th)


def edges(c):
    c = np.asarray(c, float)
    m = 0.5 * (c[1:] + c[:-1])
    return np.r_[c[0] - (m[0] - c[0]), m, c[-1] + (c[-1] - m[-1])]


def _fill_nan(a):
    from scipy import ndimage
    bad = ~np.isfinite(a)
    if not bad.any() or bad.all():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def upsample(lat, lon, z, m, factor=4):
    """Bilinear upsampling of field z and mask m (as a float, kept where >= 0.5: the edge
    follows the mask to within half a cell, without the staircase). factor 1 = cells."""
    if factor <= 1:
        return lat, lon, z, m
    from scipy.interpolate import RegularGridInterpolator
    la = np.linspace(edges(lat)[0], edges(lat)[-1], len(lat) * factor + 1)
    lo = np.linspace(edges(lon)[0], edges(lon)[-1], len(lon) * factor + 1)
    la, lo = 0.5 * (la[1:] + la[:-1]), 0.5 * (lo[1:] + lo[:-1])
    LA, LO = np.meshgrid(np.clip(la, lat[0], lat[-1]), np.clip(lo, lon[0], lon[-1]), indexing='ij')
    pts = np.column_stack([LA.ravel(), LO.ravel()])
    zi = RegularGridInterpolator((lat, lon), _fill_nan(np.asarray(z, float)))(pts)
    mi = RegularGridInterpolator((lat, lon), m.astype(float))(pts)
    return la, lo, zi.reshape(LA.shape), mi.reshape(LA.shape) >= 0.5


# ----------------------------------------------------------------------------- map frame
def _ne_dirs():
    out = [os.environ.get('CARTOPY_DATA_DIR', '')]
    out += ['/data/25/yuan/mfiles/cartopy_data', '/mnt/data64/yuan/mfiles/cartopy_data',
            os.path.expanduser('~/.local/share/cartopy')]
    try:
        import cartopy
        out += [cartopy.config.get('pre_existing_data_dir', ''), cartopy.config.get('data_dir', '')]
    except Exception:
        pass
    return [d for d in dict.fromkeys(out) if d and os.path.isdir(d)]


def _ne_file(category, name, res=('50m', '110m', '10m')):
    """Path of a Natural Earth shapefile already on disk (never downloads), or None."""
    for d in _ne_dirs():
        for r in res:
            p = os.path.join(d, 'shapefiles', 'natural_earth', category, f'ne_{r}_{name}.shp')
            if os.path.isfile(p):
                return p
    return None


def _shp_reader(path):
    try:
        import cartopy.io.shapereader as shpreader
        return shpreader.Reader(path)
    except Exception:
        return None


def _natural_earth_states():
    """dict(states=[(name, [rings])], coast=[lines]) from Natural Earth files on disk.
    States (admin_1) when present; otherwise the coastline only (+ the 129E WA border is
    drawn by MapFrame). Nothing is downloaded (d204403 has no internet)."""
    out = dict(states=[], coast=[], src='none')
    try:
        p = _ne_file('cultural', 'admin_1_states_provinces') or \
            _ne_file('cultural', 'admin_1_states_provinces_lakes')
        rd = _shp_reader(p) if p else None
        if rd is not None:
            for rec in rd.records():
                if rec.attributes.get('admin', '') != 'Australia':
                    continue
                g = rec.geometry
                polys = list(g.geoms) if hasattr(g, 'geoms') else [g]
                out['states'].append((rec.attributes.get('name', ''),
                                      [np.asarray(q.exterior.coords) for q in polys]))
            out['src'] = os.path.basename(p)
            return out
        p = _ne_file('physical', 'coastline') or _ne_file('physical', 'land')
        rd = _shp_reader(p) if p else None
        if rd is not None:
            for g in rd.geometries():
                b = g.bounds
                if b[2] < 105 or b[0] > 160 or b[3] < -50 or b[1] > 0:
                    continue
                parts = list(g.geoms) if hasattr(g, 'geoms') else [g]
                for q in parts:
                    xy = np.asarray(q.exterior.coords if hasattr(q, 'exterior') else q.coords)
                    out['coast'].append(xy)
            out['src'] = os.path.basename(p) + ' (no admin_1 states on disk: 129E border drawn)'
            return out
        print(f"  (no Natural Earth files found under: {', '.join(_ne_dirs()) or 'nothing'})")
    except Exception as e:
        print(f"  (Natural Earth not readable: {e})")
    return out


_STATES = None


class MapFrame:
    """Albers map axes for a lon/lat box: white sea, other states pale grey, WA coast,
    graticule with outside labels. Use .xy(lon, lat) for anything drawn on it."""

    def __init__(self, ax, extent, grat=None, proj=None, states='#e8e8e8', grid_col='#858c8c'):
        global _STATES
        self.ax, self.extent = ax, extent
        self.xy = proj or Albers()
        lo0, lo1, la0, la1 = extent
        bl = np.r_[np.linspace(lo0, lo1, 60), np.full(60, lo1), np.linspace(lo1, lo0, 60), np.full(60, lo0)]
        bt = np.r_[np.full(60, la0), np.linspace(la0, la1, 60), np.full(60, la1), np.linspace(la1, la0, 60)]
        bx, by = self.xy(bl, bt)
        # axes box = the lon/lat box's projected bounds, trimmed to the inscribed part
        self.xlim = (bx.min(), bx.max()); self.ylim = (by.min(), by.max())
        ax.set_xlim(*self.xlim); ax.set_ylim(*self.ylim)
        ax.set_aspect('equal'); ax.set_facecolor('white')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)
        if _STATES is None:
            _STATES = _natural_earth_states()
        from matplotlib.patches import Polygon
        self.has_coast = bool(_STATES['states'] or _STATES['coast'])
        if not getattr(MapFrame, '_reported', False):
            print(f"  map base: {_STATES['src']}")
            MapFrame._reported = True
        for r in _STATES['coast']:                     # coastline only (no admin_1 on disk)
            x, y = self.xy(r[:, 0], r[:, 1])
            ax.plot(x, y, color='k', lw=0.6, zorder=6)
        if _STATES['coast'] and not _STATES['states']:  # WA-NT/SA border: the 129th meridian
            bl = np.linspace(-31.69, -14.87, 50)
            x, y = self.xy(np.full_like(bl, 129.0), bl)
            ax.plot(x, y, color='k', lw=0.6, zorder=6)
        for name, rings in _STATES['states']:
            wa = name == 'Western Australia'
            for r in rings:
                x, y = self.xy(r[:, 0], r[:, 1])
                ax.add_patch(Polygon(np.column_stack([x, y]), closed=True,
                                     fc='none' if wa else states, ec='k' if wa else '0.5',
                                     lw=0.7 if wa else 0.4, zorder=0.5 if not wa else 6))
        # graticule
        step = grat or (2.0 if (lo1 - lo0) < 14 else 5.0)
        glon = np.arange(np.ceil(lo0 / step) * step - step, lo1 + step, step)
        glat = np.arange(np.ceil(la0 / step) * step - step, la1 + step, step)
        tl, tL = np.linspace(la0 - 5, la1 + 5, 200), np.linspace(lo0 - 5, lo1 + 5, 200)
        for g in glon:
            x, y = self.xy(np.full_like(tl, g), tl)
            ax.plot(x, y, color=grid_col, lw=0.35, zorder=1)
            xb = np.interp(self.ylim[0], y, x) if y[0] < self.ylim[0] < y[-1] else None
            if xb is not None and self.xlim[0] < xb < self.xlim[1]:
                ax.annotate(f"{g:g}°E", (xb, self.ylim[0]), xytext=(0, -3), textcoords='offset points',
                            ha='center', va='top', fontsize=7, annotation_clip=False)
        for g in glat:
            x, y = self.xy(tL, np.full_like(tL, g))
            ax.plot(x, y, color=grid_col, lw=0.35, zorder=1)
            if x[0] < self.xlim[0] < x[-1]:
                yl = np.interp(self.xlim[0], x, y)
                if self.ylim[0] < yl < self.ylim[1]:
                    ax.annotate(f"{abs(g):g}°S", (self.xlim[0], yl), xytext=(-3, 0),
                                textcoords='offset points', ha='right', va='center', fontsize=7,
                                annotation_clip=False)

    def raster(self, lat, lon, z, m, cmap, norm, zorder=2):
        """Field on cell edges (lat, lon centres ascending), masked by m."""
        LO, LA = np.meshgrid(edges(lon), edges(lat))
        X, Y = self.xy(LO, LA)
        return self.ax.pcolormesh(X, Y, np.ma.masked_where(~m, z), cmap=cmap, norm=norm,
                                  shading='flat', zorder=zorder, rasterized=True)

    def veil(self, lat, lon, inside, colour='white', alpha=0.6, zorder=4):
        """Fade everything outside `inside` (bool grid on lat/lon centres) - e.g. the area
        outside the coverage mask; the sea stays as drawn."""
        from matplotlib.colors import ListedColormap
        LO, LA = np.meshgrid(edges(lon), edges(lat))
        X, Y = self.xy(LO, LA)
        return self.ax.pcolormesh(X, Y, np.ma.masked_where(inside, np.ones(inside.shape)),
                                  cmap=ListedColormap([colour]), alpha=alpha, shading='flat',
                                  zorder=zorder, rasterized=True)

    def outline(self, lat, lon, m, zorder=5, **kw):
        """Edge of a boolean grid mask (0.5 contour, padded so it closes)."""
        la = np.r_[2 * lat[0] - lat[1], lat, 2 * lat[-1] - lat[-2]]
        lo = np.r_[2 * lon[0] - lon[1], lon, 2 * lon[-1] - lon[-2]]
        LO, LA = np.meshgrid(lo, la)
        X, Y = self.xy(LO, LA)
        return self.ax.contour(X, Y, np.pad(m.astype(float), 1), levels=[0.5], zorder=zorder, **kw)

    def lines(self, rings, zorder=5, **kw):
        from matplotlib.collections import LineCollection
        lo0, lo1, la0, la1 = self.extent
        segs = []
        for r in rings:
            if (r[:, 0].max() < lo0 - 1 or r[:, 0].min() > lo1 + 1
                    or r[:, 1].max() < la0 - 1 or r[:, 1].min() > la1 + 1):
                continue
            x, y = self.xy(r[:, 0], r[:, 1])
            segs.append(np.column_stack([x, y]))
        self.ax.add_collection(LineCollection(segs, zorder=zorder, **kw))
        return len(segs)

    def inside(self, lon, lat):
        lo0, lo1, la0, la1 = self.extent
        return (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)


def mask_extent(lat, lon, m, margin=0.6, full=(112.0, 131.0, -36.0, -13.0)):
    """lon/lat box around the True cells of m (+margin), or the full grid box."""
    if m is None or not m.any():
        return list(full)
    ii, jj = np.where(m)
    return [lon[jj.min()] - margin, lon[jj.max()] + margin, lat[ii.min()] - margin, lat[ii.max()] + margin]


# ----------------------------------------------------------------------------- domains
_LABEL_POS = None

SHORT = {'wac': 'WAC', 'yilgarn': 'Yilgarn', 'pilbara': 'Pilbara', 'capricorn': 'Capricorn',
         'albany_fraser': 'Albany–Fraser', 'perth': 'Perth B.', 'canning': 'Canning B.',
         'officer': 'Officer B.', 'eucla': 'Eucla B.', 'kimberley': 'Kimberley B.'}


def label_positions():
    """{normalised GSWA label text: (lon, lat, angle)} from tectonic_map_2022_labels.csv"""
    global _LABEL_POS
    if _LABEL_POS is None:
        import csv
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'basemap',
                         'tectonic_map_2022_labels.csv')
        _LABEL_POS = {}
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                rows = csv.reader(l for l in f if l.strip() and not l.startswith('#'))
                for r in rows:
                    try:
                        key = r[0].replace('\\n', ' ').replace('–', '-').lower()
                        _LABEL_POS.setdefault(key, (float(r[1]), float(r[2]), float(r[3])))
                    except (ValueError, IndexError):
                        continue
    return _LABEL_POS


def domain_outline_rings(D, key):
    """Rings outlining a domain: the units named by its name: rules (the 10M outline
    records, so no internal unit boundaries); all its unit rings if none are named."""
    dom = D[key]
    named = {v for f, v in dom['units'] if f == 'name'}
    units = D.unit_rings(key)
    idx = [i for u, ii in units.items() if u in named for i in ii]
    if not idx:
        idx = [i for ii in units.values() for i in ii]
    return [D.rings[i] for i in idx if len(D.rings[i]) >= 3]


def domain_label_xy(D, key, grid=None):
    """(lon, lat, angle) for a domain label: the GSWA 2022 map position of its name,
    else the domain node nearest the domain's centroid on `grid` (LON, LAT, inside)."""
    pos = label_positions()
    lab = D[key]['label'].replace('–', '-').lower()
    if lab in pos:
        return pos[lab]
    if grid is not None:
        LON, LAT, inside = grid
        if inside.any():
            cx, cy = LON[inside].mean(), LAT[inside].mean()
            k = np.argmin((LON[inside] - cx) ** 2 + (LAT[inside] - cy) ** 2)
            return LON[inside][k], LAT[inside][k], 0.0
    return None


# ----------------------------------------------------------------------------- sites
# colours of the Yilgarn study-site map (terrane -> colour)
TERRANE_COLOURS = {'narryer': '#762a83', 'youanmi': '#ff7f0e', 'kalgoorlie': '#2ca02c',
                   'kurnalpi': '#9467bd', 'burtville': '#8c564b', 'south west': '#e377c2',
                   'capricorn': '#17becf', 'pinjarra': '#bcbd22', 'albany-fraser': '#1f77b4'}


def terrane_colour(t, i=0):
    import matplotlib
    c = TERRANE_COLOURS.get(str(t).replace('–', '-').strip().lower())
    return c or matplotlib.colormaps['tab10'](i % 10)


def load_sites(path):
    """Study sites: '# Name Terrane LAT LONG' rows (tab, comma or spaces; the terrane may
    contain spaces). -> list of dict(no, name, terrane, lat, lon) in file order."""
    out = []
    for line in open(path, encoding='utf-8'):
        s = line.strip()
        if not s or s.startswith('#') or s.lower().startswith(('name', 'no')):
            continue
        p = [x.strip() for x in (s.split('\t') if '\t' in s else s.split(',') if ',' in s else s.split())]
        p = [x for x in p if x]
        try:
            lat, lon = float(p[-2]), float(p[-1])
        except (ValueError, IndexError):
            continue
        if p[0].isdigit():
            no, name, terr = int(p[0]), p[1], ' '.join(p[2:-2])
        else:
            no, name, terr = len(out) + 1, p[0], ' '.join(p[1:-2])
        out.append(dict(no=no, name=name, terrane=terr or '-', lat=lat, lon=lon))
    if not out:
        raise ValueError(f"no sites read from {path}")
    return out


def sample_grid(lat, lon, g, plat, plon):
    """Bilinear value of grid g [nlat, nlon] at points (NaNs filled from the nearest cell)."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((lat, lon), _fill_nan(np.asarray(g, float)),
                                bounds_error=False, fill_value=np.nan)
    return f(np.column_stack([plat, plon]))


def nearest_cell(lat, lon, plat, plon):
    i = np.clip(np.rint((np.asarray(plat) - lat[0]) / (lat[1] - lat[0])).astype(int), 0, len(lat) - 1)
    j = np.clip(np.rint((np.asarray(plon) - lon[0]) / (lon[1] - lon[0])).astype(int), 0, len(lon) - 1)
    return i, j
