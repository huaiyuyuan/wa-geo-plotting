"""
footprint.py - WA Array coverage footprint for maps and regional statistics.

Two files, two roles (see the footprint README):
  wa_array_footprint.npz  'mask' on the 70x60 inversion grid -> SELECTION rule
                          (which model nodes / cells count)
  wa_array_outline.txt    smoothed lon/lat boundary, slightly outside the mask
                          -> DRAWING only (outline, veil, zoom)
This is coverage (many array paths cross the cell), not resolution; use the
posterior std for resolution.

  from footprint import Footprint
  fp = Footprint()                       # paths from config.py
  keep = fp.contains(lon, lat)           # bool per model node
  fp.veil(ax, transform); fp.draw(ax, transform); fp.zoom(ax)
"""
import os
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def _config_paths():
    try:
        import config
        return getattr(config, 'FOOTPRINT_NPZ', None), getattr(config, 'FOOTPRINT_OUTLINE', None)
    except Exception:
        return None, None


class Footprint:
    def __init__(self, npz=None, outline=None, key='mask'):
        c_npz, c_out = _config_paths()
        self.npz_path = npz or c_npz
        self.outline_path = outline or c_out
        if not self.npz_path or not os.path.isfile(self.npz_path):
            raise FileNotFoundError(f"footprint npz not found: {self.npz_path} "
                                    f"(set config.FOOTPRINT_NPZ or WA_FOOTPRINT_DIR)")
        d = np.load(self.npz_path, allow_pickle=True)
        mask = np.asarray(d[key]).astype(bool)
        lat2d, lon2d = np.asarray(d['lat2d'], float), np.asarray(d['lon2d'], float)
        if np.ptp(lat2d[:, 0]) == 0:          # latitude varies along axis 1: make rows = lat
            mask, lat2d, lon2d = mask.T, lat2d.T, lon2d.T
        lat, lon = lat2d[:, 0], lon2d[0, :]
        if lat[0] > lat[-1]:
            lat, mask = lat[::-1], mask[::-1]
        if lon[0] > lon[-1]:
            lon, mask = lon[::-1], mask[:, ::-1]
        self.lat, self.lon, self.mask = lat, lon, mask
        self.dlat = np.median(np.diff(lat))
        self.dlon = np.median(np.diff(lon))
        self.key = key
        self.olon = self.olat = None
        if self.outline_path and os.path.isfile(self.outline_path):
            xy = np.loadtxt(self.outline_path, comments='#', usecols=(0, 1))
            self.olon, self.olat = xy[:, 0], xy[:, 1]

    # ---------------------------------------------------------------- selection
    def contains(self, lon, lat):
        """True for points whose nearest mask cell is inside the footprint
        (points more than half a cell beyond the grid are outside)."""
        lon = np.atleast_1d(np.asarray(lon, float))
        lat = np.atleast_1d(np.asarray(lat, float))
        i = np.clip(np.rint((lat - self.lat[0]) / self.dlat).astype(int), 0, len(self.lat) - 1)
        j = np.clip(np.rint((lon - self.lon[0]) / self.dlon).astype(int), 0, len(self.lon) - 1)
        on_grid = ((lat >= self.lat[0] - self.dlat / 2) & (lat <= self.lat[-1] + self.dlat / 2) &
                   (lon >= self.lon[0] - self.dlon / 2) & (lon <= self.lon[-1] + self.dlon / 2))
        return on_grid & self.mask[i, j]

    def describe(self, keep):
        return (f"footprint ({self.key}, {os.path.basename(self.npz_path)}): "
                f"{int(np.sum(keep))} of {len(keep)} nodes inside")

    # ---------------------------------------------------------------- drawing
    def _outline_xy(self):
        if self.olon is not None:
            return self.olon, self.olat
        raise FileNotFoundError(f"footprint outline not found: {self.outline_path}")

    def bbox(self, margin=0.6):
        x, y = self._outline_xy()
        return [x.min() - margin, x.max() + margin, y.min() - margin, y.max() + margin]

    def draw(self, ax, transform=None, color='k', lw=1.3, zorder=9):
        x, y = self._outline_xy()
        kw = {'transform': transform} if transform is not None else {}
        ax.plot(x, y, '-', color=color, lw=lw, zorder=zorder, **kw)

    def veil(self, ax, transform=None, color='white', alpha=0.75, zorder=7):
        """Fill everything outside the outline (a big box with the outline as a hole)."""
        x, y = self._outline_xy()
        ring = np.column_stack([x, y])
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[:1]])
        area = 0.5 * np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
        if area > 0:                          # hole must run opposite to the box
            ring = ring[::-1]
        box = np.array([[60, -80], [180, -80], [180, 20], [60, 20], [60, -80]], float)
        verts = np.vstack([box, ring])
        codes = ([Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY] +
                 [Path.MOVETO] + [Path.LINETO] * (len(ring) - 2) + [Path.CLOSEPOLY])
        kw = {'transform': transform} if transform is not None else {}
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor='none',
                               alpha=alpha, zorder=zorder, **kw))

    def zoom(self, ax, margin=0.6):
        e = self.bbox(margin)
        if hasattr(ax, 'set_extent'):
            import cartopy.crs as ccrs
            ax.set_extent(e, crs=ccrs.PlateCarree())
        else:
            ax.set_xlim(e[:2]); ax.set_ylim(e[2:])

    def focus(self, ax, transform=None, veil_alpha=0.75, zoom=True, veil_zorder=7):
        """Veil outside + outline + zoom, in one call (safe if the outline is missing)."""
        if self.olon is None:
            print(f"  (footprint outline not found: {self.outline_path} - "
                  f"nodes still selected by the mask, no outline drawn)")
            return
        if veil_alpha:
            self.veil(ax, transform, alpha=veil_alpha, zorder=veil_zorder)
        self.draw(ax, transform)
        if zoom:
            self.zoom(ax)


def inset_colorbar(ax, im, label='', extend='neither', box=(0.80, 0.50, 0.185, 0.47)):
    """Colourbar INSIDE the map at the north-east corner (outside the footprint there),
    on a white backing; ticks and label face inward so nothing leaves the frame.
    box = (x0, y0, w, h) of the backing in axes fractions."""
    from matplotlib.patches import Rectangle
    x0, y0, w, h = box
    ax.add_patch(Rectangle((x0, y0), w, h, transform=ax.transAxes, facecolor='white',
                           edgecolor='0.6', lw=0.5, alpha=0.9, zorder=19))
    cax = ax.inset_axes([x0 + w - 0.055, y0 + 0.06, 0.035, h - 0.12])
    cax.set_zorder(20)                        # above veil (7) and coastlines (10)
    cb = ax.figure.colorbar(im, cax=cax, extend=extend)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.ax.tick_params(labelsize=7, length=2)
    if label:
        cb.set_label(label, fontsize=8)
    return cb


def add_args(ap):
    """Standard CLI flags for map scripts."""
    ap.add_argument('--footprint', action='store_true',
                    help='focus on the WA Array coverage footprint: use only nodes inside '
                         'the mask, veil outside the outline, zoom to it')
    ap.add_argument('--footprint-npz', default=None, help='override config.FOOTPRINT_NPZ')
    ap.add_argument('--footprint-outline', default=None, help='override config.FOOTPRINT_OUTLINE')
    ap.add_argument('--footprint-veil', type=float, default=0.75,
                    help='veil opacity outside the footprint (0 = outline only)')
    ap.add_argument('--no-footprint-zoom', action='store_true',
                    help='keep the full map extent')


def from_args(args):
    if not getattr(args, 'footprint', False):
        return None
    fp = Footprint(args.footprint_npz, args.footprint_outline)
    return fp
