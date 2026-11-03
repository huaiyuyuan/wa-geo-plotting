"""Surface-geology strips for cross-sections.

For each profile this module
  1. samples surface lithology (or tectonic domains) at points along the
     profile with a point-in-polygon test, and
  2. finds where MajorCrustalBoundaries polylines cross the profile,
then draws a thin colour strip above the top Vs panel and dashed vertical
lines through every panel at the boundary crossings.

It only needs numpy + matplotlib and reads the npz files written by extract/:

  polygons  (make_litho.py, make_tectonic_basemap.py)
      rings    object array of (N, 2) lon/lat arrays
      colors   per-ring colour (any matplotlib colour spec)
      parents  per-ring class name (LITHOLOGY, or PARENTNAME for tectonics)
  boundaries (make_boundaries.py)
      lines    object array of (N, 2) lon/lat polylines
      names    optional per-line name

All distances are in km along the profile, matching the x axis of the
cross-section panels.

GSWA data (c) Geological Survey of Western Australia, CC-BY-4.0.
"""
from __future__ import annotations

import os

from dataclasses import dataclass, field

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.path import Path
from matplotlib.transforms import blended_transform_factory

R_EARTH = 6371.0


# --------------------------------------------------------------------------
# Profile geometry
# --------------------------------------------------------------------------
def _xyz(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def _lonlat(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arctan2(z, np.hypot(x, y)))


def great_circle_points(lon1, lat1, lon2, lat2, step_km=1.0, n=None):
    """Evenly spaced points on the great circle; returns lon, lat, dist_km."""
    p1, p2 = _xyz(lon1, lat1), _xyz(lon2, lat2)
    omega = float(np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0)))
    total = omega * R_EARTH
    if n is None:
        n = max(2, int(np.ceil(total / step_km)) + 1)
    f = np.linspace(0.0, 1.0, n)
    if omega < 1e-12:
        pts = np.repeat(p1[None], n, axis=0)
    else:
        pts = (np.sin((1 - f) * omega)[:, None] * p1
               + np.sin(f * omega)[:, None] * p2) / np.sin(omega)
    lon, lat = _lonlat(pts)
    return lon, lat, f * total


def along_track_km(lon, lat):
    """Cumulative great-circle distance along a polyline (km)."""
    p = _xyz(np.asarray(lon, float), np.asarray(lat, float))
    seg = np.arccos(np.clip(np.sum(p[1:] * p[:-1], axis=1), -1.0, 1.0)) * R_EARTH
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_profile(lon, lat, dist, step_km=0.5):
    """Densify a profile to a fixed spacing for sampling the strip.

    The cross-section's own points are often several km apart, which is too
    coarse for 1:500k lithology; this interpolates along the existing
    polyline so the strip x axis stays identical to the panels'.
    """
    dist = np.asarray(dist, float)
    n = max(2, int(np.ceil((dist[-1] - dist[0]) / step_km)) + 1)
    d = np.linspace(dist[0], dist[-1], n)
    return np.interp(d, dist, lon), np.interp(d, dist, lat), d


def _local_xy(lon, lat, lon0, lat0):
    """Azimuthal-equidistant projection (km) about (lon0, lat0)."""
    lam = np.radians(np.asarray(lon, float) - lon0)
    phi, phi0 = np.radians(np.asarray(lat, float)), np.radians(lat0)
    cosc = np.clip(np.sin(phi0) * np.sin(phi)
                   + np.cos(phi0) * np.cos(phi) * np.cos(lam), -1.0, 1.0)
    c = np.arccos(cosc)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(c < 1e-12, 1.0, c / np.sin(c))
    x = R_EARTH * k * np.cos(phi) * np.sin(lam)
    y = R_EARTH * k * (np.cos(phi0) * np.sin(phi)
                       - np.sin(phi0) * np.cos(phi) * np.cos(lam))
    return np.column_stack([x, y])


# --------------------------------------------------------------------------
# Polygon sampling (lithology / tectonic domains)
# --------------------------------------------------------------------------
class PolygonIndex:
    """Point-in-polygon lookup over the rings of a litho/tectonic npz.

    Rings in the npz are not tagged as outer or hole, so containment is
    resolved per class by parity: a point is inside class C when it falls
    inside an odd number of C's rings (a hole stored as a ring of its parent
    cancels the outer ring). Among the classes that pass, the one whose
    smallest containing ring is smallest wins, so inliers beat the polygon
    they sit in.
    """

    def __init__(self, rings, colors, parents):
        self.rings = [np.asarray(r, float)[:, :2] for r in rings]
        self.colors = np.array([mcolors.to_hex(mcolors.to_rgba(c), keep_alpha=True)
                                for c in colors])
        self.parents = np.asarray(parents).astype(str)
        b = np.array([[r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max()]
                      for r in self.rings])
        self.xmin, self.xmax, self.ymin, self.ymax = b.T
        self.area = np.array([0.5 * abs(np.dot(r[:, 0], np.roll(r[:, 1], 1))
                                        - np.dot(r[:, 1], np.roll(r[:, 0], 1)))
                              for r in self.rings])
        self._paths: dict[int, Path] = {}

    @classmethod
    def from_npz(cls, path, class_key="parents", colours="map2022"):
        """Load a litho/tectonic npz.

        class_key="parents" uses the stored grouping and its colours.
        Any other per-ring key (e.g. "names" = TECTNAME) becomes the class, coloured by
        colours = "map2022"  : GSWA Simplified Tectonic Map 1:10M (2022) colours, by unit
                               name, then parent; then TECTCOLOUR; then the parent-domain
                               colour (as on the index map)
                  "tectcolour": TECTCOLOUR from the shapefile, then the palette
                  "palette"  : pastel palette only
        Falls back to "parents" with a warning if the key is missing.
        """
        d = np.load(path, allow_pickle=True)
        if class_key != "parents" and class_key not in d.files:
            print(f"  ({path}: no '{class_key}' array, using 'parents'; "
                  f"re-run the extract script to add it)")
            class_key = "parents"
        if class_key == "parents":
            return cls(d["rings"], d["colors"], d["parents"])
        names = np.asarray(d[class_key]).astype(str)
        cols = palette_for(names).astype(object)
        src = np.array(["palette"] * len(names), dtype=object)
        if colours == "map2022" and "colors" in d.files:     # last resort as on the index map:
            cols, src[:] = d["colors"].astype(object), "domain"   # the parent-domain colour
        if colours in ("map2022", "tectcolour") and "tect_colors" in d.files:
            tc = np.asarray(d["tect_colors"]).astype(str)
            have = tc != ""
            cols[have], src[have] = tc[have], "TECTCOLOUR"
        if colours == "map2022":
            table = map2022_colours()
            parents = np.asarray(d["parents"]).astype(str) if "parents" in d.files else names
            for k, (n, par) in enumerate(zip(names, parents)):
                for key, tag in ((n, "map2022"), (par, "map2022 (parent)")):
                    c = table.get(key.strip().lower())
                    if c:
                        cols[k], src[k] = c, tag
                        break
        from collections import Counter
        keep = np.array([c != "none" for c in cols])       # 'none': concealed, not on the map
        src[~keep] = "hidden"
        cnt = Counter(src)
        print(f"  ({os.path.basename(str(path))}: domain colours " +
              ", ".join(f"{v} {k}" for k, v in cnt.most_common()) + " rings)")
        rings = [r for r, k in zip(d["rings"], keep) if k]
        return cls(rings, cols[keep].astype(str), names[keep])

    def _path(self, k):
        p = self._paths.get(k)
        if p is None:
            p = self._paths[k] = Path(self.rings[k])
        return p

    def sample(self, lon, lat):
        """Ring index chosen for each point, -1 where no polygon contains it."""
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        pts = np.column_stack([lon, lat])
        cand = np.flatnonzero((self.xmax >= lon.min()) & (self.xmin <= lon.max())
                              & (self.ymax >= lat.min()) & (self.ymin <= lat.max()))
        hits: list[list[int]] = [[] for _ in range(lon.size)]
        for k in cand:
            m = ((lon >= self.xmin[k]) & (lon <= self.xmax[k])
                 & (lat >= self.ymin[k]) & (lat <= self.ymax[k]))
            if not m.any():
                continue
            ii = np.flatnonzero(m)
            inside = self._path(k).contains_points(pts[ii])
            for i in ii[inside]:
                hits[i].append(k)

        out = np.full(lon.size, -1, dtype=int)
        for i, ks in enumerate(hits):
            if not ks:
                continue
            if len(ks) == 1:
                out[i] = ks[0]
                continue
            by_class: dict[str, list[int]] = {}
            for k in ks:
                by_class.setdefault(self.parents[k], []).append(k)
            best, best_area = -1, np.inf
            for members in by_class.values():
                if len(members) % 2 == 0:  # inside a hole of this class
                    continue
                k = min(members, key=lambda j: self.area[j])
                if self.area[k] < best_area:
                    best, best_area = k, self.area[k]
            out[i] = best
        return out


_PALETTE = ("#c6dbef", "#fdd0a2", "#c7e9c0", "#dadaeb", "#fcbba1", "#d9d9d9",
            "#fee391", "#9ecae1", "#a1d99b", "#bcbddc", "#fdae6b", "#ccebc5",
            "#f2f0f7", "#ffffcc", "#b3cde3", "#decbe4")


_MAP2022 = None


def map2022_colours():
    """{lower-case unit name: '#rrggbb'} from basemap/tectonic_map_2022_colours.csv."""
    global _MAP2022
    if _MAP2022 is None:
        import csv
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "basemap", "tectonic_map_2022_colours.csv")
        _MAP2022 = {}
        if os.path.isfile(path):
            with open(path) as f:
                rows = csv.reader(l for l in f if not l.startswith("#"))
                next(rows, None)
                for r in rows:
                    if len(r) == 2 and (r[1].startswith("#") or r[1].strip() == "none"):
                        _MAP2022[r[0].strip().lower()] = r[1].strip()
        else:
            print(f"  (no {path}: map2022 domain colours unavailable)")
    return _MAP2022


def palette_for(names):
    """Stable pastel colour per unique name (same name -> same colour everywhere)."""
    import zlib
    return np.array([_PALETTE[zlib.crc32(n.encode()) % len(_PALETTE)] for n in names])


@dataclass
class Run:
    d0: float
    d1: float
    name: str
    color: str


def runs_from_samples(dist, idx, index: PolygonIndex, none_color="#ffffff00"):
    """Collapse per-point samples into contiguous same-class runs.

    Run edges sit halfway between samples, so the runs tile the profile
    exactly from dist[0] to dist[-1].
    """
    dist = np.asarray(dist, float)
    names = np.where(idx >= 0, index.parents[np.maximum(idx, 0)], "")
    cols = np.where(idx >= 0, index.colors[np.maximum(idx, 0)], none_color)
    edges = np.concatenate([[dist[0]], 0.5 * (dist[1:] + dist[:-1]), [dist[-1]]])
    runs, start = [], 0
    for i in range(1, len(dist) + 1):
        if i == len(dist) or names[i] != names[start] or cols[i] != cols[start]:
            runs.append(Run(edges[start], edges[i], names[start], cols[start]))
            start = i
    return runs


# --------------------------------------------------------------------------
# Boundary crossings
# --------------------------------------------------------------------------
@dataclass
class Crossing:
    dist: float
    names: list = field(default_factory=list)
    scale: str = ""          # 'lithospheric' wins when merged crossings differ


class BoundarySet:
    """Crustal-boundary polylines with a profile-intersection test."""

    def __init__(self, lines, names=None, scale=None):
        self.lines = [np.asarray(l, float)[:, :2] for l in lines]
        self.names = (np.asarray(names).astype(str) if names is not None
                      else np.array([""] * len(self.lines)))
        self.scale = (np.char.lower(np.asarray(scale).astype(str)) if scale is not None
                      else np.array([""] * len(self.lines)))
        b = np.array([[l[:, 0].min(), l[:, 0].max(), l[:, 1].min(), l[:, 1].max()]
                      for l in self.lines])
        self.xmin, self.xmax, self.ymin, self.ymax = b.T

    @classmethod
    def from_npz(cls, path):
        d = np.load(path, allow_pickle=True)
        names = d["names"] if "names" in d.files else None
        if names is None:
            print(f"  ({path}: no 'names' array; re-run extract/make_boundaries.py "
                  f"to label crossings)")
        return cls(d["lines"], names, d["scale"] if "scale" in d.files else None)

    def crossings(self, lon, lat, dist, dedupe_km=3.0, pad_deg=0.2):
        """Where boundary polylines cross the profile.

        Intersections are found in an azimuthal-equidistant frame centred on
        the profile, then mapped back to along-profile distance using the
        profile's own dist array. Crossings closer than dedupe_km are merged,
        because shared terrane edges are often digitised more than once.
        """
        lon, lat, dist = (np.asarray(a, float) for a in (lon, lat, dist))
        mid = len(lon) // 2
        lon0, lat0 = lon[mid], lat[mid]
        P = _local_xy(lon, lat, lon0, lat0)
        p, r = P[:-1], P[1:] - P[:-1]                     # (M, 2)
        cand = np.flatnonzero((self.xmax >= lon.min() - pad_deg)
                              & (self.xmin <= lon.max() + pad_deg)
                              & (self.ymax >= lat.min() - pad_deg)
                              & (self.ymin <= lat.max() + pad_deg))
        found = []
        bx0, by0 = P.min(axis=0) - 1.0
        bx1, by1 = P.max(axis=0) + 1.0
        for k in cand:
            Q = _local_xy(self.lines[k][:, 0], self.lines[k][:, 1], lon0, lat0)
            # only boundary segments whose box touches the profile's box (km frame)
            a0, a1 = Q[:-1], Q[1:]
            near = ((np.maximum(a0[:, 0], a1[:, 0]) >= bx0) & (np.minimum(a0[:, 0], a1[:, 0]) <= bx1)
                    & (np.maximum(a0[:, 1], a1[:, 1]) >= by0) & (np.minimum(a0[:, 1], a1[:, 1]) <= by1))
            if not near.any():
                continue
            q, s = a0[near], (a1 - a0)[near]              # (K, 2)
            denom = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
            qp = q[None, :, :] - p[:, None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / denom
                u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / denom
            ok = (np.abs(denom) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
            for i, j in zip(*np.nonzero(ok)):
                found.append((dist[i] + t[i, j] * (dist[i + 1] - dist[i]),
                              self.names[k], self.scale[k]))
        found.sort()
        merged: list[Crossing] = []
        for d, name, sc in found:
            if merged and d - merged[-1].dist < dedupe_km:
                if name and name not in merged[-1].names:
                    merged[-1].names.append(name)
                if sc == "lithospheric":
                    merged[-1].scale = sc
                continue
            merged.append(Crossing(d, [name] if name else [], sc))
        return merged


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def strip_axes_above(ax, height_in=0.14, gap_in=0.03, level=0, lift_in=0.0):
    """Thin axes above `ax` sharing its x axis; level=1 stacks above level 0.

    Uses inset_axes in axes-fraction coordinates so it works with any layout
    the cross-section script already uses (gridspec, per-panel colorbars,
    stacked sections), without shrinking the Vs panel. lift_in adds a fixed extra
    gap below this strip (in inches) — used to open room for the craton bar.
    """
    fig = ax.figure
    ax.apply_aspect()  # settle fixed-aspect panels (--stack-ve) before sizing
    h_ax_in = ax.get_position().height * fig.get_figheight()
    h, g = height_in / h_ax_in, gap_in / h_ax_in
    sax = ax.inset_axes([0.0, 1.0 + g + level * (h + g) + lift_in / h_ax_in, 1.0, h], sharex=ax)
    sax.set_yticks([])
    sax.set_ylim(0, 1)
    sax.tick_params(axis="x", bottom=False, labelbottom=False)
    for sp in sax.spines.values():
        sp.set_linewidth(0.6)
    return sax


def _move_titles(src, dst, pad=2.0):
    """Move left/centre/right titles from src to dst, keeping their font size."""
    for loc, artist in (("left", src._left_title), ("center", src.title),
                        ("right", src._right_title)):
        t = artist.get_text()
        if t:
            fs = artist.get_fontsize()
            src.set_title("", loc=loc)
            dst.set_title(t, loc=loc, fontsize=fs, pad=pad)


_SHORTEN = (" Superterrane", " Terrane", " Province", " Domain", " Zone")


def _shorten(name):
    for suf in _SHORTEN:
        name = name.replace(suf, "")
    return name


def annotate_runs(sax, runs, fontsize=6, min_km=0.0):
    """Write each run's name inside it when the text fits (tries a shortened
    form without 'Terrane'/'Province'/... if the full name does not)."""
    fig = sax.figure
    w_in = sax.get_position().width * fig.get_figwidth()
    span = (runs[-1].d1 - runs[0].d0) if runs else 1.0
    tr = blended_transform_factory(sax.transData, sax.transAxes)
    for rn in runs:
        if not rn.name or rn.d1 - rn.d0 < min_km:
            continue
        run_in = (rn.d1 - rn.d0) / span * w_in
        for txt in (rn.name, _shorten(rn.name)):
            if len(txt) * fontsize * 0.55 / 72 < run_in * 0.92:
                sax.text(0.5 * (rn.d0 + rn.d1), 0.5, txt, transform=tr, ha="center",
                         va="center", fontsize=fontsize, clip_on=True, zorder=8,
                         bbox=dict(fc=rn.color, ec="none", pad=0.6))
                break


def draw_strip(sax, runs, label=None, label_fontsize=7):
    for rn in runs:
        if rn.name:
            sax.axvspan(rn.d0, rn.d1, ymin=0, ymax=1, color=rn.color, lw=0)
    if label:
        sax.text(-0.005, 0.5, label, transform=sax.transAxes, ha="right",
                 va="center", fontsize=label_fontsize)


def _label_groups(crossings, km_per_in, min_sep_in=0.35):
    """Group named crossings closer than min_sep_in (on paper) so each group gets
    ONE label at its first crossing, listing each distinct name once."""
    groups = []
    for c in crossings:
        if not c.names:
            continue
        if groups and (c.dist - groups[-1][0]) / km_per_in < min_sep_in:
            for n in c.names:
                if n not in groups[-1][1]:
                    groups[-1][1].append(n)
        else:
            groups.append([c.dist, list(c.names)])
    return groups


ARROW = {"lithospheric": dict(lw=1.6, head=9, length=17),
         "crustal": dict(lw=0.9, head=6, length=12)}


def draw_boundaries(axes, crossings, strip_axes=(), color="k", lw=0.8,
                    ls=(0, (4, 3)), alpha=0.75, marker_size=5, label_names=False,
                    name_fontsize=6, span_km=None, min_sep_in=0.35, panel_lines=False):
    """Each major crustal boundary = a black down-arrow standing on the top strip
    (lithospheric-scale thick, crustal-scale thinner) and a thin tick through the
    strips. panel_lines=True also draws dashed lines through the model panels."""
    if panel_lines:
        for ax in axes:
            for c in crossings:
                ax.axvline(c.dist, color=color, lw=lw, ls=ls, alpha=alpha, zorder=6)
    for sax in strip_axes:
        for c in crossings:
            sax.axvline(c.dist, color=color, lw=0.6, zorder=6)
    if not strip_axes:
        return
    top = strip_axes[-1]
    tr = blended_transform_factory(top.transData, top.transAxes)
    tallest = 0
    for c in crossings:
        st = ARROW["crustal" if c.scale == "crustal" else "lithospheric"]
        top.annotate("", xy=(c.dist, 1.0), xycoords=tr, xytext=(0, st["length"]),
                     textcoords="offset points", annotation_clip=False, zorder=7,
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=st["lw"],
                                     mutation_scale=st["head"], shrinkA=0, shrinkB=0))
        tallest = max(tallest, st["length"])
    if label_names:
        w_in = top.get_position().width * top.figure.get_figwidth()
        if span_km is None:
            span_km = abs(np.diff(top.get_xlim())[0])
        for d0, names in _label_groups(crossings, span_km / w_in, min_sep_in):
            top.annotate(" / ".join(names), (d0, 1.0), xycoords=tr,
                         xytext=(1, tallest + 2), textcoords="offset points",
                         rotation=35, ha="left", va="bottom",
                         fontsize=name_fontsize, annotation_clip=False)


def legend_handles(all_runs, min_km=0.0, max_items=None):
    """Patches for the classes seen along the profiles, longest first."""
    total: dict[tuple, float] = {}
    for runs in all_runs:
        for rn in runs:
            if rn.name:
                key = (rn.name, rn.color)
                total[key] = total.get(key, 0.0) + (rn.d1 - rn.d0)
    items = sorted((v, k) for k, v in total.items() if v >= min_km)[::-1]
    if max_items:
        items = items[:max_items]
    return [Patch(facecolor=c, edgecolor="none", label=n) for _, (n, c) in items]


# --------------------------------------------------------------------------
# Greenstone + mineral-deposit overlay track (one combined strip)
# --------------------------------------------------------------------------
def greenstone_index_from_npz(path, rocktypes=("greenstone",), colour="#a1d99b"):
    """PolygonIndex of just the greenstone (or given rock-type) polygons from a
    yilgarn_units.npz, all one class so runs_from_samples fills a single band."""
    d = np.load(path, allow_pickle=True)
    rock = np.asarray(d["rocktype"]).astype(str)
    keep = np.isin(rock, list(rocktypes))
    rings = [r for r, k in zip(d["rings"], keep) if k]
    n = len(rings)
    return PolygonIndex(rings, [colour] * n, ["Greenstone"] * n)


def _stitch_rings(segs, tol=0.25):
    """Chain open boundary arcs end-to-end (nearest endpoint within tol deg) into rings, so a
    craton outline stored as separate arcs becomes closed polygons for a point-in-polygon test.
    Each returned ring is closed (first point repeated)."""
    segs = [np.asarray(s, float)[:, :2] for s in segs if len(s) >= 2]
    used = [False] * len(segs)
    rings = []
    for i in range(len(segs)):
        if used[i]:
            continue
        chain = [segs[i]]; used[i] = True
        while True:
            tail = chain[-1][-1]; best, rev, dmin = None, False, tol
            for j, s in enumerate(segs):
                if used[j]:
                    continue
                for r, pt in ((False, s[0]), (True, s[-1])):
                    d = float(np.hypot(*(pt - tail)))
                    if d < dmin:
                        dmin, best, rev = d, j, r
            if best is None:
                break
            used[best] = True
            chain.append(segs[best][::-1] if rev else segs[best])
        ring = np.vstack(chain)
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[0]])
        rings.append(ring)
    return rings


def craton_index_from_outline(path, colour="red"):
    """PolygonIndex from a craton outline text file (lon lat per line; blank line or a
    %/# comment separates arcs). The arcs are stitched into closed rings first (the file
    stores the Yilgarn/Pilbara boundary as open arcs, not closed polygons), all one 'Craton'
    class, so the in-craton test along a profile is continuous rather than per-arc."""
    segs, cur = [], []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith(("%", "#")):
                if len(cur) >= 2:
                    segs.append(np.asarray(cur, float))
                cur = []
                continue
            p = s.split()
            if len(p) >= 2:
                try:
                    cur.append((float(p[0]), float(p[1])))
                except ValueError:
                    pass
    if len(cur) >= 2:
        segs.append(np.asarray(cur, float))
    rings = _stitch_rings(segs)
    print(f"  craton outline: {len(segs)} arcs -> {len(rings)} closed ring(s)")
    return PolygonIndex(rings, [colour] * len(rings), ["Craton"] * len(rings))


def draw_craton_bar(sax, runs, colour="0.45", y0=1.02, y1=1.30):
    """Grey bar in the gap just ABOVE the domain-bar strip `sax` (outside the domain
    colours, no extra row) spanning each in-craton run. y0/y1 are domain-bar axes
    fraction (>1 = above its top edge); clip_on=False lets it draw into the gap."""
    from matplotlib.patches import Rectangle
    tr = blended_transform_factory(sax.transData, sax.transAxes)
    for rn in runs:
        if rn.name:                # inside the craton
            sax.add_patch(Rectangle((rn.d0, y0), rn.d1 - rn.d0, y1 - y0, transform=tr,
                                     facecolor=colour, edgecolor="none", zorder=7,
                                     clip_on=False))


def sample_vpvs_idw(stn_lon, stn_lat, stn_val, qlon, qlat, max_dist_deg=0.8, k=8, power=2.0):
    """Inverse-distance-weighted Vp/Vs (or any per-station value) at each profile point
    from the H-K stations. Blank (NaN) where the nearest station is farther than
    max_dist_deg, so off-footprint gaps stay empty like the Vp/Vs map. Distances are in a
    local azimuthal-equidistant km frame about the profile midpoint."""
    stn_lon = np.asarray(stn_lon, float); stn_lat = np.asarray(stn_lat, float)
    stn_val = np.asarray(stn_val, float)
    qlon = np.asarray(qlon, float); qlat = np.asarray(qlat, float)
    out = np.full(qlon.size, np.nan)
    if stn_lon.size == 0 or qlon.size == 0:
        return out
    mid = qlon.size // 2
    S = _local_xy(stn_lon, stn_lat, qlon[mid], qlat[mid])     # (m, 2) km
    Q = _local_xy(qlon, qlat, qlon[mid], qlat[mid])           # (n, 2) km
    maxkm = max_dist_deg * 111.195
    kk = int(min(k, stn_lon.size))
    D = np.hypot(Q[:, 0][:, None] - S[:, 0][None, :],
                 Q[:, 1][:, None] - S[:, 1][None, :])         # (n, m) km
    idx = np.argsort(D, axis=1)[:, :kk]                       # k nearest stations
    dn = np.take_along_axis(D, idx, axis=1)
    vn = stn_val[idx]
    within = dn <= maxkm
    w = np.where(within, 1.0 / np.maximum(dn, 1e-6) ** power, 0.0)
    wsum = w.sum(1)
    out = np.where(wsum > 0, (w * vn).sum(1) / np.maximum(wsum, 1e-12), np.nan)
    out[dn[:, 0] > maxkm] = np.nan                            # nearest station too far -> blank
    return out


def draw_vpvs_strip(sax, dist, val, clim=(1.68, 1.86), cmap=None, label=None,
                    label_fontsize=7):
    """Colour-coded Vp/Vs bar along the profile (like the domain / Au-greenstone bars, but
    the colour encodes the value). Values outside clim saturate at the ends; NaN cells are
    transparent. `cmap` should be the same 'rgb' map used for the Vp/Vs maps/panel."""
    dist = np.asarray(dist, float); val = np.asarray(val, float)
    if cmap is None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'rgb', [(0.78, 0, 0), (0.90, 0.55, 0), (0.1, 0.65, 0.2),
                    (0, 0.45, 0.80), (0, 0, 0.6)], N=256)
    try:
        cmap = cmap.copy(); cmap.set_bad(alpha=0.0)
    except Exception:
        pass
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])      # imshow clips -> saturated ends
    img = np.ma.masked_invalid(val[None, :])
    sax.imshow(img, aspect="auto", extent=[dist[0], dist[-1], 0.0, 1.0],
               origin="lower", cmap=cmap, norm=norm, interpolation="nearest", zorder=2)
    sax.set_ylim(0, 1)
    if label:
        sax.text(-0.005, 0.5, label, transform=sax.transAxes, ha="right",
                 va="center", fontsize=label_fontsize)


def draw_fault_arrow(top_sax, strip_axes, dc, label=None, color="red", lw=2.6, head=13,
                     length=20.0, name_fontsize=7):
    """Fault marker in the same style as the crustal-boundary crossings: a down-arrow standing
    on the TOP strip at along-profile distance dc, a thin tick through every strip, and a rotated
    label above — but RED with a large stroke (vs the black boundary arrows). E.g. the Ida Fault."""
    tr = blended_transform_factory(top_sax.transData, top_sax.transAxes)
    top_sax.annotate("", xy=(dc, 1.0), xycoords=tr, xytext=(0, length),
                     textcoords="offset points", annotation_clip=False, zorder=9,
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                     mutation_scale=head, shrinkA=0, shrinkB=0))
    for sax in strip_axes:                       # thin red tick through the strips
        sax.axvline(dc, color=color, lw=1.0, zorder=7)
    if label:
        top_sax.annotate(label, (dc, 1.0), xycoords=tr, xytext=(1, length + 2),
                         textcoords="offset points", rotation=35, ha="left", va="bottom",
                         fontsize=name_fontsize, color=color, fontweight="bold",
                         annotation_clip=False)


def project_to_profile(plon, plat, dist, dep_lon, dep_lat, swath_km=50.0):
    """Project points onto the profile. Returns along-track distance of the nearest
    profile node (km), perpendicular distance (km), and keep = perp <= swath_km.
    Nearest node is found in an azimuthal-equidistant km frame about the profile midpoint."""
    plon = np.asarray(plon, float); plat = np.asarray(plat, float); dist = np.asarray(dist, float)
    dep_lon = np.asarray(dep_lon, float); dep_lat = np.asarray(dep_lat, float)
    if dep_lon.size == 0:
        return np.array([]), np.array([]), np.array([], bool)
    mid = len(plon) // 2
    P = _local_xy(plon, plat, plon[mid], plat[mid])          # (n, 2) km
    Q = _local_xy(dep_lon, dep_lat, plon[mid], plat[mid])    # (m, 2) km
    d2 = (Q[:, None, 0] - P[None, :, 0]) ** 2 + (Q[:, None, 1] - P[None, :, 1]) ** 2
    j = d2.argmin(1)
    perp = np.sqrt(d2[np.arange(len(Q)), j])
    return dist[j], perp, perp <= swath_km


def draw_deposits_on_strip(sax, proj_km, size, marker="o", facecolor=(1.0, 0.84, 0.0),
                           edgecolor="k", y=0.5, style="markers"):
    """Projected deposits on a strip axis. style='markers' = circles sized by `size`
    (points^2); style='lines' = thin vertical ticks of uniform width (no size)."""
    proj_km = np.asarray(proj_km, float)
    if proj_km.size == 0:
        return
    tr = blended_transform_factory(sax.transData, sax.transAxes)
    if style == "lines":
        sax.vlines(proj_km, 0.0, 1.0, transform=tr, color=facecolor, lw=0.7,
                   alpha=0.85, zorder=9)
    else:
        sax.scatter(proj_km, np.full(proj_km.size, y), s=size, marker=marker,
                    facecolor=facecolor, edgecolor=edgecolor, linewidths=0.3,
                    transform=tr, zorder=9, clip_on=False)


@dataclass
class StripResult:
    strip_axes: list
    runs: dict          # {"litho": [Run...], "tectonic": [...], "geol": [...]}
    crossings: list     # [Crossing...]


def add_profile_strips(top_ax, panel_axes, lon, lat, dist, *, litho=None,
                       tectonic=None, boundaries=None, step_km=0.5,
                       height_in=0.14, gap_in=0.03, labels=True,
                       label_domains=True, label_boundaries=False, dedupe_km=3.0,
                       boundary_lines=False, greenstone=None, deposits=None,
                       dep_swath_km=50.0, geol_label="Au / greenstone", craton=None,
                       faults=None, highlight_boundaries=None, vpvs=None):
    """One-call entry point for plot_xsection.

    top_ax      the uppermost Vs panel of the section (strips go above it)
    panel_axes  every panel of the section (boundary lines go through all)
    lon, lat, dist  the profile exactly as plotted (dist = panel x axis, km)
    litho, tectonic  PolygonIndex or None; drawn bottom-up in that order
    boundaries  BoundarySet or None
    greenstone  PolygonIndex (one-class greenstone) -> filled band on a combined track
    deposits    dict(lon, lat, size[, marker, facecolor]) -> markers on that track,
                projected onto the profile within dep_swath_km (km)
    """
    lon_f, lat_f, d_f = resample_profile(lon, lat, dist, step_km)
    strips, runs = [], {}
    vpvs_sax = None
    # Crustal Vp/Vs colour bar FIRST (level 0) so it sits directly on top of the Vs panel
    if vpvs is not None:
        vpvs_sax = strip_axes_above(top_ax, max(height_in, 0.17), gap_in, level=0)
        vv = sample_vpvs_idw(vpvs["lon"], vpvs["lat"], vpvs["val"], lon_f, lat_f,
                             max_dist_deg=vpvs.get("max_dist", 0.8))
        draw_vpvs_strip(vpvs_sax, d_f, vv, clim=vpvs.get("clim", (1.68, 1.86)),
                        cmap=vpvs.get("cmap"),
                        label=(vpvs.get("label", "Vp/Vs") if labels else None))
        strips.append(vpvs_sax)
        runs["vpvs"] = None
        print(f"  Vp/Vs track: {int(np.isfinite(vv).sum())}/{vv.size} cells filled "
              f"(<= {vpvs.get('max_dist', 0.8):.2f} deg from a station)")
    tect_sax = None
    # When the craton bar sits on the Vp/Vs strip, lift everything above it by _lift inches
    # so the grey bar shows in the opened gap instead of being hidden under the domain bar.
    _lift = 0.16 if (vpvs_sax is not None and craton is not None) else 0.0
    for key, index, lab in (("litho", litho, "Litho"),
                            ("tectonic", tectonic, "Domain")):
        if index is None:
            continue
        r = runs_from_samples(d_f, index.sample(lon_f, lat_f), index)
        sax = strip_axes_above(top_ax, height_in, gap_in, level=len(strips), lift_in=_lift)
        draw_strip(sax, r, label=lab if labels else None)
        if key == "tectonic":
            tect_sax = sax
            if label_domains:
                annotate_runs(sax, r)
        strips.append(sax)
        runs[key] = r
    # Craton extent: a grey bar in the gap just above the Vp/Vs bar (or the domain bar if
    # there is no Vp/Vs strip), no extra row so the next strip stays close
    _craton_base = vpvs_sax if vpvs_sax is not None else tect_sax
    if craton is not None and _craton_base is not None:
        cr = runs_from_samples(d_f, craton.sample(lon_f, lat_f), craton)
        if _lift:                       # taller bar to fill the opened gap above the Vp/Vs strip
            draw_craton_bar(_craton_base, cr, y0=1.10, y1=1.90)
        else:
            draw_craton_bar(_craton_base, cr)
        runs["craton"] = cr
    # Combined greenstone-fill + gold-deposit track, above the domain bar
    if greenstone is not None or deposits is not None:
        sax = strip_axes_above(top_ax, max(height_in, 0.17), gap_in, level=len(strips), lift_in=_lift)
        gr = []
        if greenstone is not None:
            gr = runs_from_samples(d_f, greenstone.sample(lon_f, lat_f), greenstone)
            for rn in gr:
                if rn.name:
                    sax.axvspan(rn.d0, rn.d1, color=rn.color, lw=0, alpha=0.9)
        if deposits is not None and len(np.asarray(deposits.get("lon", []))):
            proj, perp, keep = project_to_profile(lon, lat, dist,
                                                  deposits["lon"], deposits["lat"], dep_swath_km)
            s = np.asarray(deposits.get("size", 18.0), float)
            s = s[keep] if s.ndim else s
            draw_deposits_on_strip(sax, proj[keep], s,
                                   marker=deposits.get("marker", "o"),
                                   facecolor=deposits.get("facecolor", (1.0, 0.84, 0.0)),
                                   style=deposits.get("style", "markers"))
            print(f"  gold track: {int(keep.sum())}/{keep.size} deposits within {dep_swath_km:.0f} km")
        if labels:
            sax.text(-0.005, 0.5, geol_label, transform=sax.transAxes, ha="right",
                     va="center", fontsize=7)
        strips.append(sax)
        runs["geol"] = gr
    # Named faults: red down-arrow on the TOP strip, crustal-boundary style (e.g. Ida Fault)
    if faults and strips:
        for flabel, flon, flat, _fhalf in faults:
            proj, _perp, keep = project_to_profile(lon, lat, dist, [flon], [flat], 1e9)
            if keep.size and keep[0]:
                dc = float(proj[0])
                draw_fault_arrow(strips[-1], strips, dc, label=flabel)
                print(f"  fault arrow: {flabel} at {dc:.0f} km")
            else:
                print(f"  fault arrow: {flabel} projects off the profile - skipped")
    crossings = []
    if boundaries is not None:
        # the plotted great-circle path is a straight line in the crossing frame
        # (azimuthal equidistant about its midpoint), so its own nodes are exact here
        crossings = boundaries.crossings(lon, lat, dist, dedupe_km=dedupe_km)
        hl = {str(k).lower(): v for k, v in (highlight_boundaries or [])}
        def _is_hl(c):
            return bool(hl) and any(str(nm).lower() in hl for nm in c.names)
        normal = [c for c in crossings if not _is_hl(c)]
        draw_boundaries(panel_axes, normal, strip_axes=strips,
                        label_names=label_boundaries, span_km=dist[-1] - dist[0],
                        panel_lines=boundary_lines)
        if strips:                                # highlighted boundaries -> red fault arrow
            for c in crossings:
                if not _is_hl(c):
                    continue
                lab = next((hl[str(nm).lower()] for nm in c.names if str(nm).lower() in hl), None)
                draw_fault_arrow(strips[-1], strips, c.dist, label=lab)
                print(f"  highlight: {lab} at {c.dist:.0f} km (boundary {', '.join(c.names)})")
    # Vp/Vs colorbar on the right, spanning the stacked bar strips' height, aligned in x with
    # the panel colorbars (same figure-coord offset _pcolor uses: panel right edge + 0.10 in).
    if vpvs is not None and strips and vpvs.get("colorbar", True):
        try:
            import matplotlib as _mpl
            fig = top_ax.figure
            fig.canvas.draw()                      # settle inset-axes positions before reading them
            p_top = top_ax.get_position()
            boxes = [s.get_position() for s in strips]
            y0 = min(b.y0 for b in boxes); y1 = max(b.y1 for b in boxes)
            fw = fig.get_figwidth()
            cax = fig.add_axes([p_top.x1 + 0.10 / fw, y0, 0.16 / fw, max(y1 - y0, 0.02)])
            sm = _mpl.cm.ScalarMappable(
                norm=mcolors.Normalize(*vpvs.get("clim", (1.68, 1.86))),
                cmap=vpvs.get("cmap") or "jet_r")
            cb = fig.colorbar(sm, cax=cax, extend="both")
            cb.set_label(vpvs.get("label", "Vp/Vs"), fontsize=7)
            cb.ax.tick_params(labelsize=6)
        except Exception as e:
            print(f"  Vp/Vs colorbar skipped: {e}")
    # Keep the panel title(s) above the strips (and boundary names) rather than under them.
    if strips:
        _pad = 2.0
        if crossings:
            _pad = max(_pad, 48.0 if label_boundaries else 22.0)
        if faults or highlight_boundaries:
            _pad = max(_pad, 30.0)                # clear the red fault arrow + label
        _move_titles(top_ax, strips[-1], pad=_pad)
    top_ax.set_xlim(dist[0], dist[-1])
    return StripResult(strips, runs, crossings)
