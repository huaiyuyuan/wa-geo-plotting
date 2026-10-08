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
        cnt = Counter(src)
        print(f"  ({os.path.basename(str(path))}: domain colours " +
              ", ".join(f"{v} {k}" for k, v in cnt.most_common()) + " rings)")
        return cls(d["rings"], cols.astype(str), names)

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
                    if len(r) == 2 and r[1].startswith("#"):
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
        for k in cand:
            Q = _local_xy(self.lines[k][:, 0], self.lines[k][:, 1], lon0, lat0)
            q, s = Q[:-1], Q[1:] - Q[:-1]                 # (K, 2)
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
def strip_axes_above(ax, height_in=0.14, gap_in=0.03, level=0):
    """Thin axes above `ax` sharing its x axis; level=1 stacks above level 0.

    Uses inset_axes in axes-fraction coordinates so it works with any layout
    the cross-section script already uses (gridspec, per-panel colorbars,
    stacked sections), without shrinking the Vs panel.
    """
    fig = ax.figure
    ax.apply_aspect()  # settle fixed-aspect panels (--stack-ve) before sizing
    h_ax_in = ax.get_position().height * fig.get_figheight()
    h, g = height_in / h_ax_in, gap_in / h_ax_in
    sax = ax.inset_axes([0.0, 1.0 + g + level * (h + g), 1.0, h], sharex=ax)
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


@dataclass
class StripResult:
    strip_axes: list
    runs: dict          # {"litho": [Run...], "tectonic": [...]}
    crossings: list     # [Crossing...]


def add_profile_strips(top_ax, panel_axes, lon, lat, dist, *, litho=None,
                       tectonic=None, boundaries=None, step_km=0.5,
                       height_in=0.14, gap_in=0.03, labels=True,
                       label_domains=True, label_boundaries=False, dedupe_km=3.0,
                       boundary_lines=False):
    """One-call entry point for plot_xsection.

    top_ax      the uppermost Vs panel of the section (strips go above it)
    panel_axes  every panel of the section (boundary lines go through all)
    lon, lat, dist  the profile exactly as plotted (dist = panel x axis, km)
    litho, tectonic  PolygonIndex or None; drawn bottom-up in that order
    boundaries  BoundarySet or None
    """
    lon_f, lat_f, d_f = resample_profile(lon, lat, dist, step_km)
    strips, runs = [], {}
    for key, index, lab in (("litho", litho, "Litho"),
                            ("tectonic", tectonic, "Domain")):
        if index is None:
            continue
        r = runs_from_samples(d_f, index.sample(lon_f, lat_f), index)
        sax = strip_axes_above(top_ax, height_in, gap_in, level=len(strips))
        draw_strip(sax, r, label=lab if labels else None)
        if key == "tectonic" and label_domains:
            annotate_runs(sax, r)
        strips.append(sax)
        runs[key] = r
    crossings = []
    if boundaries is not None:
        crossings = boundaries.crossings(lon_f, lat_f, d_f, dedupe_km=dedupe_km)
        draw_boundaries(panel_axes, crossings, strip_axes=strips,
                        label_names=label_boundaries, span_km=dist[-1] - dist[0],
                        panel_lines=boundary_lines)
    # Keep the panel title(s) above the strips (and boundary names) rather than under them.
    if strips:
        _move_titles(top_ax, strips[-1],
                     pad=(48.0 if label_boundaries else 22.0) if crossings else 2.0)
    top_ax.set_xlim(dist[0], dist[-1])
    return StripResult(strips, runs, crossings)
