"""Tests for plotting/xsection_strips.py on synthetic geology.

Run:  python -m pytest tests/  (or)  python tests/test_xsection_strips.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
import xsection_strips as xs  # noqa: E402


def objarr(items):
    out = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        out[i] = x
    return out


def box(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])


GRANITE, GREEN = "#f4b6c2", "#4c9a52"


def make_index(with_inlier=True):
    rings = [box(117, 124, -32, -28),                 # granite background
             box(119, 120, -31, -29),                 # greenstone belt
             box(119.4, 119.6, -30.2, -29.8)]         # hole in the belt
    colors = [GRANITE, GREEN, GREEN]
    parents = ["granite", "greenstone", "greenstone"]
    if with_inlier:                                   # granite filling the hole
        rings.append(box(119.4, 119.6, -30.2, -29.8))
        colors.append(GRANITE)
        parents.append("granite inlier")
    return xs.PolygonIndex(objarr(rings), colors, parents)


def test_great_circle_distance():
    lon, lat, d = xs.great_circle_points(115.86, -31.95, 121.47, -30.75, step_km=5)
    assert 530 < d[-1] < 560                           # Perth-Kalgoorlie ~545 km
    assert np.allclose(xs.along_track_km(lon, lat), d, atol=1e-6)


def test_point_in_polygon_with_holes_and_inliers():
    idx = make_index(with_inlier=True)
    lon = np.array([118.0, 119.2, 119.5, 130.0])
    lat = np.full(4, -30.0)
    got = idx.parents[idx.sample(lon, lat)]
    assert list(got[:3]) == ["granite", "greenstone", "granite inlier"]
    assert idx.sample(lon, lat)[3] == -1               # outside everything


def test_unfilled_hole_falls_through_to_background():
    idx = make_index(with_inlier=False)
    k = idx.sample(np.array([119.5]), np.array([-30.0]))[0]
    assert idx.parents[k] == "granite"


def test_runs_tile_profile():
    idx = make_index()
    lon, lat, d = xs.great_circle_points(117.5, -30, 123.5, -30, step_km=0.5)
    runs = xs.runs_from_samples(d, idx.sample(lon, lat), idx)
    assert runs[0].d0 == d[0] and runs[-1].d1 == d[-1]
    assert all(np.isclose(a.d1, b.d0) for a, b in zip(runs[:-1], runs[1:]))
    names = [r.name for r in runs]
    assert names == ["granite", "greenstone", "granite inlier", "greenstone", "granite"]
    belt = [r for r in runs if r.name == "greenstone"]
    width = sum(r.d1 - r.d0 for r in belt)             # 1 deg minus 0.2 deg hole
    assert abs(width - 0.8 * 111.32 * np.cos(np.radians(30))) < 3


def test_boundary_crossings_and_dedupe():
    ns = np.array([[121.0, -33.0], [121.0, -27.0]])
    ns_dup = ns + [0.005, 0.0]                         # same edge digitised twice
    far = np.array([[130.0, -33.0], [130.0, -27.0]])
    zig = np.array([[118.0, -31.0], [118.5, -29.0], [119.0, -31.0], [119.5, -29.0]])
    bs = xs.BoundarySet(objarr([ns, ns_dup, far, zig]),
                        names=["Ida", "Ida", "far", "zig"])
    lon, lat, d = xs.great_circle_points(117.5, -30, 123.5, -30, step_km=2)
    cr = bs.crossings(lon, lat, d, dedupe_km=3)
    assert len(cr) == 4                                # 3 zig + 1 merged Ida
    ida = cr[-1]
    assert ida.names == ["Ida"]
    _, _, d_ida = xs.great_circle_points(117.5, -30, 121.0, -30, step_km=2)
    assert abs(ida.dist - d_ida[-1]) < 2               # gc vs parallel curvature


def test_add_profile_strips_smoke(tmp_path=None):
    idx = make_index()
    bs = xs.BoundarySet(objarr([np.array([[121.0, -33], [121.0, -27]])]))
    lon, lat, d = xs.great_circle_points(117.5, -30, 123.5, -30, step_km=10)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axs[0].set_title("A-A'")
    res = xs.add_profile_strips(axs[0], axs, lon, lat, d, litho=idx, tectonic=idx,
                                boundaries=bs)
    assert len(res.strip_axes) == 2 and len(res.crossings) == 1
    assert res.strip_axes[-1].get_title() == "A-A'" and axs[0].get_title() == ""
    out = Path(tmp_path or "/tmp") / "strip_smoke.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    assert out.stat().st_size > 0


def test_class_key_names_and_fallback(tmp_path=None):
    import tempfile
    td = Path(tmp_path or tempfile.mkdtemp())
    rings = objarr([box(117, 120, -32, -28), box(120, 124, -32, -28)])
    common = dict(rings=rings, colors=np.array(["#2ca02c"] * 2),
                  parents=np.array(["Yilgarn Craton"] * 2))
    np.savez(td / "with.npz", names=np.array(["Youanmi Terrane", "Kalgoorlie Terrane"]), **common)
    np.savez(td / "without.npz", **common)
    lon, lat = np.array([118.0, 122.0]), np.array([-30.0, -30.0])
    t = xs.PolygonIndex.from_npz(td / "with.npz", class_key="names")
    assert list(t.parents[t.sample(lon, lat)]) == ["Youanmi Terrane", "Kalgoorlie Terrane"]
    assert t.colors[0] != t.colors[1]            # palette, not the shared craton colour
    f = xs.PolygonIndex.from_npz(td / "without.npz", class_key="names")
    assert set(f.parents) == {"Yilgarn Craton"}  # graceful fallback


def test_left_title_moves_above_strips():
    idx = make_index()
    lon, lat, d = xs.great_circle_points(117.5, -30, 123.5, -30, step_km=10)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_title("Vsv (km/s)", loc="left", fontsize=9)
    res = xs.add_profile_strips(ax, [ax], lon, lat, d, litho=idx)
    assert ax.get_title(loc="left") == ""
    assert res.strip_axes[-1].get_title(loc="left") == "Vsv (km/s)"
    plt.close(fig)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
