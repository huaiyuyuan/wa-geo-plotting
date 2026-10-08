"""basemap/wa_basemap.add_map2022_units: 2022 map colours, holes and inliers.

Run:  python -m pytest tests/  (or)  python tests/test_map2022_units.py
"""
import sys, tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "basemap"))
import wa_basemap as wb  # noqa: E402


def box(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float)


def objarr(items):
    out = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        out[i] = x
    return out


def pixel(fig, ax, lon, lat):
    fig.canvas.draw()
    x, y = ax.transData.transform((lon, lat))
    a = np.asarray(fig.canvas.buffer_rgba())
    return matplotlib.colors.to_hex(a[a.shape[0] - int(round(y)), int(round(x))][:3] / 255)


def test_colours_holes_inliers():
    td = Path(tempfile.mkdtemp())
    rings = [box(0, 10, 0, 10), box(4, 6, 4, 6),       # Youanmi with a hole (same winding)
             box(12, 14, 0, 10),                         # parent-only match
             box(16, 18, 0, 10)]                         # TECTCOLOUR fallback
    np.savez(td / "t.npz", rings=objarr(rings),
             names=np.array(["Youanmi Terrane", "Youanmi Terrane", "Kalgoorlie Terrane", "X"]),
             parents=np.array(["Yilgarn Craton", "Yilgarn Craton", "Yilgarn Craton", "None"]),
             colors=np.array(["#aaaaaa"] * 4), tect_colors=np.array(["", "", "", "#123456"]))
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    src = wb.add_map2022_units(ax, td / "t.npz", lw=0, verbose=False)
    ax.set_xlim(-1, 19); ax.set_ylim(-1, 11); ax.set_facecolor("w")
    assert src == {"map2022": 1, "map2022 (parent)": 1, "TECTCOLOUR": 1}
    assert pixel(fig, ax, 2, 2) == "#fbcbbb"             # Youanmi, from the table
    assert pixel(fig, ax, 5, 5) == "#ffffff"             # hole stays open
    assert pixel(fig, ax, 13, 5) == "#fbcbbb"            # parent fallback (Yilgarn)
    assert pixel(fig, ax, 17, 5) == "#123456"            # TECTCOLOUR
    plt.close(fig)


def test_legend_and_labels_smoke():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(112, 130); ax.set_ylim(-36, -13)
    ia = wb.add_map2022_legend(ax)
    assert ia is not None and len(ia.patches) > 15 and len(ia.texts) >= 20
    n1 = wb.add_map2022_labels(ax, level=1)
    n2 = wb.add_map2022_labels(ax, level=2)
    assert 20 <= n1 < n2
    names = [t.get_text() for t in ax.texts]
    assert "YILGARN CRATON" in names and "C A P R I C O R N   O R O G E N" in names
    assert not any(ord(c) > 0x2000 and c != "\u2013" for n in names for c in n)
    fig.savefig(Path(tempfile.mkdtemp()) / "leg.png", dpi=60)
    plt.close(fig)


if __name__ == "__main__":
    test_legend_and_labels_smoke()
    print("ok  test_legend_and_labels_smoke")
    test_colours_holes_inliers()
    print("ok  test_colours_holes_inliers\n2 passed")
