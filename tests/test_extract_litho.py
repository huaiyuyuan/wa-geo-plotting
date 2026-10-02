"""make_litho.py saves GSWA unit colours + record index; wa_basemap.add_gswa_units
draws each record as one shape with real holes (an inlier shows its own colour)."""
import struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "basemap"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_extract_tectonics import write_dbf  # noqa: E402


def write_polygons(path, records):
    """records: list of lists of rings (first = outer, clockwise; rest = holes)."""
    body = b""
    for i, rings in enumerate(records, 1):
        pts = np.vstack(rings).astype(float)
        idx = np.cumsum([0] + [len(r) for r in rings[:-1]])
        content = (struct.pack("<i", 5)
                   + struct.pack("<4d", pts[:, 0].min(), pts[:, 1].min(),
                                 pts[:, 0].max(), pts[:, 1].max())
                   + struct.pack("<2i", len(rings), len(pts))
                   + struct.pack(f"<{len(rings)}i", *idx)
                   + struct.pack(f"<{2*len(pts)}d", *pts.ravel()))
        body += struct.pack(">2i", i, len(content) // 2) + content
    hdr = (struct.pack(">i", 9994) + b"\0" * 20 + struct.pack(">i", (100 + len(body)) // 2)
           + struct.pack("<2i", 1000, 5) + struct.pack("<8d", *([0.0] * 8)))
    Path(path).write_bytes(hdr + body)


cw = lambda x0, x1, y0, y1: [[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]]    # outer
ccw = lambda x0, x1, y0, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]   # hole


def test_gswa_colours_and_holes():
    recs = [[cw(0, 10, 0, 10), ccw(4, 6, 4, 6)],        # granite with a hole
            [cw(4, 6, 4, 6)],                            # greenstone inlier in the hole
            [cw(10, 14, 0, 10)]]                         # unit with CMYK colour
    rows = [("granitic rocks", "Granite unit", "255 190 190"),
            ("greenstones", "Belt", "#7FE5A0"),
            ("sedimentary rocks", "Basin", "C0 M0 Y40 K0")]
    with tempfile.TemporaryDirectory() as td:
        shp = Path(td) / "l.shp"
        write_polygons(shp, recs)
        write_dbf(shp.with_suffix(".dbf"), ["LITHOLOGY", "TECTNAME", "TECTCOLOUR"], rows)
        out = Path(td) / "l.npz"
        r = subprocess.run([sys.executable, str(REPO / "extract/make_litho.py"),
                            "--shp", str(shp), "--out-npz", str(out)],
                           check=True, capture_output=True, text=True)
        assert "'rgb': 1" in r.stdout and "'hex': 1" in r.stdout and "'cmyk': 1" in r.stdout
        d = np.load(out, allow_pickle=True)
        assert list(d["tect_colors"]) == ["#ffbebe", "#ffbebe", "#7fe5a0", "#ffff99"]
        assert list(d["rec_idx"]) == [0, 0, 1, 2]

        import wa_basemap as wb
        fig = plt.figure(figsize=(3.5, 2.5), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
        wb.add_gswa_units(ax, str(out), lw=0)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        px = lambda x, y: tuple(img[int((1 - y / 10) * (img.shape[0] - 1)),
                                    int(x / 14 * (img.shape[1] - 1))])
        assert px(5, 5) == (0x7f, 0xe5, 0xa0), px(5, 5)    # inlier visible, not granite
        assert px(2, 2) == (0xff, 0xbe, 0xbe), px(2, 2)    # granite
        assert px(12, 5) == (0xff, 0xff, 0x99), px(12, 5)  # CMYK unit
        plt.close(fig)

        # granite record ALONE: its hole must be empty (white), i.e. a real hole
        g = dict(d)
        keep = np.array([0, 1])
        np.savez(Path(td) / "g.npz", **{k: (v[keep] if k in ("rings", "colors", "parents",
                 "names", "tect_colors", "tect_colour_raw", "rec_idx") else v) for k, v in g.items()})
        fig = plt.figure(figsize=(3.5, 2.5), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
        wb.add_gswa_units(ax, str(Path(td) / "g.npz"), lw=0)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        assert px(5, 5) == (255, 255, 255), px(5, 5)       # hole is empty
        assert px(2, 2) == (0xff, 0xbe, 0xbe)
        plt.close(fig)


if __name__ == "__main__":
    test_gswa_colours_and_holes()
    print("ok  test_gswa_colours_and_holes")
