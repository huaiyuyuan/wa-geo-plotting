"""make_tectonic_basemap.py: UTF-8 en-dash names must match the colour table, and
West Australian Craton children are grouped by their own unit name."""
import struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def write_polygons(path, polys):
    body = b""
    for i, ring in enumerate(polys, 1):
        pts = np.asarray(ring, float)
        content = (struct.pack("<i", 5)
                   + struct.pack("<4d", pts[:, 0].min(), pts[:, 1].min(),
                                 pts[:, 0].max(), pts[:, 1].max())
                   + struct.pack("<2i", 1, len(pts)) + struct.pack("<i", 0)
                   + struct.pack(f"<{2*len(pts)}d", *pts.ravel()))
        body += struct.pack(">2i", i, len(content) // 2) + content
    hdr = (struct.pack(">i", 9994) + b"\0" * 20 + struct.pack(">i", (100 + len(body)) // 2)
           + struct.pack("<2i", 1000, 5) + struct.pack("<8d", *([0.0] * 8)))
    Path(path).write_bytes(hdr + body)


def write_dbf(path, fields, rows, width=60):
    nrec, nf = len(rows), len(fields)
    hlen, rlen = 32 + 32 * nf + 1, 1 + width * nf
    hdr = struct.pack("<B3BIHH20x", 3, 126, 1, 1, nrec, hlen, rlen)
    fd = b"".join(f.encode().ljust(11, b"\0") + b"C" + b"\0" * 4 + bytes([width, 0]) + b"\0" * 14
                  for f in fields)
    recs = b"".join(b" " + b"".join(v.encode("utf-8").ljust(width) for v in r) for r in rows)
    Path(path).write_bytes(hdr + fd + b"\r" + recs + b"\x1a")


def box(x0, x1, y0, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def test_dash_and_wac_split():
    rows = [("Albany\u2013Fraser Orogen", "Albany\u2013Fraser Orogen, Northern Foreland"),
            ("West Australian Craton", "Pilbara Craton"),
            ("West Australian Craton", "Fortescue Basin"),
            ("Yilgarn Craton", "Youanmi Terrane")]
    with tempfile.TemporaryDirectory() as td:
        shp = Path(td) / "t.shp"
        write_polygons(shp, [box(122, 124, -34, -32), box(117, 120, -22, -20),
                             box(116, 121, -23, -21), box(117, 120, -31, -28)])
        write_dbf(shp.with_suffix(".dbf"), ["PARENTNAME", "TECTNAME"], rows)
        out = Path(td) / "t.npz"
        r = subprocess.run([sys.executable, str(REPO / "extract/make_tectonic_basemap.py"),
                            "--shp", str(shp), "--out-npz", str(out)],
                           check=True, capture_output=True, text=True)
        assert "no colour" not in r.stdout, r.stdout
        d = np.load(out, allow_pickle=True)
        assert list(d["parents"]) == ["Albany-Fraser Orogen", "Pilbara Craton",
                                      "Fortescue Basin", "Yilgarn Craton"]
        assert d["names"][0] == "Albany-Fraser Orogen, Northern Foreland"
        assert d["colors"][0] == "#9467bd"           # Albany-Fraser colour, not default grey
        assert d["colors"][2] == "#e6a157"           # Fortescue gets its own colour


if __name__ == "__main__":
    test_dash_and_wac_split()
    print("ok  test_dash_and_wac_split")
