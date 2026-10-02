"""make_boundaries.py: names must follow each part's record, through
multipart records and null shapes. Writes a tiny shapefile + DBF by hand."""
import struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def write_shp(path, records):
    """records: list of None (null shape) or list of parts [(N,2) arrays]."""
    body = b""
    for i, parts in enumerate(records, 1):
        if parts is None:
            content = struct.pack("<i", 0)
        else:
            pts = np.vstack(parts)
            idx = np.cumsum([0] + [len(p) for p in parts[:-1]])
            content = (struct.pack("<i", 3)
                       + struct.pack("<4d", pts[:, 0].min(), pts[:, 1].min(),
                                     pts[:, 0].max(), pts[:, 1].max())
                       + struct.pack("<2i", len(parts), len(pts))
                       + struct.pack(f"<{len(parts)}i", *idx)
                       + struct.pack(f"<{2*len(pts)}d", *pts.ravel()))
        body += struct.pack(">2i", i, len(content) // 2) + content
    hdr = (struct.pack(">i", 9994) + b"\0" * 20 + struct.pack(">i", (100 + len(body)) // 2)
           + struct.pack("<2i", 1000, 3) + struct.pack("<8d", *([0.0] * 8)))
    Path(path).write_bytes(hdr + body)


def write_dbf(path, names, width=20):
    nrec, hlen, rlen = len(names), 32 + 32 + 1, 1 + width
    hdr = struct.pack("<B3BIHH20x", 3, 126, 1, 1, nrec, hlen, rlen)
    fld = b"NAME".ljust(11, b"\0") + b"C" + b"\0" * 4 + bytes([width, 0]) + b"\0" * 14
    recs = b"".join(b" " + n.encode().ljust(width) for n in names)
    Path(path).write_bytes(hdr + fld + b"\r" + recs + b"\x1a")


def test_names_follow_parts():
    a = np.array([[118.0, -32.0], [118.0, -28.0]])
    b1, b2 = np.array([[120.0, -32.0], [120.0, -30.0]]), np.array([[120.5, -30.0], [120.5, -28.0]])
    c = np.array([[122.0, -32.0], [122.0, -28.0]])
    recs = [[a], None, [b1, b2], [c]]
    names = ["Alpha Fault", "Null Shape", "Beta Shear (2 parts)", "Gamma Fault"]
    with tempfile.TemporaryDirectory() as td:
        shp = Path(td) / "b.shp"
        write_shp(shp, recs)
        write_dbf(shp.with_suffix(".dbf"), names)
        out = Path(td) / "b.npz"
        subprocess.run([sys.executable, str(REPO / "extract/make_boundaries.py"),
                        "--shp", str(shp), "--out-npz", str(out)], check=True,
                       capture_output=True)
        d = np.load(out, allow_pickle=True)
        assert "allow_pickle" not in d.files
        assert list(d["names"]) == ["Alpha Fault", "Beta Shear (2 parts)",
                                    "Beta Shear (2 parts)", "Gamma Fault"]
        assert list(d["rec_idx"]) == [0, 2, 2, 3]
        assert d["lines"].shape == (4,)               # 1-D even though all lines are 2-pt
        assert np.allclose(np.asarray(d["lines"][2], float), b2)


if __name__ == "__main__":
    test_names_follow_parts()
    print("ok  test_names_follow_parts")
