"""basemap/footprint.py: the same cells are selected whichever way the npz grid
is stored (north/south row first, lat along rows or columns)."""
import sys, tempfile
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "basemap"))
import footprint as F  # noqa: E402


def test_orientation_independent():
    lat = -13.0 - 0.333333 * np.arange(70)
    lon = 112.0 + 0.322034 * np.arange(60)
    LO, LA = np.meshgrid(lon, lat)
    poly = np.array([[116.8, -32], [123, -32], [123, -28.4], [119.5, -27.6], [116.8, -28.4], [116.8, -32]])
    m = MPath(poly).contains_points(np.column_stack([LO.ravel(), LA.ravel()])).reshape(LO.shape)
    plon = np.array([119.5, 119.5, 113.0, 122.9, 123.4, 140.0])
    plat = np.array([-30.0, -27.0, -30.0, -31.9, -31.9, -30.0])
    want = [True, False, False, True, False, False]
    with tempfile.TemporaryDirectory() as td:
        for name, (mm, la, lo) in {"north": (m, LA, LO), "south": (m[::-1], LA[::-1], LO[::-1]),
                                   "transposed": (m.T, LA.T, LO.T)}.items():
            f = Path(td) / f"{name}.npz"
            np.savez(f, mask=mm, lat2d=la, lon2d=lo)
            got = list(F.Footprint(str(f), outline="/nonexistent").contains(plon, plat))
            assert got == want, (name, got)


if __name__ == "__main__":
    test_orientation_independent()
    print("ok  test_orientation_independent")
