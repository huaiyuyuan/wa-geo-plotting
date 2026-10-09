"""basemap/domains.py: domain membership from 10M units (names, parents, holes)."""
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "basemap"))
from domains import Domains, model_fields, load_domains  # noqa: E402


def box(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float)


def objarr(items):
    a = np.empty(len(items), object)
    for i, x in enumerate(items):
        a[i] = x
    return a


def test_membership():
    td = Path(tempfile.mkdtemp())
    U = [("West Australian Craton", "STATE", box(0, 20, 0, 20)),
         ("Yilgarn Craton", "Yilgarn Craton", box(5, 15, 0, 10)),
         ("Yilgarn Craton", "Yilgarn Craton", box(9, 11, 4, 6)),          # hole in the outline
         ("Youanmi Terrane", "Yilgarn Craton", box(9, 11, 4, 6)),         # ...filled by a child
         ("Fraser Zone", "Kepa Kurl Booya Province", box(15, 18, 0, 3)),
         ("Perth Basin", "Phanerozoic basins", box(1, 4, 0, 10))]
    np.savez(td / "t.npz", rings=objarr([u[2] for u in U]), names=np.array([u[0] for u in U]),
             parents=np.array([u[1] for u in U]))
    D = Domains(td / "t.npz")
    lon = np.array([10.0, 6.0, 2.0, 16.0, 19.0])
    lat = np.array([5.0, 8.0, 5.0, 1.0, 19.0])
    assert list(D.contains("yilgarn", lon, lat)) == [True, True, False, False, False]
    assert list(D.contains("perth", lon, lat)) == [False, False, True, False, False]
    assert list(D.contains("albany_fraser", lon, lat)) == [False, False, False, True, False]
    assert D.contains("wac", lon, lat).all()
    d = dict(Lon=lon, Lat=lat, Ndat=np.array([60, 10, 60, 60, 60]))
    assert list(D.node_mask(d, "yilgarn")) == [True, False, False, False, False]   # Ndat cut
    V = np.array([[3.0, 3.5], [9, 9], [9, 9], [9, 9], [9, 9]], float)
    assert np.allclose(D.mean_profile(d, V, "yilgarn"), [3.0, 3.5])


def test_fields_and_csv():
    d = dict(Vsv=np.full((3, 2), 3.6), Xi=np.array([[1.0, 1.05]] * 3), Vpvs=np.full((3, 2), 1.73))
    F = model_fields(d)
    assert set(F) == {"vsv", "viso", "xi"}                  # Vp/Vs fixed -> left out
    assert np.all(F["vsv"] <= F["viso"] + 1e-12)            # Vsv <= Viso when xi >= 1
    keys = list(load_domains())
    assert keys[:5] == ["yilgarn", "wac", "perth", "albany_fraser", "capricorn"]


if __name__ == "__main__":
    test_membership(); test_fields_and_csv()
    print("ok  test_membership\nok  test_fields_and_csv\n2 passed")
