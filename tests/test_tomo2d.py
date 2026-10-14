"""Tests for plotting/tomo2d.py (v3b dispersion-map loader, coverage mask, Albers frame)."""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'plotting'))
import tomo2d as t2                                            # noqa: E402


def test_density_mask_fill_largest_no_dilation():
    d = np.zeros((20, 20))
    d[3:12, 3:12] = 20          # main region
    d[6:8, 6:8] = 0             # hole -> filled
    d[16:18, 16:18] = 50        # island -> dropped
    m = t2.density_mask(d, 10)
    assert m[6, 6] and m[3, 3] and not m[16, 16]
    assert not m[2, 3] and not m[12, 5]          # no dilation
    assert m.sum() == 81


def _write(tmp, lat_desc=True, transpose=False):
    lat = np.linspace(-36, -13, 7); lon = np.linspace(112, 131, 5)
    LON, LAT = np.meshgrid(lon, lat)
    v = np.stack([LAT + 100, LAT + 200])          # value encodes latitude
    if lat_desc:
        LON, LAT, v = LON[::-1], LAT[::-1], v[:, ::-1]
    if transpose:
        LON, LAT, v = LON.T, LAT.T, np.swapaxes(v, 1, 2)
    p = os.path.join(tmp, 'Z', 'phase_v_tomo'); os.makedirs(p, exist_ok=True)
    f = os.path.join(p, 'TransD.region.phase.npz')
    np.savez(f, fk=np.array([5.0, 10.0]), vall2=v, std2=np.zeros_like(v) + 0.5,
             maski=np.where(v > 0, 1.0, np.nan), lat2d=LAT, lon2d=LON)
    return f


def test_load_orientation_and_set():
    for desc in (False, True):
        for tr in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                T = t2.load_tomo(_write(tmp, desc, tr))
                assert np.all(np.diff(T['lat']) > 0) and np.all(np.diff(T['lon']) > 0)
                assert np.allclose(T['v'][0][:, 0], T['lat'] + 100)
                assert (T['comp'], T['typ']) == ('Z', 'phase')
                assert T['maski'].all()


def test_coverage_on_other_orientation():
    with tempfile.TemporaryDirectory() as tmp:
        T = t2.load_tomo(_write(tmp))
        LON, LAT = T['LON'], T['LAT']
        dens = np.where(LAT < -25, 20.0, 0.0)
        fp = os.path.join(tmp, 'fp.npz')
        np.savez(fp, lat2d=LAT[::-1].T, lon2d=LON[::-1].T, density=dens[::-1].T)
        cov, _ = t2.load_coverage(T['lat'], T['lon'], fp)
        assert np.array_equal(cov, LAT < -25)


def test_albers_orientation_and_periods():
    P = t2.Albers()
    x0, y0 = P(121, -25)
    x1, y1 = P(122, -25)
    x2, y2 = P(121, -24)
    assert x1 > x0 and abs(y1 - y0) < 5 and y2 > y0      # east right, north up
    assert abs(np.hypot(x1 - x0, y1 - y0) - 101) < 3       # ~101 km per degree at 25S
    per = np.array([2.5, 5, 10, 20, 40])
    assert t2.pick_periods(per, ['5', '60', '10']) == [1, 2]
    assert t2.pick_periods(per, ['all']) == list(range(5))
    assert t2.caveats_for('Z', 'group', 35) and not t2.caveats_for('Z', 'phase', 35)
