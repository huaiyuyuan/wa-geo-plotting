#!/usr/bin/env python3
"""
export_matlab.py - everything the MATLAB scripts need to redraw the profile-location map
and the cross-section stacks, in one .mat file (MATLAB v5, no toolboxes needed to read).

  python3 plotting/export_matlab.py --fvs $FVS_ZT --vel vsv_true --load-sections $XS \\
      --moho $MOHO --out $ST/figures/iter8/wa_plotdata.mat [--footprint]

then in MATLAB (matlab/ folder of this repo on the path):
  wa_plot_map('wa_plotdata.mat')                 % -> xsection_index_map.png/.pdf
  wa_plot_stack('wa_plotdata.mat', 'dvsv')       % -> xsection_stack.dvsv.ns/.ew png/pdf

Contents (lon/lat; MATLAB projects with the same Albers as the Python map):
  units      10M tectonic units in the 2022 map colours (concealed units left out), big first;
             rings NaN-separated, outer anticlockwise / holes clockwise
  bnd        major crustal boundaries (lithospheric/crustal, low-confidence flag)
  ocean, states, coast, wa   Natural Earth 50m (via cartopy) clipped to the region
  labels     unit names at the GSWA map positions; legend = the map's rock-type x age chart
  stations, footprint (optional), sections (path + ticks), extent
  xsec       per section: dist, z, fields (vsv, dvsv[, xi, dxi]), Moho, domain-strip runs,
             boundary crossings; groups ns / ew as in plot_xsection.py --stack
  cmaps      the exact colour maps (16 colours; relative maps with the white centre)
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
for _p in (os.path.join(_HERE, '..', 'basemap'), os.path.join(_HERE, '..')):
    if os.path.abspath(_p) not in map(os.path.abspath, sys.path):
        sys.path.insert(1, os.path.abspath(_p))
import plot_xsection as px                                            # noqa: E402
import xsection_strips as xs                                          # noqa: E402
import wa_basemap as wb                                               # noqa: E402


def objarr(items):
    out = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        out[i] = x
    return out


def rgb(c):
    import matplotlib.colors as mc
    return np.asarray(mc.to_rgb(c), float)


def nan_join(rings):
    """[ring, ring, ...] -> one Nx2 array, rings separated by NaN rows (MATLAB polyshape)."""
    parts = []
    for r in rings:
        r = np.asarray(r, float)[:, :2]
        parts += [r, np.full((1, 2), np.nan)]
    return np.vstack(parts[:-1]) if parts else np.zeros((0, 2))


def natural_earth(box):
    """Ocean, non-WA Australian states, coastline and WA outline (50m), clipped to box."""
    import cartopy.io.shapereader as shp
    from shapely.geometry import box as sbox
    from shapely.geometry.polygon import orient
    B = sbox(box[0], box[2], box[1], box[3])

    def polys(geom):
        g = geom.intersection(B)
        if g.is_empty:
            return []
        gs = getattr(g, 'geoms', [g])
        out = []
        for p in gs:
            if p.geom_type != 'Polygon':
                continue
            p = orient(p, 1.0)                       # exterior anticlockwise, holes clockwise
            out.append(nan_join([np.asarray(p.exterior.coords)] +
                                [np.asarray(h.coords) for h in p.interiors]))
        return out

    ocean = []
    for rec in shp.Reader(shp.natural_earth('50m', 'physical', 'ocean')).records():
        ocean += polys(rec.geometry)
    states, wa = [], []
    for rec in shp.Reader(shp.natural_earth('50m', 'cultural', 'admin_1_states_provinces')).records():
        if rec.attributes.get('admin', '') != 'Australia':
            continue
        if rec.attributes.get('name', '') == 'Western Australia':
            wa += polys(rec.geometry)
        else:
            states += polys(rec.geometry)
    coast = []
    for rec in shp.Reader(shp.natural_earth('50m', 'physical', 'coastline')).records():
        g = rec.geometry.intersection(B)
        for ln in getattr(g, 'geoms', [g]):
            if ln.geom_type == 'LineString' and not ln.is_empty:
                coast.append(np.asarray(ln.coords)[:, :2])
    return dict(ocean=objarr(ocean), states=objarr(states), coast=objarr(coast), wa=objarr(wa))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs', required=True)
    ap.add_argument('--vel', default='model', choices=['model', 'vsv_true'])
    ap.add_argument('--load-sections', required=True)
    ap.add_argument('--moho', default=None, help='AR23 Moho file (default config.MOHO if present)')
    ap.add_argument('--ds', type=float, default=0.08)
    ap.add_argument('--d-max', type=float, default=60.0)
    ap.add_argument('--labels', default='major', choices=['major', 'all', 'none'])
    ap.add_argument('--label-scale', type=float, default=1.4)
    ap.add_argument('--index-tick', type=float, default=100)
    ap.add_argument('--stations', default=None)
    ap.add_argument('--no-stations', action='store_true')
    ap.add_argument('--footprint', action='store_true')
    ap.add_argument('--footprint-npz', default=None)
    ap.add_argument('--footprint-outline', default=None)
    ap.add_argument('--footprint-veil', type=float, default=0.5)
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--bnd-npz', default=None)
    ap.add_argument('--ncolors', type=int, default=16)
    ap.add_argument('--xi-ref', default='depth', choices=['depth', 'global'])
    ap.add_argument('--out', default='wa_plotdata.mat')
    a = ap.parse_args()
    from scipy.io import savemat

    d = px._load_fvs(a.fvs, a.vel)
    print(f"Fvs: {a.fvs} ({len(d['Lon'])} nodes)")
    Lon, Lat = np.asarray(d['Lon'], float), np.asarray(d['Lat'], float)
    sections = px._load_sections(a.load_sections)
    tect_npz = px._resolve_npz('TECTONICS_NPZ', 'wa_tectonics.npz', a.tect_npz)
    bnd_npz = px._resolve_npz('BOUNDARIES_NPZ', 'wa_crustal_boundaries.npz', a.bnd_npz)
    cb = (Lon.min() - 1.0, Lon.max() + 1.0, Lat.min() - 1.0, Lat.max() + 1.0)
    M = {}

    # --- map layers -------------------------------------------------------------------
    shapes, src = wb.map2022_unit_shapes(tect_npz, clip_box=cb)
    print('units: ' + ', '.join(f'{v} {k}' for k, v in src.items()))
    M['units'] = dict(xy=objarr([nan_join(s['rings']) for s in shapes]),
                      rgb=np.array([rgb(s['colour']) for s in shapes]),
                      name=objarr([s['name'] for s in shapes]))
    B = np.load(bnd_npz, allow_pickle=True)
    n = len(B['lines'])
    scale = B['scale'] if 'scale' in B.files else np.array(['lithospheric'] * n)
    conf = B['conf'] if 'conf' in B.files else np.array(['high'] * n)
    lines, crust, low = [], [], []
    for ln, sc, cf in zip(B['lines'], scale, conf):
        ln = np.asarray(ln, float)[:, :2]
        if len(ln) < 2 or not wb._in_box(ln, cb):
            continue
        lines.append(ln); crust.append(str(sc).lower() == 'crustal')
        low.append(str(cf).lower() in wb.LOW_CONF)
    M['bnd'] = dict(xy=objarr(lines), crustal=np.array(crust, float), low=np.array(low, float))
    print(f"boundaries: {len(lines)} lines")
    big = (min(cb[0], 105.0), max(cb[1], 140.0), max(cb[2], -48.0), min(cb[3], -5.0))
    try:
        M.update(natural_earth(big))
        print(f"Natural Earth: {len(M['ocean'])} ocean, {len(M['states'])} state, "
              f"{len(M['coast'])} coast parts")
    except Exception as e:
        print(f"  (Natural Earth via cartopy unavailable: {e}) - no coast/ocean in the .mat")
        M.update(ocean=objarr([]), states=objarr([]), coast=objarr([]), wa=objarr([]))
    labels = [] if a.labels == 'none' else wb.map2022_label_items(1 if a.labels == 'major' else 2, cb)
    M['labels'] = dict(text=objarr([L['text'] for L in labels]),
                       lon=np.array([L['lon'] for L in labels]), lat=np.array([L['lat'] for L in labels]),
                       angle=np.array([L['angle'] for L in labels]),
                       rgb=np.array([rgb(L['colour']) for L in labels]).reshape(-1, 3),
                       bold=np.array([L['weight'] == 'bold' for L in labels], float),
                       size=np.array([L['size_pt'] * a.label_scale for L in labels]))
    print(f"labels: {len(labels)} ({a.labels})")
    leg = json.load(open(os.path.join(os.path.dirname(wb.__file__), 'tectonic_map_2022_legend.json')))
    M['legend'] = dict(
        width=leg['width'], height=leg['height'],
        title='Tectonic units: rock type and age (GSWA 2022, CC BY 4.0)',
        pts=objarr([np.asarray(s['pts'], float) for s in leg['shapes']]),
        fill=objarr([rgb(s['fill']) if s['fill'] else np.zeros((0, 3)) for s in leg['shapes']]),
        stroke=objarr([rgb(s['stroke']) if s['stroke'] else np.zeros((0, 3)) for s in leg['shapes']]),
        lw=np.array([s['lw'] for s in leg['shapes']]),
        closed=np.array([s['closed'] for s in leg['shapes']], float),
        text=objarr([t['s'] for t in leg['texts']]),
        tx=np.array([[(t['x0'] + t['x1']) / 2,
                      (t['y0'] + t['y1']) / 2 if t['rot'] else t['y'] + 0.42 * t['size']]
                     for t in leg['texts']]),
        tlen=np.array([(t['y1'] - t['y0']) if t['rot'] else (t['x1'] - t['x0']) for t in leg['texts']]),
        tsize=np.array([t['size'] for t in leg['texts']]),
        trot=np.array([t['rot'] for t in leg['texts']], float))
    sta = np.zeros((0, 2))
    if not a.no_stations:
        try:
            import config
            spath, scols = a.stations or config.STATIONS, config.STATION_COLS
        except Exception:
            spath, scols = a.stations, (2, 1)
        if spath and os.path.isfile(spath):
            slon, slat = px._load_stations(spath, scols)
            sta = np.column_stack([slon, slat])
    M['stations'] = sta
    print(f"stations: {len(sta)}")
    M['footprint'] = dict(xy=np.zeros((0, 2)), veil=0.0)
    if a.footprint:
        from footprint import Footprint
        fp = Footprint(a.footprint_npz, a.footprint_outline)
        if fp.olon is not None:
            M['footprint'] = dict(xy=np.column_stack([fp.olon, fp.olat]), veil=a.footprint_veil)
    M['extent'] = np.array([Lon.min() - 0.5, Lon.max() + 0.5, Lat.min() - 0.5, Lat.max() + 0.5])
    M['tick_km'] = float(a.index_tick)

    # --- cross-sections -----------------------------------------------------------------
    z = np.asarray(d['z'], float)
    zm = z <= a.d_max if a.d_max else np.ones(len(z), bool)
    z = z[zm]
    V = np.asarray(d['Vsv'], float)[:, zm]
    vmean = np.nanmean(V, axis=0)
    has_xi = px._varies(d, 'Xi')
    if has_xi:
        X = np.asarray(d['Xi'], float)[:, zm]
        xref = px._xi_ref(X, a.xi_ref)
    is_zt = has_xi or 'ZT' in os.path.basename(a.fvs)
    vname = px._vname(d, is_zt)
    moho = a.moho
    if moho is None:
        try:
            import config
            moho = config.MOHO if os.path.isfile(config.MOHO) else None
        except Exception:
            moho = None
    mi = None
    if moho:
        from scipy.interpolate import RectBivariateSpline
        md = np.loadtxt(moho, skiprows=11)
        mla, mlo = np.unique(md[:, 0]), np.unique(md[:, 1])
        mi = RectBivariateSpline(mla, mlo, md[:, 2].reshape(len(mla), len(mlo)), kx=1, ky=1)
        print(f"Moho: {moho}")
    tect = xs.PolygonIndex.from_npz(tect_npz, class_key='names') if tect_npz else None
    bset = xs.BoundarySet.from_npz(bnd_npz) if bnd_npz else None
    secs = []
    for (la1, lo1, la2, lo2, lab) in sections:
        plat, plon, dist = px._great_circle_path(la1, lo1, la2, lo2, a.ds)
        v = px._sample_profile(Lon, Lat, V, plon, plat)
        S = dict(label=lab, ends=np.array([la1, lo1, la2, lo2]), lon=plon, lat=plat, dist=dist,
                 vsv=v, dvsv=(v - vmean[None, :]) / vmean[None, :] * 100)
        if has_xi:
            x = px._sample_profile(Lon, Lat, X, plon, plat)
            S['xi'], S['dxi'] = x, (x / xref[None, :] - 1.0) * 100
        S['moho'] = (np.array([float(np.ravel(mi(p, q))[0]) for p, q in zip(plat, plon)])
                     if mi is not None else np.full(len(dist), np.nan))
        lon_f, lat_f, d_f = xs.resample_profile(plon, plat, dist, 0.5)
        runs = xs.runs_from_samples(d_f, tect.sample(lon_f, lat_f), tect) if tect else []
        runs = [r for r in runs if r.name]
        S['runs'] = dict(d0=np.array([r.d0 for r in runs]), d1=np.array([r.d1 for r in runs]),
                         rgb=np.array([rgb(r.color) for r in runs]).reshape(-1, 3),
                         name=objarr([r.name for r in runs]))
        cr = bset.crossings(lon_f, lat_f, d_f, dedupe_km=3.0) if bset else []
        S['cross'] = dict(dist=np.array([c.dist for c in cr]),
                          crustal=np.array([c.scale == 'crustal' for c in cr], float),
                          names=objarr([' / '.join(c.names) for c in cr]))
        secs.append(S)
        print(f"  {lab}: {dist[-1]:.0f} km, {len(runs)} domain runs, {len(cr)} boundary crossings")
    groups, longest = px._split_by_orientation(sections)
    order = {s[4]: i + 1 for i, s in enumerate(sections)}          # MATLAB 1-based
    M['xsec'] = objarr(secs)
    M['z'] = z
    M['groups'] = dict(ns=np.array([order[s[4]] for s in groups['ns'][0]], float),
                       ew=np.array([order[s[4]] for s in groups['ew'][0]], float),
                       longest_km=longest)
    M['vname'] = vname
    M['model'] = os.path.basename(a.fvs)

    # --- colour maps and field settings (as in plot_xsection.py --stack) -------------------
    nc = a.ncolors
    M['cmaps'] = dict(abs=px._cmap(px.CMAP_ABS, nc)(np.linspace(0, 1, nc))[:, :3],
                      rel=px._cmap('RdBu', nc)(np.linspace(0, 1, px._cmap('RdBu', nc).N))[:, :3])
    M['fields'] = dict(
        vsv=dict(cmap='abs', clim=np.array([np.nan, np.nan]), sigma=1.5, label=f'{vname} (km/s)',
                 title=vname.upper()),
        dvsv=dict(cmap='rel', clim=np.array([-6.0, 6.0]), sigma=0.0, label=f'd{vname} (%)',
                  title='DVSV'),
        xi=dict(cmap='rel', clim=np.array([0.90, 1.10]), sigma=0.0, label='xi', title='XI'),
        dxi=dict(cmap='rel', clim=np.array([-5.0, 5.0]), sigma=0.0,
                 label=f'dlnXi (%) vs {px._xi_ref_label(a.xi_ref)}', title='dlnXi'))
    M['layout'] = dict(page_w_in=9.5, ve=float(px.DEFAULT_VE), strip_h_in=0.14, strip_gap_in=0.03)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    savemat(a.out, M, do_compression=True, long_field_names=True)
    print(f"Saved: {a.out}")


if __name__ == '__main__':
    main()
