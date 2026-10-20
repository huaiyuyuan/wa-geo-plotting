#!/usr/bin/env python3
"""
plot_wa_map.py - standalone WA map plotter (profile locator / station map).

The map figure that was embedded in plot_xsection.py (_plot_index_map) lives here now,
as plot_map(). It draws the GSWA 2022 tectonic basemap, major crustal boundaries, unit
labels/legend, profile lines (straight or along picked polylines with distance ticks),
station layers (WA Array + arbitrary CSV overlays), and the array footprint.

It can be driven two ways:
  - called by plot_xsection.py --index-map (passing the Vs model d for the extent), or
  - standalone (its own CLI below): a WA map with stations / domains / boundaries /
    profile lines and an explicit --extent, with NO Vs model required.

Shared geometry/geology helpers (_great_circle_path, _polyline_path, _load_geology,
_resolve_npz, _load_stations, _index_footprint) stay in plot_xsection and are imported
lazily to avoid a circular import (plot_xsection imports this module only inside its
--index-map branch).
"""
import os, sys
import numpy as np
import matplotlib


def _load_station_csv(path):
    """Station lon/lat from a CSV with named Latitude/Longitude columns (any order;
    e.g. the GSWA 'New.WA.Seismic' exports). Returns (lon, lat)."""
    import csv
    lon, lat = [], []
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = next(r, [])
        hl = [h.strip().lower() for h in header]
        def find(*names):
            for n in names:
                if n in hl:
                    return hl.index(n)
            return None
        ila = find('latitude', 'lat')
        ilo = find('longitude', 'lon', 'long')
        if ila is None or ilo is None:
            raise SystemExit(f"{path}: need Latitude/Longitude columns; header was {header}")
        for row in r:
            if len(row) <= max(ila, ilo):
                continue
            try:
                lat.append(float(row[ila])); lon.append(float(row[ilo]))
            except ValueError:
                continue
    return np.array(lon), np.array(lat)



def plot_map(sections, out_dir, d=None, extent=None, label='sections', moho=None,
             litho_npz=None, tick_km=100, stations=None, style='map2022',
             footprint=None, footprint_veil=0.5, tect_npz=None, bnd_npz=None,
             projection='albers', legend='upper left', labels='major',
             label_scale=1.4, polylines=None, overlays=None, geo=None):
    """Profile lines / station map on the WA tectonic basemap, distance ticks + labels
    every tick_km. style: map2022 = 10M units in the GSWA 2022 simplified tectonic map
    colours + major crustal boundaries; gswa = 500k units in TECTCOLOUR; craton =
    domains + Yilgarn/Pilbara granite-greenstone from litho_npz; plain = domains.

    extent: (lon0, lon1, lat0, lat1) sets the frame. If None it is taken from the Vs
    model d (if given) else from the sections/stations/overlays. d is used only for the
    extent and, in the no-tectonic fallback, to show the model nodes; it is optional, so
    a map can be drawn with no Vs model at all. geo: a loaded geology dict (as from
    plot_xsection._load_geology) for the non-map2022 boundary overlay; default None."""
    import matplotlib.pyplot as plt
    import plot_xsection as px            # lazy: reuse shared geometry/geology helpers
    if geo is None:
        geo = getattr(px, '_GEO', None)
    # points used for the extent (and the fallback node scatter when d is given)
    if d is not None:
        Lon = np.asarray(d['Lon'], float); Lat = np.asarray(d['Lat'], float)
    else:
        xs, ys = [], []
        for (la1, lo1, la2, lo2, _l) in sections:
            xs += [lo1, lo2]; ys += [la1, la2]
        if stations is not None:
            xs += list(stations[0]); ys += list(stations[1])
        for _ov in (overlays or []):
            xs += list(_ov[0]); ys += list(_ov[1])
        if not xs:
            sys.exit("plot_map: nothing to set the extent from (give d, extent, "
                     "sections or stations)")
        Lon = np.asarray(xs, float); Lat = np.asarray(ys, float)
    if extent is not None:
        ext_lo0, ext_lo1, ext_la0, ext_la1 = extent
    else:
        ext_lo0, ext_lo1 = Lon.min() - 0.5, Lon.max() + 0.5
        ext_la0, ext_la1 = Lat.min() - 0.5, Lat.max() + 0.5
    try:
        from plot_depth_slice import _add_states_ocean, _HAS_CARTOPY
        import cartopy.crs as ccrs
        if not _HAS_CARTOPY:
            raise ImportError('cartopy')
        fig = plt.figure(figsize=(13, 13))
        # albers = the projection of the GSWA 1:10M simplified tectonic map (2022):
        # Albers equal-area, central meridian 121E, standard parallels 17.5S / 31.5S
        # (fitted to the PDF graticule, 0.1 pt rms) - meridians converge, 129E tilts.
        proj = (ccrs.AlbersEqualArea(central_longitude=121.0, standard_parallels=(-17.5, -31.5))
                if projection == 'albers' else ccrs.PlateCarree())
        ax = fig.add_axes([0.06, 0.05, 0.88, 0.90], projection=proj)
        ax.set_facecolor('white' if style == 'map2022' else 'lightgrey')
        ax.coastlines('50m', linewidth=0.8, color='k', zorder=10)
        tr = ccrs.PlateCarree()
        _geo = True
    except Exception:
        fig, ax = plt.subplots(figsize=(11, 13))
        tr = None; _geo = False

    # Clip tectonic drawing to the WA data bounds (+margin) so the map auto-fits
    # to WA, not the full tectonic polygon extent (which reaches the NT/ocean).
    cb = (Lon.min()-1.0, Lon.max()+1.0, Lat.min()-1.0, Lat.max()+1.0)
    try:
        from wa_basemap import (add_tectonic_background, add_tectonic_outlines,
                                tectonic_legend, add_craton_geology,
                                add_gswa_units, has_gswa_colours)
        if style == 'gswa' and not (litho_npz and has_gswa_colours(litho_npz)):
            print("  (index map: lithology npz has no GSWA colours - re-run "
                  "extract/make_litho.py; using --index-style craton)")
            style = 'craton'
        litho_drawn = None
        if style == 'map2022':
            from wa_basemap import add_map2022_units, add_crustal_boundaries
            add_map2022_units(ax, tect_npz, transform=tr, clip_box=cb, zorder=0)
            if labels and labels != 'none':
                from wa_basemap import add_map2022_labels
                nl = add_map2022_labels(ax, transform=tr, level=1 if labels == 'major' else 2,
                                        scale=label_scale, clip_box=cb)
                print(f"  unit labels: {nl} ({labels})")
            try:
                add_crustal_boundaries(ax, bnd_npz, transform=tr, clip_box=cb, zorder=4.2,
                                       lw_litho=1.1, lw_crust=0.5, alpha=0.85)
            except Exception as e:
                print(f"  (crustal boundaries unavailable: {e})")
        else:
            add_tectonic_background(ax, alpha=0.5, zorder=0, transform=tr, clip_box=cb)
        if style == 'gswa':
            add_gswa_units(ax, litho_npz, transform=tr, clip_box=cb, zorder=1)
        elif style == 'craton' and litho_npz:
            litho_drawn = add_craton_geology(ax, litho_npz, zorder=1, transform=tr,
                                             clip_box=cb)
        if style != 'map2022':
            add_tectonic_outlines(ax, lw=0.4, alpha=0.6, zorder=3, major_only=True,
                                  transform=tr, clip_box=cb)
        _tect = True
    except Exception as e:
        print(f"(tectonic bg unavailable: {e})")
        if d is not None:                 # show the model nodes as a stand-in backdrop
            ax.scatter(Lon, Lat, c='0.7', s=10, zorder=1,
                       **({'transform': tr} if tr else {}))
        _tect = False

    lkw = {'transform': tr} if tr else {}
    xyc = tr._as_mpl_transform(ax) if tr else 'data'     # annotate in lon/lat on any projection
    if footprint is not None:          # outline + veil, no zoom (sections may extend past it)
        footprint.focus(ax, tr, veil_alpha=footprint_veil, zoom=False,
                        veil_zorder=4.4)     # over the geology, under stations + sections
    n_sta = 0
    if stations is not None:            # above the ocean mask (5), below section lines
        slon, slat = stations
        n_sta = len(slon)
        ax.scatter(slon, slat, marker='s', s=16, c='red', edgecolors='w',
                   linewidths=0.35, zorder=5.5, **lkw)
    for ov in (overlays or []):         # extra station layers (e.g. phase 4, permanent)
        olon, olat, ocol, omark, osize, _olabel = ov
        ax.scatter(olon, olat, marker=omark, s=osize, c=ocol, edgecolors='w',
                   linewidths=0.4, zorder=5.6, **lkw)
    if style != 'map2022' and geo is not None and geo['bnd'] is not None:
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(geo['bnd'].lines, colors='firebrick',
                                         linewidths=0.9, alpha=0.8, zorder=4, **lkw))
    for i, (lat1, lon1, lat2, lon2, lab) in enumerate(sections):
        if polylines and lab in polylines:        # draw the true picked path, not a chord
            plo, pla = polylines[lab]
            plat, plon, dist = px._polyline_path(plo, pla)
        else:
            plat, plon, dist = px._great_circle_path(lat1, lon1, lat2, lon2)
        ax.plot(plon, plat, 'w-', lw=3.4, zorder=5, **lkw)   # white halo
        ax.plot(plon, plat, 'k-', lw=2.0, zorder=6, **lkw)
        ax.plot(plon[0], plat[0], 'ko', ms=6, zorder=7, **lkw)
        ax.plot(plon[-1], plat[-1], 'ks', ms=6, zorder=7, **lkw)
        for dkm in np.arange(0, dist[-1], tick_km):
            j = np.argmin(np.abs(dist - dkm))
            ax.plot(plon[j], plat[j], '|', color='k', ms=7, mew=1.2, zorder=7, **lkw)
            ax.annotate(f'{int(dkm)}', (plon[j], plat[j]), fontsize=5.5, xycoords=xyc,
                        ha='center', va='bottom', zorder=8,
                        xytext=(0, 3), textcoords='offset points',
                        bbox=dict(fc='w', ec='none', alpha=0.6, pad=0.3))
        import matplotlib.patheffects as _pe
        ax.annotate(lab, (plon[0], plat[0]), fontsize=9, fontweight='bold', xycoords=xyc,
                    color='red', zorder=8, xytext=(5,5),
                    textcoords='offset points',
                    path_effects=[_pe.withStroke(linewidth=1.6, foreground='white')])

    # Grey ocean mask + coastline on top (like the depth-slice panels)
    if _geo:
        if style == 'map2022':      # as on the GSWA map: white sea, other states pale grey
            _add_states_ocean(ax, ocean='white', states='#e8e8e8')
        else:
            _add_states_ocean(ax)
        ax.set_extent([ext_lo0, ext_lo1, ext_la0, ext_la1], crs=tr)
        # graticule as on the GSWA map: 5 deg, thin grey, labels '115°' / '20°' bottom + left
        import matplotlib.ticker as mticker
        gcol = '#858c8c'
        glon, glat = np.arange(100, 151, 5), np.arange(-45, 1, 5)
        # lines drawn as our own polylines above the sea and land masks (cartopy's
        # gridliner draws at its own zorder, under the ocean); gridliner only for labels
        lat_d, lon_d = np.linspace(-50, 5, 221), np.linspace(95, 155, 241)
        for lo in glon:
            ax.plot(np.full_like(lat_d, lo), lat_d, color=gcol, lw=0.35, zorder=5.2, transform=tr)
        for la in glat:
            ax.plot(lon_d, np.full_like(lon_d, la), color=gcol, lw=0.35, zorder=5.2, transform=tr)
        gl = ax.gridlines(crs=tr, draw_labels=True, xlocs=glon, ylocs=glat,
                          color='none', linewidth=0)
        gl.top_labels = gl.right_labels = False
        gl.x_inline = gl.y_inline = False
        gl.rotate_labels = False
        gl.xformatter = mticker.FuncFormatter(lambda v, pos=None: f'{v:g}\u00b0')
        gl.yformatter = mticker.FuncFormatter(lambda v, pos=None: f'{abs(v):g}\u00b0')
        gl.xlabel_style = gl.ylabel_style = {'size': 8, 'color': gcol}
    else:
        ax.set_xlim(ext_lo0, ext_lo1)
        ax.set_ylim(ext_la0, ext_la1); ax.set_aspect('equal')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(f'Cross-section locations ({len(sections)} lines, ticks every {tick_km:g} km'
                 + (f'; {n_sta} stations)' if n_sta else ')'),
                 fontsize=11)
    if _tect and style == 'map2022':
        from wa_basemap import boundaries_legend   # credit now in the map-legend title
        boundaries_legend(ax, loc='lower left', fontsize=7, lw_litho=1.1, lw_crust=0.5)
        if legend and legend != 'none':
            from wa_basemap import add_map2022_legend
            add_map2022_legend(ax, loc=legend)
    elif _tect and style == 'gswa':
        ax.text(0.01, 0.99, 'Geology: GSWA 1:500k tectonic units, GSWA colours\n'
                '(\u00a9 Geological Survey of Western Australia, CC-BY-4.0)',
                transform=ax.transAxes, ha='left', va='top', fontsize=7, zorder=12,
                bbox=dict(fc='w', ec='0.6', alpha=0.9, pad=3))
    elif _tect:
        tectonic_legend(ax, loc='upper left', fontsize=7,
                        clip_box=(ext_lo0, ext_lo1, ext_la0, ext_la1),
                        exclude=('STATE',),
                        litho=litho_drawn)
    # station-type legend (WA Array + any overlay layers), as a separate legend
    if stations is not None or overlays:
        from matplotlib.lines import Line2D
        from matplotlib.legend import Legend
        sh = []
        if stations is not None:
            sh.append(Line2D([], [], marker='s', color='none', markerfacecolor='red',
                             markeredgecolor='w', markersize=7, label='WA Array'))
        for ov in (overlays or []):
            sh.append(Line2D([], [], marker=ov[3], color='none', markerfacecolor=ov[2],
                             markeredgecolor='w', markersize=8, label=ov[5]))
        if sh:
            leg_s = Legend(ax, sh, [h.get_label() for h in sh], loc='upper right',
                           fontsize=7, framealpha=0.9, title='Stations', title_fontsize=7)
            ax.add_artist(leg_s)
    import os
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'xsection_index_map' +
                       ('.fp' if footprint is not None else '') + '.png')   # whole-WA name unchanged
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.png','.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"Index map: {out}")




# ── standalone CLI ────────────────────────────────────────────────────────────
def main():
    import argparse
    import plot_xsection as px
    ap = argparse.ArgumentParser(
        description="Standalone WA map: profile lines and station layers on the GSWA "
                    "tectonic basemap. No Vs model needed (extent from --extent or the "
                    "sections/stations).")
    ap.add_argument('--sections', default=None, help='sections.txt (lat1 lon1 lat2 lon2 label)')
    ap.add_argument('--load-polylines', default=None, help='polylines npz (draw true picked paths)')
    ap.add_argument('--extent', nargs=4, type=float, default=None,
                    metavar=('LON0', 'LON1', 'LAT0', 'LAT1'), help='map frame')
    ap.add_argument('--stations', default=None, help='WA Array stations file (code lat lon)')
    ap.add_argument('--no-stations', action='store_true')
    ap.add_argument('--map-stations', nargs=4, action='append', default=[],
                    metavar=('CSV', 'COLOUR', 'MARKER', 'LABEL'),
                    help='extra station layer from a CSV with Latitude/Longitude columns '
                         '(repeatable), e.g. --map-stations phase4.csv lime s "Phase 4"')
    ap.add_argument('--footprint', action='store_true')
    ap.add_argument('--footprint-npz', default=None)
    ap.add_argument('--footprint-outline', default=None)
    ap.add_argument('--footprint-veil', type=float, default=0.5)
    ap.add_argument('--moho', default=None)
    ap.add_argument('--style', dest='index_style', default='map2022',
                    help='map2022 | gswa | craton | plain')
    ap.add_argument('--tick', dest='index_tick', type=float, default=100)
    ap.add_argument('--labels', dest='index_labels', default='major',
                    choices=['major', 'all', 'none'])
    ap.add_argument('--label-scale', type=float, default=1.4)
    ap.add_argument('--legend', dest='index_legend', default='upper left')
    ap.add_argument('--proj', dest='index_proj', default='albers', choices=['albers', 'plate'])
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--bnd-npz', default=None)
    ap.add_argument('--litho-npz', default=None)
    ap.add_argument('--out-dir', default='.')
    args = ap.parse_args()

    sections = px._load_sections(args.sections) if args.sections else []
    poly = None
    if args.load_polylines:
        p = np.load(args.load_polylines, allow_pickle=True)
        poly = {str(p['short'][i]): (np.asarray(p['lon'][i], float),
                                     np.asarray(p['lat'][i], float))
                for i in range(len(p['short']))}
    sta = None
    if not args.no_stations:                 # default to config.STATIONS, as --index-map does
        try:
            import config
            sta_path, cols = args.stations or config.STATIONS, config.STATION_COLS
        except Exception:
            sta_path, cols = args.stations, (2, 1)
        if sta_path and os.path.isfile(sta_path):
            sta = px._load_stations(sta_path, cols)
            print(f"  stations: {len(sta[0])} <- {sta_path}")
        else:
            print(f"  stations: file not found ({sta_path}) - none drawn "
                  f"(pass --stations, or --no-stations to silence)")
    overlays = []
    for csv_path, col, mark, lab in args.map_stations:
        if not os.path.isfile(csv_path):
            print(f"  map-stations: file not found ({csv_path}) - skipped"); continue
        olon, olat = _load_station_csv(csv_path)
        overlays.append((olon, olat, col, mark, 22, lab))
        print(f"  map-stations: {len(olon)} '{lab}' <- {csv_path}")
    geo = None
    if args.index_style != 'map2022':
        try:
            geo = px._load_geology('domain,boundaries,names',
                                   args.litho_npz, args.tect_npz, args.bnd_npz)
        except Exception as e:
            print(f"  geology unavailable ({e})")
    fp = None
    if args.footprint:
        try:
            fp = px._index_footprint(args)
        except Exception as e:
            print(f"  footprint unavailable ({e})")

    if not sections and sta is None and not overlays and args.extent is None:
        ap.error("give something to draw: --sections, --stations, --map-stations, or --extent")

    plot_map(sections, args.out_dir, d=None,
             extent=tuple(args.extent) if args.extent else None,
             moho=args.moho,
             litho_npz=(None if args.index_style == 'plain' else
                        px._resolve_npz('LITHOLOGY_NPZ', 'wa_lithology.npz', args.litho_npz)),
             tick_km=args.index_tick, stations=sta, style=args.index_style,
             footprint=fp, footprint_veil=args.footprint_veil,
             tect_npz=px._resolve_npz('TECTONICS_NPZ', 'wa_tectonics.npz', args.tect_npz),
             bnd_npz=px._resolve_npz('BOUNDARIES_NPZ', 'wa_crustal_boundaries.npz', args.bnd_npz),
             projection=args.index_proj, legend=args.index_legend, labels=args.index_labels,
             label_scale=args.label_scale, polylines=poly, overlays=overlays, geo=geo)


if __name__ == '__main__':
    main()
