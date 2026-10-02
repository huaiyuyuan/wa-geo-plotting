# wa-geo-plotting — handoff for the repo-split conversation

## Goal
Split the WA-specific plotting/basemap toolkit out of `noise_asdf` into its own repo
`wa-geo-plotting`. It plots TransD Vs/xi models (Fvs npz) with GSWA geology overlays.

## What's here (this skeleton)
- `config.py` — ALL local paths centralised (shapefiles, Moho, stations, derived npz).
  The WA/machine specificity is isolated here; plotting code reads from config.
- `extract/` — pure-Python shapefile→npz converters (no geopandas needed):
  - make_tectonic_basemap.py (10M units), make_litho.py (500k lithology),
    make_boundaries.py (crustal boundary lines)
- `basemap/wa_basemap.py` — overlay functions (tectonic fill/outlines, lithology, boundaries)
- `plotting/` — plot_fig1.py, plot_xsection.py (copied from noise_asdf stage6.1_transd_prep/desktop)

## TODO in the new conversation
1. **Create the GitHub repo** `wa-geo-plotting` (private; mostly personal use).
   Licence note: GSWA shapefiles are CC-BY-4.0 — attribution required.
2. **Refactor imports** — plotting scripts currently `from plot_depth_slice import ...`
   and hardcode paths. Point them at config.py instead. plot_xsection.py imports
   `_make_ax/_add_states_ocean` from plot_depth_slice — bring plot_depth_slice.py over too
   (copy from noise_asdf/stage6.1_transd_prep/desktop/plot_depth_slice.py; the deployed
   version is the _v2 content).
3. **Bring the remaining plotters** from noise_asdf/stage6.1_transd_prep/desktop/:
   plot_depth_slice.py, plot_vsv_tectonic.py, plot_depth_panels.py, wa_basemap.py
   (newest versions). Decide: copy (fork) or git-subtree.
4. **Regenerate the npz caches** into data/ via the extract/ scripts (config paths).
   The lithology geojson is 139 MB — gitignored; regenerate from shapefile.
5. **Build the NEW feature** (the immediate science ask, not yet done):
   **lithology + crustal-boundary strips ON the cross-sections** — for each section,
   sample surface lithology + mark terrane-boundary crossings along the profile, drawn
   as a colored strip above each Vs panel. This proves the shallow fast-V bands =
   greenstone belts. Design:
     - sample lithology polygon at each great-circle point (point-in-polygon)
     - draw a thin colour bar above the top panel (lithology colours)
     - vertical dashed lines where MajorCrustalBoundaries cross the profile
   (make_boundaries.py produces the boundary lines; need a great-circle ∩ polyline test)
6. **config.py env-var support** is already in; document it in README.

## Key facts to carry
- Fvs npz keys: Lon, Lat, z, Vsv, Xi, Vpvs, Pani, Misfit, Ndat, Vsv_err, Xi_err,
  Vpvs_err, Vsv_med, dlnVsv, ... (from make_Fvs --with-posterior)
- Lithology npz: rings (object array of Nx2), colors, parents(=LITHOLOGY), 
  parent_color_keys/vals. Velocity-oriented colours (mafic/greenstone=green, granite=pink).
- Tectonic npz: same structure, parents=PARENTNAME (domains).
- Boundaries npz: lines (object array of Nx2 polylines).
- Stations: code lat lon (stations_v2.txt, 913 stations, whitespace; cols lon=2 lat=1).
- GSWA data © GSWA, CC-BY-4.0.
- Shapefile sources: config.SHP_TECTONICS / SHP_LITHOLOGY / SHP_BOUNDARIES.
- Moho: AR23-moho-hmp.txt, skip 11 header lines.

## Current state of plot_xsection.py (most-developed script)
- per-section mode: Vsv/Viso | dlnVsv(vs model mean) | [xi if ZT] | uncertainty,
  Moho overlay, 100km ticks, longest=2:1-ish, each section own width.
- --stack FIELD: all sections one page, same height, width∝length, --stack-ve,
  --vmin/--vmax, per-panel colorbars, --d-max.
- --ginput picker (tectonic bg), --index-map (WA-scaled, legend, distance ticks),
  --save-sections/--load-sections, --nproc parallel, fit-once interpolation.
