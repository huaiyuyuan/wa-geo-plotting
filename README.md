# wa-geo-plotting

WA-specific geoscience plotting + basemap toolkit for the WA Array seismic models
(Vs / radial anisotropy). Builds Fig-1 base maps, cross-sections, and depth-slice
maps with GSWA tectonic/lithology/crustal-boundary overlays.

**Mostly a personal toolkit (HY / GSWA).** It depends on local GSWA shapefiles and
the TransD inversion outputs (Fvs npz). Paths are centralised in `config.py` — edit
that for your machine. Others *can* use it with their own shapefiles + an Fvs-format npz.

## Layout
```
config.py              # ALL local paths (shapefiles, Moho, stations, derived npz) — edit this
extract/               # one-off: shapefile → npz basemap caches
    make_tectonic_basemap.py   # 10M tectonic units → wa_tectonics.npz (+ geojson)
    make_litho.py              # 500k lithology → wa_lithology.npz (+ geojson)
    make_boundaries.py         # Major Crustal Boundaries (lines) → wa_crustal_boundaries.npz
basemap/
    wa_basemap.py      # add_tectonic_background / _outlines, lithology, boundaries overlays
plotting/
    plot_fig1.py       # Fig 1: lithology/tectonic fill + boundaries + stations + sections
    plot_xsection.py   # cross-sections (per-section + stacked), geology strips
    (plot_depth_slice.py, plot_vsv_tectonic.py, plot_depth_panels.py — add from noise_asdf)
data/                  # small derived npz may be committed; large geojson is .gitignore'd
docs/
```

## Dependencies
- numpy, scipy, matplotlib, cartopy (for ocean mask / coastline)
- No geopandas/fiona/pyshp needed — pure-Python shapefile readers in extract/

## Setup
1. Edit `config.py` (or set env vars WA_SHAPEFILE_ROOT, WA_DERIVED, WA_MOHO, WA_STATIONS).
2. Build the basemap caches (once):
   ```
   python3 extract/make_tectonic_basemap.py --shp $SHP_TECTONICS \
       --out-npz data/wa_tectonics.npz --out-json data/wa_tectonics.geojson
   python3 extract/make_litho.py --shp $SHP_LITHOLOGY \
       --out-npz data/wa_lithology.npz --out-json data/wa_lithology.geojson
   python3 extract/make_boundaries.py --shp $SHP_BOUNDARIES \
       --out-npz data/wa_crustal_boundaries.npz
   ```
3. Set `WA_TECTONICS_NPZ` etc. (or point config.py at data/).

## Provenance / licence
GSWA shapefiles (tectonic units, 500k geology, crustal boundaries) are
**CC-BY-4.0 © Geological Survey of Western Australia** — attribute GSWA in any
figure or redistribution. The derived npz/geojson carry this in their metadata.

### References
- Geological Survey of Western Australia 2022, 1:10 000 000 simplified tectonic map of
  Western Australia, March 2022: Geological Survey of Western Australia, Non-series map,
  <https://geodocsget.dmirs.wa.gov.au/api/GeoDocsGet?filekey=b03c3f5a-84d5-4f26-8628-fb839d4bb609-s5axiawwyhlfngq2e2p61uc8x9wumauj1eg4q45f>.
  *Domain colours (`basemap/tectonic_map_2022_colours.csv`: strips, `--index-style map2022`).*
- Geological Survey of Western Australia 2020, 1:500 000 tectonic units of Western Australia,
  March 2020 update: Geological Survey of Western Australia, digital data layer,
  <www.dmirs.wa.gov.au/geoview>. *(500k units; cite the release you downloaded.)*
- GSWA Major Crustal Boundaries (2025) and 1:10 000 000 tectonic units digital data layers,
  as downloaded (DEMIRS Data and Software Centre) — boundary arrows/lines and 10M unit polygons.

Suggested caption credit: *"Domain colours after GSWA (2022); tectonic units and crustal
boundaries © State of Western Australia (DEMIRS), CC BY 4.0."*

## Profile location (index) map
```
python3 plotting/plot_xsection.py --fvs $FVS --load-sections $XS --index-map --index-only \
    [--footprint] --out-dir $OUT/figures/iterN
```
Default `--index-style map2022`: 10M units filled with the 2022 simplified-tectonic-map
colours, major crustal boundaries, stations, sections with distance ticks, drawn in the
map's own projection (`--index-proj albers`: Albers equal-area, central meridian 121°E,
standard parallels 17.5°S/31.5°S, fitted to the PDF graticule; `plate` = plain lon/lat).
The map's own legend (rock type x age chart) is inset top-left as vectors
(`--index-legend` corner or `none`); unit names sit at GSWA's label positions
(`--index-labels major|all|none`, `--label-scale`): cratons red bold, terranes red, orogens
grey letter-spaced, basins/provinces black. 5° graticule in the map's grey, labelled
bottom/left as on the PDF. Legend and labels come from the PDF via
`extract/make_map2022_assets.py` (needs pdfplumber; outputs
`basemap/tectonic_map_2022_legend.json`, `tectonic_map_2022_labels.csv` are committed).

## Stacked cross-sections
`plot_xsection.py --stack FIELD` puts all sections on one page per orientation:
`xsection_stack.<field>.ns` (north-south lines, ordered west to east) and `.ew` (west-east
lines, north to south), on one shared km/inch. `--stack-split none` = everything on one page.

## MATLAB version of the map and stacks
```
python3 plotting/export_matlab.py --fvs $FVS_ZT --vel vsv_true --load-sections $XS \
    --moho $MOHO --out $ST/figures/iter8/wa_plotdata.mat [--footprint] [--labels all]
```
Then in MATLAB, with `matlab/` on the path (base MATLAB, R2017b+; no toolboxes):
```
wa_plot_map('wa_plotdata.mat')            % xsection_index_map.png/.pdf
wa_plot_stack('wa_plotdata.mat', 'dvsv')  % xsection_stack.dvsv.ns/.ew png/pdf ('vsv','xi','dxi')
```
The .mat holds the drawn geometry and colours (units in the 2022 map colours, boundaries,
Natural Earth sea/states/coast, labels, the vector legend, stations, sections) and, per
section, the sampled model fields, Moho, domain-strip runs and boundary crossings, plus the
exact colour maps; `matlab/wa_albers.m` is the map projection.

## Relation to noise_asdf
The inversion pipeline (noise_asdf) produces the Fvs npz this toolkit plots.
make_Fvs.py / read_posterior.py stay in noise_asdf; this repo takes the Fvs npz
as input. Keep them loosely coupled through that interface.
