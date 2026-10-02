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

## Relation to noise_asdf
The inversion pipeline (noise_asdf) produces the Fvs npz this toolkit plots.
make_Fvs.py / read_posterior.py stay in noise_asdf; this repo takes the Fvs npz
as input. Keep them loosely coupled through that interface.
