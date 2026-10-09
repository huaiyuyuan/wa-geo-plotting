# wa-geo-plotting cheatsheet

Every figure we make for the stage-6 TransD models: where the inputs are, where the
figures go, and the exact command. Run everything on d204403 from the repo root.

```bash
source /data/25/yuan/github/wa-geo-plotting/env_stage6.sh    # sets the variables below, cd's to the repo
```

---------------------------------------------------------------------------------------
## 1. Paths

### Variables (from `env_stage6.sh`)
| Variable | Path | What |
|---|---|---|
| `WGP` | `/data/25/yuan/github/wa-geo-plotting` | this repo |
| `ST` | `/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/stage6` | stage-6 models |
| `FIG` | `$ST/figures` | all figures, one folder per model |
| `XS` | `$FIG/xsections/sections.txt` | section lines (NS1–NS7, EW1–EW7); older lists = `sections.txt.bak.<time>` |
| `MOHO` | `/workspace/Others.data/AusMoho2023/AR23-moho-hmp.txt` | AusMoho 2023 (AR23) |

### Models (`$ST/Fvs.iter.N.<Z|ZT>.npz`)
| Var | File | Content | Vp/Vs |
|---|---|---|---|
| `FVS2` | `Fvs.iter.2.Z.npz` | Rayleigh-only Vsv, with Moho | fixed 1.73 |
| `FVS8` | `Fvs.iter.8.ZT.npz` | **final radial anisotropy**: Vsv pinned to iter2, xi inverted; `Vsv` array = Viso → use `--vel vsv_true` | fixed 1.73 |
| `FVS10` | `Fvs.iter.10.Z.npz` | Vsv, no Moho | fixed 1.73 |
| `FVS11` | `Fvs.iter.11.Z.npz` | Vsv + **Vp/Vs inverted** | 1.668–1.881 |
| | `Fvs.iter.1/3/5/6.*` | earlier runs (iter3, iter6 superseded by iter8) | 5, 6 vary |

Check what varies in any model:
```bash
for f in $ST/Fvs*.npz; do python3 -c "
import numpy as np,sys; d=np.load(sys.argv[1])
r=lambda k: f'{np.nanmin(d[k]):.3f}-{np.nanmax(d[k]):.3f}' if k in d.files else '-'
print(sys.argv[1].split('/')[-1], 'Xi', r('Xi'), 'Vpvs', r('Vpvs'))" $f; done
```

### Other inputs (set in `config.py`)
| What | Path |
|---|---|
| stations (912) | `/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/pathqc.v2/stations_v2.txt` |
| array footprint | `/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/footprint/wa_array_footprint.npz`, `wa_array_outline.txt` |
| GSWA basemap caches | `/workspace/shape.files/derived/wa_tectonics.npz`, `wa_lithology.npz`, `wa_crustal_boundaries.npz` |
| GSWA shapefiles | `/workspace/shape.files/` (10M tectonic units, 500k tectonic units, Major Crustal Boundaries 2025) |
| 2022 map colours / labels / legend | `basemap/tectonic_map_2022_colours.csv`, `tectonic_map_2022_labels.csv`, `tectonic_map_2022_legend.json` |

### Output folders and file names
| Figure | Folder | File |
|---|---|---|
| profile-location (index) map | `$FIG/<iter>/` | `xsection_index_map.png` (whole WA), `xsection_index_map.fp.png` (footprint) |
| one section | `$FIG/<iter>/` | `xsection.<NS1…EW7>.png` |
| stacked sections | `$FIG/<iter>/` | `xsection_stack.<field>.ns.png`, `.ew.png`; side by side: `xsection_stack.vsv_vpvs.ns.png` |
| depth-slice maps | `$FIG/<iter>/` | `<iter>.<vsv|viso|xi|vpvs>.<abs|rel>.<fp|wa>.png` |
| model vs model, sections | `$FIG/compare_<A>_<B>/` | `compare.<section>.png` |
| model vs model, maps | `$FIG/compare_<A>_<B>/` | `compare.<vsv|xi>.<abs|rel>.<fp|wa>.png` |
| regional 1-D profile | `$FIG/<iter>/` | `regional_profile.png` (+ `.txt` table) |

Every PNG has a `.pdf` beside it. `fp` = array-footprint view, `wa` = whole WA.

---------------------------------------------------------------------------------------
## 2. Conventions (defaults)
- Absolute values: continuous **Spectral** (warm = slow). Relative values and xi: **RdBu** with a
  small white centre and gentle transition (`--div-gap 0.07 --div-white 2`). Vp/Vs: **viridis**.
- 16 colour levels; cross-sections VE 3, depth 0–60 km.
- Depth slices at **5 / 15 / 30 / 45 km** (not 40: Moho artefact).
- xi models: plot true Vsv with `--vel vsv_true` (Vsv = Viso / sqrt((2 + xi²)/3)).
- Section strip: 10M tectonic units in the GSWA 2022 map colours + major crustal boundaries
  as down-arrows (thick = lithospheric, thin = crustal), names above.
- Map style: Albers equal-area (121°E, 17.5°S/31.5°S) as the GSWA 2022 map; white sea.

---------------------------------------------------------------------------------------
## 3. Profile-location map and section lines

```bash
# map with the current lines (whole WA; add --footprint for the array view)
python3 plotting/plot_xsection.py --fvs $FVS8 --load-sections $XS \
    --index-map --index-only --out-dir $FIG/iter8
python3 plotting/plot_xsection.py --fvs $FVS8 --load-sections $XS \
    --index-map --index-only --footprint --out-dir $FIG/iter8

# pick NEW lines on the map (needs a display; current lines shown dashed).
# left click start/end, right click undo, Enter save, Esc quit.
# saved over $XS (old file kept as $XS.bak.<time>), named NS1.. (W->E) / EW1.. (N->S)
python3 plotting/plot_xsection.py --fvs $FVS8 --load-sections $XS --ginput \
    --save-sections $XS --index-map --index-only --out-dir $FIG/iter8
```
Options: `--index-labels major|all|none`, `--label-scale 1.4`, `--index-legend "upper left"|…|none`,
`--index-proj albers|plate`, `--index-style map2022|gswa|craton|plain`, `--no-stations`,
`--index-tick 100`.

---------------------------------------------------------------------------------------
## 4. Cross-sections

### 4a. Stacked pages, north-south and west-east (one page each)
```bash
# iter8 (xi model): Vsv, dVsv, xi, dlnXi - no Moho line
for f in vsv dvsv; do
  python3 plotting/plot_xsection.py --fvs $FVS8 --vel vsv_true --load-sections $XS \
      --stack $f --out-dir $FIG/iter8
done
for f in xi dxi; do
  python3 plotting/plot_xsection.py --fvs $FVS8 --load-sections $XS --stack $f --out-dir $FIG/iter8
done

# iter11: Vsv and Vp/Vs side by side (and the relative pair)
python3 plotting/plot_xsection.py --fvs $FVS11 --load-sections $XS --stack vsv,vpvs --out-dir $FIG/iter11
python3 plotting/plot_xsection.py --fvs $FVS11 --load-sections $XS --stack dvsv,dvpvs --out-dir $FIG/iter11

# iter10 (no Moho model)
for f in vsv dvsv; do
  python3 plotting/plot_xsection.py --fvs $FVS10 --load-sections $XS --stack $f --out-dir $FIG/iter10
done
```
Fields: `vsv dvsv xi dxi vpvs dvpvs vdiff err` (comma list = columns side by side).
Add `--moho $MOHO` for the AR23 Moho (dashed) with a grey veil below (`--no-moho-mask` for the line only).
Fix a colour range across runs: `--vmin 3.2 --vmax 4.0`. Taller/flatter rows: `--stack-ve 4`.
All on one page: `--stack-split none`. Model minus reference: `--stack vdiff --ref-fvs $FVS2`.

### 4b. One figure per section
```bash
# iter8: true Vsv, dVsv, xi and dlnXi rows, with the Moho
python3 plotting/plot_xsection.py --fvs $FVS8 --vel vsv_true --load-sections $XS \
    --moho $MOHO --xi-panel both --out-dir $FIG/iter8 --nproc 8
# Z models (iter10, iter11)
python3 plotting/plot_xsection.py --fvs $FVS10 --load-sections $XS --out-dir $FIG/iter10 --nproc 8
```
Strips: default `--strips domain,boundaries,names`; add the 500k lithology strip with
`--strips litho,domain,boundaries,names`; dashed boundary lines through the panels `--bnd-lines`;
none `--no-strips`.

### 4c. Two models side by side, per section
```bash
python3 plotting/plot_xsection_compare.py \
    --models $FVS10:"iter10 (no Moho)" $FVS2:"iter2 (Moho)" \
    --load-sections $XS --moho $MOHO --d-max 60 --diff \
    --out-dir $FIG/compare_iter10_iter2 --nproc 4
# pinning check: iter8 true Vsv vs iter2 (should be ~0)
python3 plotting/plot_xsection_compare.py \
    --models $FVS8:"iter8" $FVS2:"iter2" --load-sections $XS --diff --clim-diff 1 \
    --out-dir $FIG/compare_iter8_iter2 --nproc 4
```
Model spec `PATH[:LABEL[:VEL]]`; VEL `auto` (default) turns an xi model into true Vsv.

---------------------------------------------------------------------------------------
## 5. Depth-slice maps (5 / 15 / 30 / 45 km)

```bash
# iter8 (xi): Vsv (true), Viso, xi - abs + rel; footprint view, then whole WA
python3 plotting/make_depth_maps.py --fvs-zt $FVS8 --tag-zt iter8 \
    --depths 5 15 30 45 --modes abs rel --out-dir $FIG/iter8
python3 plotting/make_depth_maps.py --fvs-zt $FVS8 --tag-zt iter8 \
    --depths 5 15 30 45 --modes abs rel --no-footprint --out-dir $FIG/iter8

# iter10 (Vsv) and iter11 (Vsv + Vp/Vs, picked up automatically)
for it in 10 11; do
  for ext in "" "--no-footprint"; do
    python3 plotting/make_depth_maps.py --fvs-z $ST/Fvs.iter.$it.Z.npz --tag-z iter$it \
        --depths 5 15 30 45 --modes abs rel $ext --out-dir $FIG/iter$it
  done
done
```
Overlay: `--overlay boundaries` (default) `| outlines | both | filled | none`.
One map by hand: `plotting/plot_vsv_tectonic.py --fvs … --field vsv|vsv_true|xi|vpvs --mode abs|rel
--depths 5 15 30 45 --footprint --out file.png`.

---------------------------------------------------------------------------------------
## 6. Model vs model maps (A | B | A−B per depth)

```bash
for m in abs rel; do
  python3 plotting/plot_map_compare.py --models $FVS10:"iter10 (no Moho)" $FVS2:"iter2 (Moho)" \
      --mode $m --out-dir $FIG/compare_iter10_iter2
  python3 plotting/plot_map_compare.py --models $FVS10:"iter10 (no Moho)" $FVS2:"iter2 (Moho)" \
      --mode $m --no-footprint --out-dir $FIG/compare_iter10_iter2
done
# pinning check iter8 vs iter2 (1 % scale)
python3 plotting/plot_map_compare.py --models $FVS8:"iter8" $FVS2:"iter2" --mode abs \
    --clim-diff 1 --out-dir $FIG/compare_iter8_iter2
```
`--field xi` compares xi of two xi models; `--depths` default 5 15 30 45.

---------------------------------------------------------------------------------------
## 7. Regional 1-D profiles

```bash
python3 plotting/plot_regional_profile.py --fvs $FVS8 --moho $MOHO --d-max 60 \
    --out $FIG/iter8/regional_profile.png                  # Viso + xi
python3 plotting/plot_regional_profile.py --fvs $FVS11 --second vpvs --moho $MOHO --d-max 60 \
    --out $FIG/iter11/regional_profile.png                 # Vsv + Vp/Vs
```
`--footprint` = nodes inside the array footprint only; `--spread pct` = percentile bands.

---------------------------------------------------------------------------------------
## 8. Basemap caches (only when the GSWA data change)

```bash
python3 extract/make_tectonic_basemap.py --shp /workspace/shape.files/WA_Tectonic_Units_10M_2021/WA_Tectonic_Units_10M_2021.shp \
    --out-npz /workspace/shape.files/derived/wa_tectonics.npz
python3 extract/make_litho.py --shp /workspace/shape.files/GEOLOGY_500k_Tectonics_GDA2020_SHP/ESRI/SHAPEFILES/500k_tectonicp.shp \
    --out-npz /workspace/shape.files/derived/wa_lithology.npz
python3 extract/make_boundaries.py --shp /workspace/shape.files/Major.Crustal.Boundaries.WA.2025/MajorCrustalBoundaries.shp \
    --out-npz /workspace/shape.files/derived/wa_crustal_boundaries.npz
# legend + label positions from the GSWA 2022 map PDF (needs pdfplumber)
python3 extract/make_map2022_assets.py --pdf WA_TectonicsMap_10M_A4_Mar2022.pdf
```
Unit colours: edit `basemap/tectonic_map_2022_colours.csv` (`name,#rrggbb`; `name,none` hides a
concealed unit, e.g. Madura / Coompana under the Eucla Basin). Labels: `tectonic_map_2022_labels.csv`
(level 1 = major, 2 = minor; lon/lat/angle editable).

---------------------------------------------------------------------------------------
## 9. MATLAB redraw (optional)
```bash
python3 plotting/export_matlab.py --fvs $FVS8 --vel vsv_true --load-sections $XS --moho $MOHO \
    --out $FIG/iter8/wa_plotdata.mat
```
```matlab
addpath('/data/25/yuan/github/wa-geo-plotting/matlab')
wa_plot_map('wa_plotdata.mat');  wa_plot_stack('wa_plotdata.mat', 'dvsv')
```

---------------------------------------------------------------------------------------
## 10. Housekeeping and fixes
```bash
ls -lt $FIG/iter8 | head                                   # newest figures
find $FIG -name "*.png" -newermt "$(date +%F)" | sort      # made today
eog file.png      # or: gm display -font fixed file.png   (gm needs -font on this box)
python3 tests/test_xsection_strips.py && python3 tests/test_map2022_units.py
git pull                                                   # update the repo
```
| Symptom | Fix |
|---|---|
| `argument --fvs/--load-sections: expected one argument` | a variable is empty: `source env_stage6.sh` |
| `Vpvs is fixed at 1.7300 … skipped` | model does not invert Vp/Vs: use `$FVS11` |
| picker: `FigureCanvasAgg is non-interactive` | no display / toolkit: `ssh -Y`, or `conda install -n obspy tk` (or `tornado` for the browser picker) |
| labels look wide | Arial Narrow / Liberation Sans Narrow not installed (falls back to DejaVu) |
| `gm display: Unable to load font` | `gm display -font fixed file.png` or `eog file.png` |

---------------------------------------------------------------------------------------
## 11. Credits for figure captions
- Geological Survey of Western Australia 2022, 1:10 000 000 simplified tectonic map of Western
  Australia, March 2022: Geological Survey of Western Australia, Non-series map.
- Geological Survey of Western Australia 2020, 1:500 000 tectonic units of Western Australia,
  March 2020 update: Geological Survey of Western Australia, digital data layer.
- GSWA Major Crustal Boundaries 2025, digital data layer.
- Caption: *"Domain colours after GSWA (2022); tectonic units and crustal boundaries © State of
  Western Australia (DEMIRS), CC BY 4.0. Moho: AusMoho 2023 (AR23)."*
