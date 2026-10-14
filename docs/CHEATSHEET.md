# wa-geo-plotting cheatsheet

Every figure we make for the stage-6 TransD models: where the inputs are, where the
figures go, and the exact command. Run everything on d204403 from the repo root.

```bash
source /data/25/yuan/github/wa-geo-plotting/env_stage6.sh    # sets the variables below, cd's to the repo
# other machines: export WA_DATA=~/wa_data first (section 8b)
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
| `FVS11` | `Fvs.iter.11.Z.npz` | setup 4: Vsv + **Vp/Vs inverted**, Rayleigh only; `Vsv` is pure Vsv; band80, provisional | 1.668–1.881 |
| `FVS12` | `Fvs.iter.12.ZT.npz` | Viso + xi (ZT); scripts derive Vsv = Viso/√((2+ξ²)/3) | fixed |
| (v3) | band120, 25 km, 1682 nodes | final publication Vsv (queued); same keys, point `--fvs` at it | |
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
| v3b surface-wave maps (`$V3B`) | `/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/v3b/{Z,T}/{group,phase}_v_tomo/TransD.region.{group,phase}.npz` |
| GSWA basemap caches | `/workspace/shape.files/derived/wa_tectonics.npz`, `wa_lithology.npz`, `wa_crustal_boundaries.npz` |
| GSWA shapefiles | `/workspace/shape.files/` (10M tectonic units, 500k tectonic units, Major Crustal Boundaries 2025) |
| 2022 map colours / labels / legend | `basemap/tectonic_map_2022_colours.csv`, `tectonic_map_2022_labels.csv`, `tectonic_map_2022_legend.json` |

### Output folders and file names
| Figure | Folder | File |
|---|---|---|
| profile-location (index) map | `$FIG/<iter>/` | `xsection_index_map.png` (whole WA), `xsection_index_map.fp.png` (footprint) |
| one section | `$FIG/<iter>/` | `xsection.<NS1…EW7>.png` |
| stacked sections | `$FIG/<iter>/` | `xsection_stack.<field>.ns.png`, `.ew.png`; side by side: `xsection_stack.vsv_vpvs.ns.png` |
| depth-slice maps | `$FIG/<iter>/` | `<iter>.<vsv|viso|xi|vpvs>.<abs|rel|rel-yilgarn>.<fp|wa>.png` |
| domain 1-D profiles | `$FIG/domains/` | `domain_profiles.png` + `.<model>.txt/.npz` |
| model vs model, sections | `$FIG/compare_<A>_<B>/` | `compare.<section>.png` |
| model vs model, maps | `$FIG/compare_<A>_<B>/` | `compare.<vsv|xi>.<abs|rel>.<fp|wa>.png` |
| regional 1-D profile | `$FIG/<iter>/` | `regional_profile.png` (+ `.txt` table) |
| v3b dispersion maps | `$FIG/v3b/maps/` | `v3b.<Z|T>.<phase|group>.<P>s.dv-wac.fp.png`, `...sheet...`, `v3b.captions.txt` |
| v3b domain dispersion | `$FIG/v3b/` | `domain_dispersion.png/.pdf/.csv/.npz` |
| site dispersion (19 sites) | `$FIG/v3b/sites/` | `site_dispersion.png`, `.overlay.png`, `.csv`, `.<set>.csv` |
| site Vsv profiles | `$FIG/<iter>/sites/` | `site_vsv.<iter>.png`, `.overlay.png`, `.<iter>.csv`, `.err.csv`, `.nodes.txt` |
| site Vsv, two models | `$FIG/sites/` | `site_vsv.iter2_iter10.png`, `.diff.csv`; `site_depth_maps.iter2_iter10.png` |
| domain Vsv (nine regions) | `$FIG/<iter>/domains/` | `domain_vsv.<iter>.png` + `.<iter>.txt/.npz` |

Every PNG has a `.pdf` beside it. `fp` = array-footprint view, `wa` = whole WA.

---------------------------------------------------------------------------------------
## 2. Conventions (defaults)
- Absolute values: continuous **Spectral** (warm = slow). Relative values and xi: **RdBu** with a
  small white centre and gentle transition (`--div-gap 0.07 --div-white 2`). Vp/Vs: **viridis**.
- 16 colour levels; cross-sections VE 3, depth 0–60 km.
- Depth slices at **5 / 15 / 30 / 45 km** (not 40: Moho artefact; now the make_depth_maps default).
- dln reference: whole-model mean at each depth by default; `--ref-domain yilgarn` = Yilgarn mean.
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
All on one page: `--stack-split none`. Points more than 0.5° from any model node are left blank
(as on the maps): `--max-node-dist 0.5`, `0` = off. Model minus reference: `--stack vdiff --ref-fvs $FVS2`.

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
## 7b. Domain 1-D profiles and the Yilgarn reference
Domains (Yilgarn, West Australian Craton, Perth Basin, Albany-Fraser, Capricorn) are sets of
GSWA 10M units in `basemap/domains.csv` (add Canning, Kimberley, Pilbara ... by uncommenting a
line). Nodes: inside the domain and Ndat >= 35 (`--footprint` = inside the array footprint too).
```bash
# representative profiles: Vsv / Viso / xi from iter8, Vsv / Vp/Vs from iter11
python3 plotting/plot_domain_profiles.py --fvs $FVS8:iter8 $FVS11:iter11 --moho $MOHO \
    --d-max 60 --out $FIG/domains/domain_profiles.png
# -> domain_profiles.png/.pdf + domain_profiles.<model>.txt/.npz (mean, std, p16/50/84 per depth)
python3 plotting/plot_domain_profiles.py --fvs $FVS10:iter10 $FVS2:iter2 \
    --domains yilgarn perth albany_fraser --out $FIG/domains/vsv_iter10_iter2.png

# dln relative to the Yilgarn mean at each depth (files tagged ref-yilgarn)
python3 plotting/plot_xsection.py --fvs $FVS11 --load-sections $XS --stack dvsv,dvpvs \
    --ref-domain yilgarn --out-dir $FIG/iter11        # xsection_stack.dvsv_dvpvs.ref-yilgarn.ns.png
python3 plotting/plot_xsection.py --fvs $FVS8 --vel vsv_true --load-sections $XS --stack dvsv,dxi \
    --ref-domain yilgarn --out-dir $FIG/iter8
python3 plotting/make_depth_maps.py --fvs-zt $FVS8 --tag-zt iter8 --modes rel --ref-domain yilgarn \
    --out-dir $FIG/iter8                               # iter8.vsv.rel-yilgarn.fp.png ...
python3 plotting/make_depth_maps.py --fvs-z $FVS11 --tag-z iter11 --modes rel --ref-domain yilgarn \
    --out-dir $FIG/iter11
```
Export the domains for other projects (one GeoJSON, no repo needed there):
`python3 extract/export_domains.py --all --out $FIG/domains/wa_domains.geojson`
(one feature per domain x 10M unit; properties domain/label/colour/unit; lon/lat).

`--ref-domain` works for single sections too (`xsection.<line>.ref-yilgarn.png`) and in
`plot_vsv_tectonic.py`; any domain key from `domains.csv` can be the reference.

---------------------------------------------------------------------------------------
## 7c. v3b surface-wave maps and domain dispersion curves
Input: `$V3B/{Z,T}/{group,phase}_v_tomo/TransD.region.{group,phase}.npz` (v3b band120,
iter 2; keys fk, vall2 [nper,70,60] km/s, std2, maski, lat2d, lon2d). Periods:
Z phase 2.5–60 s (29), Z group 2.5–40 (24), T phase 2.5–48 (26), T group 2.5–32 (22).

**Mask (every WA Array map):** footprint npz key `density` (10 s path samples per cell,
array-to-array paths only) ≥ 10, holes filled, largest region, no dilation — AND posterior
std ≤ 0.80 (finite `maski`) — AND finite velocity. (`mask` in that npz = old v2 mask; not used.)
```bash
# single-period maps, all four sets: dV vs the WAC mean, per-panel limits from inside the WAC
python3 plotting/plot_dispersion_maps.py --root $V3B --periods 3 5 8 12 20 30 40 60 \
    --sheet --pdf --out $FIG/v3b/maps
# one set, chosen periods; fixed +-4 % instead (file tag dv-wac-pm4)
python3 plotting/plot_dispersion_maps.py --root $V3B --sets Z.phase --periods 5 10 20 40 \
    --clim fixed --pct 4 --out $FIG/v3b/maps
# absolute velocity (Spectral) / posterior std / whole WA frame / Yilgarn reference
python3 plotting/plot_dispersion_maps.py --root $V3B --field abs --periods 10 30 --out $FIG/v3b/maps
python3 plotting/plot_dispersion_maps.py --root $V3B --field std --periods 10 30 --out $FIG/v3b/maps
python3 plotting/plot_dispersion_maps.py --root $V3B --extent wa --ref-domain yilgarn --out $FIG/v3b/maps

# per-domain mean dispersion curves (+ node counts), tables beside the figure
python3 plotting/plot_domain_dispersion.py --root $V3B --out $FIG/v3b/domain_dispersion.png
```
- Overlays: domain outlines (WAC thick; Yilgarn, Pilbara, Capricorn, Albany–Fraser, Perth,
  Canning, Officer, Eucla thin, labels at the GSWA 2022 positions; `--domains`, `--labels
  short|full|none`), dashed coverage-mask edge (`--no-cov-line`), array stations W????,
  C[WESC]??, F??? (`--show-stations all` adds the rest hollow).
- Colour scale for dV, one rule for all periods: `--clim ref` (default; ±L per panel, L = 98th
  percentile of |dV| inside the WAC, rounded up to 0.5 %, arrowed colourbar — same rule as
  noise_asdf `plot_masked_periods.py --ref-name "West Australian Craton"`) or `--clim fixed --pct P`.
- `--smooth 4` bilinear upsampling for display (1 = raw cells). `--periods all` for every period.
- Curves: cells in domain ∩ mask at each period; points with < 5 cells dropped (`--min-nodes`);
  rows = mean (±1σ across cells), dV vs WAC, cell count. Domains overlap (WAC ⊃ Yilgarn ...).
- Caption caveats (printed on the maps and in `v3b.captions.txt`): Z group 32–40 s low confidence
  (residual RMS 21–34 s); T group ~30–32 s low confidence; Z phase 52–60 s at the 50-cell floor
  (1.3–2.3k rays), smooth by construction; long-period group dropped (Z > 40 s, T > 32 s).

---------------------------------------------------------------------------------------
## 7d. Study sites (19 Yilgarn points) and domain Vsv profiles
Sites: `examples/yilgarn_sites_19.txt` (no, name, terrane, lat, lon; tab/comma/space; any
other site list in the same form works with `--sites`). Terrane colours follow the site map.
```bash
# 1. dispersion curves at the sites, from the v3b maps (bilinear; filled = inside the map mask)
python3 plotting/plot_site_dispersion.py --root $V3B --sites examples/yilgarn_sites_19.txt \
    --out $FIG/v3b/sites/site_dispersion.png
#    -> site_dispersion.png (one panel per site), .overlay.png (per set, by terrane),
#       .csv (long: site, set, period, v, std2, in_mask), .<Z|T>_<phase|group>.csv (period x site, NaN outside mask)

# 2. iter2 Vsv at the sites (nearest node), values every 0.5 km to 70 km
python3 plotting/plot_site_profiles.py --fvs $FVS2 --sites examples/yilgarn_sites_19.txt \
    --moho $MOHO --z-max 70 --out $FIG/iter2/sites/site_vsv.iter2.png
#    -> .png (one panel per site, +-1 posterior std, AR23 Moho tick), .overlay.png,
#       .csv (depth x site), .err.csv (posterior std), .nodes.txt (node used, distance, Ndat, Moho)
#    --idw 4: inverse-distance mean of the 4 nearest nodes; flags: node > 50 km or Ndat < 35
# 2b. iter2 (Moho) with iter10 (no Moho) on top: black dashed, grey band; + .diff.csv (iter2 - iter10)
python3 plotting/plot_site_profiles.py --fvs $FVS2 --fvs2 $FVS10 --moho $MOHO --z-max 70 \
    --out $FIG/sites/site_vsv.iter2_iter10.png
#    -> .png, .overlay.png (one panel per model), .iter2.csv, .iter10.csv (+ .err.csv, .nodes.txt), .diff.csv
#    + --ref-domain yilgarn: sites as dVsv (%) from each model's Yilgarn mean (its Yilgarn nodes,
#      Ndat >= 35), grey band = +-1 sigma spread of those nodes -> .dv-yilgarn.png, .dv-yilgarn.overlay.png,
#      .<model>.dv-yilgarn.csv, .<model>.yilgarn-mean.csv;  --skip-flagged drops sites > 50 km from a node

# 2c. depth slices with the sites marked (terrane-coloured triangles, numbered): rows = models
python3 plotting/plot_site_depth_maps.py --fvs $FVS2:iter2 $FVS10:iter10 --depths 5 20 35 45 \
    --out $FIG/sites/site_depth_maps.iter2_iter10.png
#    each depth column shares one Spectral scale; --rel = dVsv % vs each model's mean at that depth;
#    outside the v3b coverage mask is faded (white veil, dashed edge; --veil-alpha 0.6,
#    --coverage blank = no colour there, none = off); --extent wa for the whole state

# 3. iter2 Vsv domain means for the nine regions of the domain dispersion figure
python3 plotting/plot_domain_profiles.py --fvs $FVS2:iter2 --domains dispersion --fields vsv \
    --coverage --rel-to wac --sites examples/yilgarn_sites_19.txt --moho $MOHO --d-max 60 \
    --out $FIG/iter2/domains/domain_vsv.iter2.png
#    --domains dispersion = wac yilgarn pilbara capricorn albany_fraser perth canning officer eucla;
#    --coverage = only nodes inside the v3b coverage mask (density >= 10), the cells of the curves;
#    --rel-to wac adds a dVsv (%) panel; tables domain_vsv.iter2.iter2.txt/.npz (0.5 km steps)
#    --sites examples/yilgarn_sites_19.txt marks the 19 sites on the map; with --coverage the land
#    outside the coverage mask is faded (--no-veil to switch off, --veil-alpha, --veil-colour 0.7)

# 4. shallow Vs (top ~0.1-2 km) at the sites, from the station median profiles
#    (<no>_<name>_median_vs_station_<STA>.dat: lon lat depth_m Vs_m/s), same panel layout
python3 plotting/plot_site_shallow.py --dir <folder or station_files.zip> \
    --out $FIG/sites/site_shallow_vs.png
#    -> .png (own depth scale per panel; --zmax 300 for a common one), .overlay.png, .csv (1 m steps),
#       .summary.txt (station, distance, Vs at 0/30/100 m, Vs30, max); flags station > 50 km, Vp/Vs files
#    --slide (also for plot_site_profiles.py): 3 x 7 landscape panels for a 16:9 PowerPoint slide

# 5. lithospheric columns: one bar per site coloured by Vsv in fixed bins (3.0-4.5 every 0.25), to 50 km
python3 plotting/plot_site_columns.py --fvs $FVS2 --moho $MOHO \
    --out $FIG/iter2/sites/site_columns.iter2.png
#    shallow version (m, m/s; 250 m/s bins fitted to the profiles, to 650 m):
python3 plotting/plot_site_columns.py --shallow <folder or station_files.zip> \
    --out $FIG/sites/site_columns.shallow.png
#    colours: vivid turbo, warm = slow (--cmap Spectral for the softer one); --order terrane;
#    --bins LO HI STEP; --z-max; --field xi (0.9-1.1) / vpvs; AR23 Moho = black bar
```

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
## 8b. Plotting on another machine (basin, mantle, laptop)
Code comes from GitHub; everything else comes as one bundle made on d204403.
```bash
# on d204403: models (default iter 2 8 10 11), sections, Moho, stations, footprint, GSWA caches
tools/pack_data.sh                            # -> ~/wa_plot_data.tgz  (-i "8 11" to pick models)
tools/pack_data.sh --cartopy                  # + Natural Earth coastlines, for machines offline
# d204403 cannot reach basin directly: stage through Pawsey, pull on basin/mantle
scp ~/wa_plot_data.tgz data-mover.pawsey.org.au:~/transfer/
#   on basin:  rsync -av <user>@setonix.pawsey.org.au:~/transfer/wa_plot_data.tgz ~/

# on the other machine (once)
git clone git@github.com:huaiyuyuan/wa-geo-plotting.git
tar xzf wa_plot_data.tgz -C ~                 # -> ~/wa_data
conda create -n wgp -c conda-forge python=3.11 numpy scipy matplotlib cartopy   # (+ pdfplumber, tk)
# every session
conda activate wgp
export WA_DATA=~/wa_data; source ~/wa-geo-plotting/env_stage6.sh
```
| In the bundle (`~/wa_data/`) | From d204403 |
|---|---|
| `stage6/Fvs.iter.N.*.npz` | `$ST/` (22-35 MB each) |
| `stage6/figures/xsections/sections.txt` | `$XS` |
| `moho/AR23-moho-hmp.txt` | `/workspace/Others.data/AusMoho2023/` |
| `stations/stations_v2.txt` | `…/ASDF.processing/pathqc.v2/` |
| `footprint/wa_array_footprint.npz`, `wa_array_outline.txt` | `…/ASDF.processing/footprint/` |
| `derived/wa_tectonics.npz`, `wa_lithology.npz`, `wa_crustal_boundaries.npz` | `/workspace/shape.files/derived/` |
| `cartopy_data/` (with `--cartopy`) | `/data/25/yuan/mfiles/cartopy_data` |

With `WA_DATA` set, `env_stage6.sh` points `$ST`, `$XS`, `$MOHO`, `$FVS*` and `config.py`
(via `WA_DERIVED`, `WA_STATIONS`, `WA_FOOTPRINT_DIR`, `WA_TECTONICS_NPZ`) at the bundle, and
`$FIG` at `~/wa_data/stage6/figures` (set `FIG` first to write elsewhere). All commands above then
run unchanged. New model? Re-run `tools/pack_data.sh -i "12"` and unpack over the old bundle.
Windows laptop: use WSL (Ubuntu) and the same steps.

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
| `tar: …/Downloads/wgp_*.tgz: Cannot open` | files from Claude land in `/mnt/c/Users/MIHYUAN/Downloads/` on d204403 (WSL), not `~/Downloads` |
| `gm display: Unable to load font` | `gm display -font fixed file.png` or `eog file.png` |
| section ends saturated at all depths | extrapolation beyond the model nodes: points > 0.5 deg from a node are now blank (`--max-node-dist`, 0 = off); check a line with `python3 tests/check_section_nodes.py $FVS11 $XS EW7 EW8` |

---------------------------------------------------------------------------------------
## 11. Credits for figure captions
- Geological Survey of Western Australia 2022, 1:10 000 000 simplified tectonic map of Western
  Australia, March 2022: Geological Survey of Western Australia, Non-series map.
- Geological Survey of Western Australia 2020, 1:500 000 tectonic units of Western Australia,
  March 2020 update: Geological Survey of Western Australia, digital data layer.
- GSWA Major Crustal Boundaries 2025, digital data layer.
- Caption: *"Domain colours after GSWA (2022); tectonic units and crustal boundaries © State of
  Western Australia (DEMIRS), CC BY 4.0. Moho: AusMoho 2023 (AR23)."*
