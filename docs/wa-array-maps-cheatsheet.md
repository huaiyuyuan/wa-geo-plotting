# WA Array focus-area maps & cross-sections — cheatsheet

Model **maps** (`plotting/plot_wa_map.py`) and **cross-sections**
(`plotting/plot_xsection.py`) for the **WA Array focus area**. Maps share one
footprint, one cut boundary, and the red Yilgarn craton contour, on the GSWA 2022
tectonic basemap. Cross-sections carry the AusMoho + CCP Moho overlays and an optional
CCP RF panel. Run from the repo root. (Cross-sections: see "Cross-sections" below.)

```bash
source env_stage6.sh          # sets $WGP, $ST, $FIG, ...
CCP=/workspace/CCP/CCP.WA.Array.2027.phase.3
FINAL=$CCP/Moho.picked/moho_ccp_refined.final.mat          # CCP Moho pick (grid + stn_mask)
A=/workspace/WA.Array/Year.3/noise.processing/ASDF.processing
VS=$A/v3c/Fvs.v3c.iter.3.Z.npz                             # Vsv model
XI4=$A/v3c/Fvs.v3c.ccpmoho.iter.4.ZT.npz                   # final xi model (ZT)
HK=data/stations/HK.final.2026.merged.txt                 # per-station H-K (Vp/Vs + Moho depth)
PVH=$CCP/mat/WA.Array.phase.4.*.pvh.6.P.mat               # pvh.6 CCP cube (for Moho amplitude)
CMAPX=/data/25/yuan/mfiles/mat/cmapx.mat                   # MATLAB Vsv colormap

EXT="--extent 112 127 -36 -20 --proj albers"
LBL="--labels major --legend none"
CUT="--cut-poly data/masks/wa_array_cut.txt"               # the agreed WA Array boundary
YC="--yilgarn --yilgarn-fill none --yilgarn-bounds craton" # red craton contour, no fill
FM="--field-max-dist 0.8 --field-dgrid 0.08"               # Vp/Vs & H-K-fill footprint
CLIP="--clip-stations $HK --clip-dist 0.8"                 # CCP maps -> Vp/Vs footprint
```

---------------------------------------------------------------------------------------
## The model maps (all contourf, 16 colours, snapped ticks, red craton contour)

```bash
# Moho depth  (CCP where it exists + H-K station fill for the north; parula_r 30-55 km)
python3 plotting/plot_wa_map.py $EXT $LBL --moho-grid $FINAL --moho-field z_moho \
    --moho-fill-hk $HK $FM $CUT $YC --name moho_depth --out-dir figures

# Vp/Vs  (station-kriged; rgb red->green->blue, 1.70-1.86)
python3 plotting/plot_wa_map.py $EXT $LBL --hk $HK $FM $CUT $YC --name vpvs --out-dir figures

# Moho amplitude  (peak of the pvh.6 cube near the Moho; turbo_r 0-0.1)
python3 plotting/plot_wa_map.py $EXT $LBL --moho-grid $FINAL --moho-amp $PVH \
    $CLIP $CUT $YC --name moho_amp --out-dir figures

# Crustal-average Vsv  (0 -> Moho; cmapx)
python3 plotting/plot_wa_map.py $EXT $LBL --vsavg $VS --moho-grid $FINAL \
    --avg-window 0 moho --field-cmap $CMAPX $CLIP $CUT $YC --name vsv_crust --out-dir figures

# xi: whole crust + depth windows  (rwb, 0.96-1.04 crustal / symmetric sub-Moho)
for W in "0 moho" "0 5" "5 20" "20 moho" "moho 60"; do
  python3 plotting/plot_wa_map.py $EXT $LBL --xiavg $XI4 --moho-grid $FINAL \
      --avg-window $W $CLIP $CUT $YC --name xi --out-dir figures
done
```

`--name xi` auto-tags each window: `.0-moho` (whole crust), `.0-5`, `.5-20`, `.20-moho`, `.moho-60`.

## Tectonic maps (small WA Array area)

```bash
# Greenstone fill + red craton contour
python3 plotting/plot_wa_map.py --stations $ST/stations.phases.1.2.3.txt $EXT $LBL \
    --yilgarn --yilgarn-fill greenstone --yilgarn-bounds craton \
    --name wa_array_tectonic_greenstone --out-dir figures

# Gold deposits + red craton contour (no greenstone)
python3 plotting/plot_wa_map.py --stations $ST/stations.phases.1.2.3.txt $EXT $LBL \
    --yilgarn --yilgarn-fill none --yilgarn-bounds craton \
    --deposits gold --name wa_array_tectonic_gold --out-dir figures
```

Deposits (`--deposits gold,copper,lithium,nickel`) are drawn by `plotting/deposits.py`
(log-area sizing) and listed in the lower-right "Stations & deposits" legend.

### Line-group location map (one set of lines + greenstone + gold)

A map showing just one group of cross-section lines (with 100 km distance marks) over
greenstone + gold + the red craton — e.g. one per orientation (NS / WE / basin+AFO+L):

```bash
# build a one-group section list, then the map
gsec=data/sections/NS.txt; : > "$gsec"
for L in NS1 NS2 NS3 NS4 NS5 NS6 NS120; do awk -v l="$L" '$NF==l' "$ALLSEC" >> "$gsec"; done
python3 plotting/plot_wa_map.py $EXT --labels major --legend none \
    --sections "$gsec" --tick 100 --no-stations \
    --yilgarn --yilgarn-fill greenstone --yilgarn-bounds craton \
    --deposits gold --name NS_map --out-dir figures
```

`$ALLSEC` = all section lists concatenated (`cat "$XSW" "$CCPSEC" data/sections/new_xsections.txt
| grep -v '^#' | awk 'NF>=5' > $ALLSEC`). To bundle a group's profiles + its map into one tar
per group, loop this with a `make_bundle` helper (grab each `xsection.<label>[.gold-gs].png`,
whichever tag exists). Line groups on d204403: NS = `NS1..NS6 NS120` (+ `116.5E 117.5E`);
WE = `WE1..WE5` (EW = `EW1..EW8`); SENW = `SENW1..SENW3 L2_SENW`; basin = `WEST`; AFO = `AF`;
L-lines = `L1_SN_117E L2_SENW`.

### Sm-Nd model-age map (TDM2)

GSWA two-stage depleted-mantle model age **TDM2 (Ma)** kriged into a map field; the
tectonic units go **transparent with thick black outlines**, the busy crustal-boundary
lines (and their legend) are dropped, and the colorbar replaces the rock/age legend box
(so the colourbar *is* the age key).

```bash
# one-off: classified shapefile -> npz (pure-Python reader, numpy only). Per data drop.
python3 extract/make_smnd.py \
    --shp <...>/GEOCHRON_Samarium_Neodymium_GDA2020_SHP/.../WA_isotope_neodymium \
    --out data/geology/wa_smnd_tdm2.npz   # 1506 felsic samples, TDM2 1250-3973 Ma

# the map + stations + red craton contour. Default colours = Smithies bins (see below).
python3 plotting/plot_wa_map.py $EXT $LBL $YC \
    --smnd-tdm2 data/geology/wa_smnd_tdm2.npz --smnd-max-km 60 \
    --stations $ST/stations.phases.1.2.3.txt \
    --station-cut-poly data/masks/wa_array_cut.txt \
    --scale-bar 200 \
    --out-dir figures                            # -> tdm2_age_map_w_station.png (+ .pdf)
```

- **Scale bar:** `--scale-bar 200` draws a 200 km horizontal bar just above the lower-right
  Stations legend, like the GSWA/Smithies map. Length is true distance from the Albers
  projected width, but the bar is drawn in axes coords so it never shifts the map frame/coastline.

- **Drop off-footprint stations:** `--station-cut-poly data/masks/wa_array_cut.txt` keeps only
  stations SW of the WA Array cut line (drops the Pilbara/Capricorn/Paterson cluster north &
  east of ~Hamersley). Unlike `--cut-poly`, it clips **stations only** — the field stays full-state.

- **Red craton contour:** add `$YC` (`--yilgarn --yilgarn-fill none --yilgarn-bounds craton`),
  same as the model maps — draws the red Yilgarn/Pilbara outline over the field (zorder 4.6).
- **No-data land** (WA land with no Sm-Nd sample within `--smnd-max-km`) is filled light grey
  (#e6e6e6) so it reads as "no data" rather than blank page; the sea stays white.

- **Felsic only:** `make_smnd.py` keeps only felsic igneous rocks by default (SYM_TDM2 is a
  numeric bin, not "Non-felsic Igneous Rocks") — 1506 of 2390 — as in the GSWA/Smithies
  model-age map; `--all-rocks` keeps everything. The official bin edges are embedded in the npz.
- Interp: kriging by default (needs `pykrige`; falls back to bounded-linear if absent),
  `--field-interp rbf|cubic|linear` to switch. `--field-dgrid 0.1` grid step.
- **Sparse coverage:** `--smnd-max-km 60` blanks cells >60 km (km-accurate) from any
  Sm-Nd sample, so the field only shows where there's real data — raise/lower to taste.
- **Colours (`--tdm2-bins`):** default `smithies` = the Smithies (2022) paper bins & colours
  (2500/2800/2900/2980/3050/3120/3190/3260/3350/3505/4000 Ma: green → gold → tan → rose →
  near-white oldest) **extended to younger ages** with four extra bins (1250/1600/1900/2200/2500)
  in deep green → teal → cyan. The fine old-end steps give strong Yilgarn contrast. `official` =
  the GSWA classified SYM_TDM2 bins carried in the npz, coloured from the `tdm2age` ramp. Colorbar
  ticks sit on the bin edges. `--field-cmap NAME` switches to a continuous ramp (`tdm2`, `tdm2age`,
  any matplotlib name) instead of discrete bins.
- `--field-clim LO HI` (continuous mode only) pins the range; `--field-style smooth` = no bands.
- `--smnd-tdm2` auto-names the file `tdm2_age_map`; stations add the `_w_station` suffix.
- Add `--extra-sections …` to overlay L1/L2, or drop `--stations` for a clean field map.

=======================================================================================
## Cross-sections (`plotting/plot_xsection.py`)

Per-section panels (Viso/Vsv, dViso, ξ, dlnξ, Vp/Vs, CCP RF, uncertainty) along the
picked transects, with the AusMoho (dashed black + grey veil) and the CCP refined Moho
(solid red). Saves `xsection.<label>[.ref-*].png` (+ `.pdf`) to `--out-dir`.

```bash
XI4=$A/v3c/Fvs.v3c.ccpmoho.iter.4.ZT.npz                  # xi (ZT) model: Viso + xi
MOHO=/workspace/Others.data/AusMoho2023/AR23-moho-hmp.txt # regional AusMoho (dashed + veil)
FINAL=$CCP/Moho.picked/moho_ccp_refined.final.mat         # CCP refined Moho pick (red line)
PVH4=$(ls -1 $CCP/mat/*pvh.4*.Z.P.mat | head -1)          # CCP RF stack (NB .Z.P.mat, not *P.mat!)
SEC=data/sections/new_xsections.txt                       # L1 (S->N, 117E), L2 (SE->NW)
XOUT=$WGP/figures/xsections
```

```bash
# (1) FULL landscape: Viso + xi + CCP RF (pvh.4), contourf 15 colours, AusMoho dashed+veil,
#     CCP Moho red line, greenstone+gold track, and the grey craton bar above the domain bar.
python3 plotting/plot_xsection.py --fvs $XI4 --load-sections $SEC \
    --panels vsv,xi,ccp --ccp-pmat "$PVH4" --moho $MOHO --moho-ccp $FINAL \
    --field-style contourf --ncolors 15 --vlim 3.0 4.6 --xi-lim 0.95 1.05 \
    --geol-track --greenstone-npz data/geology/yilgarn_units.npz \
    --deposit-dir data/deposits --gold-swath 50 --craton-line \
    --ccp-neutral 0.10 --ccp-clim 0.08 --out-dir $XOUT
#   --field-style smooth  -> continuous MATLAB-style shading (no bands)
#   -> xsection.<label>.gold-gs.png (+ .pdf)

# (2) VARIATIONS wrt the mean of all loaded sections: dlnVs + dlnXi + CCP RF (tagged .ref-allprof)
python3 plotting/plot_xsection.py --fvs $XI4 --load-sections $SEC \
    --panels dvsv,dxi,ccp --xi-panel dxi --var-ref allprofiles --ccp-pmat "$PVH4" \
    --moho $MOHO --moho-ccp $FINAL --field-style contourf --ncolors 15 \
    --geol-track --greenstone-npz data/geology/yilgarn_units.npz \
    --deposit-dir data/deposits --gold-swath 50 --craton-line \
    --ccp-neutral 0.10 --ccp-clim 0.08 --out-dir $XOUT
#   -> xsection.<label>.ref-allprof.gold-gs.png (+ .pdf)
```

### Tectonic-bar overlays (greenstone + gold + craton)

- **`--geol-track`** adds a strip above the domain bar: green **greenstone fill** (where the
  profile crosses greenstone, from `--greenstone-npz`) with **gold deposits** projected onto the
  line within `--gold-swath` km (50) as log-area markers (`--gold-style markers`|`lines`).
  Gold is read from `--deposit-dir`/gold_deposits.mat (struct `gold`: lon lat size) — the same
  file as the maps — or from a `--gold-file` (`lon lat [endowment] [kind]`). Tags outputs `.gold-gs`.
- **`--craton-line`** adds a **grey bar in the gap above the domain bar** spanning where the
  profile is inside the Yilgarn/Pilbara craton (from `data/geology/yilgarn_craton_outline.txt`,
  `--craton-outline` to override). No extra row — the Au/greenstone strip stays close.

### Plot ALL the lines at once

`plot_xsection` draws exactly the lines in `--load-sections`. The section lists live in
three separate files, so to plot everything concatenate them into one (all straight
great-circle lines — curved basin/albany-fraser come out as straight endpoints):

```bash
XSW=$FIG/xsections/sections.txt                                 # NS/EW surface-wave grid
CCPSEC=$FIG/profiles/profiles.latlon.phases.1.2.3.sections.txt  # SENW, basin (WEST), albany-fraser (AF), CCP NS/WE
ALL=data/sections/all_sections.txt
cat "$XSW" "$CCPSEC" "$SEC" | grep -v '^#' | awk 'NF>=5' > "$ALL"   # drop comments/blank rows
wc -l "$ALL"                                                        # how many lines

python3 plotting/plot_xsection.py --fvs $XI4 --load-sections $ALL \
    --panels vsv,xi,ccp --ccp-pmat "$PVH4" --moho $MOHO --moho-ccp $FINAL \
    --field-style contourf --ncolors 15 --vlim 3.0 4.6 --xi-lim 0.95 1.05 \
    --ccp-neutral 0.10 --ccp-clim 0.08 --nproc 8 --out-dir $XOUT
#   variations across the whole set: --panels dvsv,dxi,ccp --xi-panel dxi --var-ref allprofiles
```

NB: duplicate labels across files overwrite each other's PNG; `plot_xsection` does
straight lines only, so basin/AF lose their picked curve (use `plot_ccp_profiles.py` for the
true curved path).

### Cross-section flags

| Flag | What it does |
|---|---|
| `--load-sections FILE` | section list, one per line: `lat1 lon1 lat2 lon2 label`. |
| `--panels vsv,dvsv,xi,dxi,vpvs,ccp,err` | which panels to draw: abs Viso/Vsv, dViso, abs ξ, dlnξ, Vp/Vs, CCP RF, uncertainty. |
| `--xi-panel xi\|dxi\|both` | absolute ξ, dlnξ, or both — **needed for the `dxi` panel**. |
| `--moho FILE` | regional Moho overlay — AusMoho grid text **or** a CCP `.mat`; dashed black + grey veil below. `--no-moho-mask` = line only. |
| `--moho-ccp FINAL.mat` | CCP refined Moho pick as a solid **red** line (NaN outside the CCP footprint). |
| `--ccp-pmat P.mat` | add the CCP RF amplitude panel (pvh.4); reads the CMP cube from `P.ofile1`. **Glob `*.Z.P.mat`, not `*P.mat` (that also matches the CMP file).** `--ccp-cmp`, `--ccp-amp-var` to override. |
| `--ccp-clim C` / `--ccp-neutral F` / `--ccp-method` | CCP symmetric limit ±C (0.08); soft-white centre fraction F (0.10 ≈ one white band at odd `--ncolors`); interp linear/nearest/cubic. |
| `--var-ref model\|yilgarn\|profile\|allprofiles` | baseline for dViso/dlnξ (per depth): whole-model mean / Yilgarn mean / each profile's own mean / **mean of all sections**. Tags outputs `.ref-profile` / `.ref-allprof`. |
| `--field-style contourf\|smooth` | discrete `--ncolors` bands (default) or continuous MATLAB-style shading. |
| `--ncolors N` | colour bands (15 ≈ 1.5× of 10; **odd** → a single white centre on diverging/CCP maps). |
| `--vlim LO HI` / `--xi-lim LO HI` | absolute Viso/Vsv range (e.g. 3.0 4.6) / absolute ξ range (e.g. 0.95 1.05; default 0.92 1.08). |
| `--clim-rel P` / `--clim-xi P` | ± percent range for dViso / dlnξ (default 6 / 5). |
| `--vel auto\|model\|vsv_true` | panel-1 velocity: **Viso** (auto/model, default) or **true Vsv** (`vsv_true` converts the ξ/ZT model). |
| `--ve N` | vertical exaggeration = **data aspect ratio** (default 3 → 3:1; `--ve 2` → 2:1; `--ve 1` = true scale). Also `--long-ratio N` to fix panel width:height instead. |
| `--vs-smooth SIGMA` | gentle NaN-aware Gaussian smooth of the **Vs panel only** (SIGMA in grid cells; try 1–2; 0 off). ξ/CCP untouched. |
| `--xi-cmap NAME` | ξ colormap: a general blue-red (`RdBu` **default**, or `coolwarm`/`bwr`/`seismic`), or `rwb` for the custom red-white-blue. |
| `--xi-gap GAP` | ξ white-centre width for `rwb` only (default 0.12; lower = more gradual; 0 = smooth). Ignored for RdBu/others. |
| `--geol-track` | greenstone-fill + gold-deposit track above the domain bar (tags `.gold-gs`). |
| `--greenstone-npz F` | greenstone polygons (default `data/geology/yilgarn_units.npz`). |
| `--deposit-dir D` / `--gold-file F` | gold from `D/gold_deposits.mat` (default repo `matlab/`; yours is `data/deposits`) or a `lon lat [endowment] [kind]` list. |
| `--gold-swath KM` / `--gold-style markers\|lines` | projection halfwidth (50) / log-area circles or vertical ticks. |
| `--craton-line` / `--craton-outline F` | grey craton bar above the domain bar / its outline (default `data/geology/yilgarn_craton_outline.txt`). |
| `--fault 'LABEL:LON,LAT'` | mark a fault (e.g. Ida) with a **red down-arrow on the top strip** (same style as the boundary crossings, red + thick) at the profile point nearest LON,LAT, red label above. Repeatable; one lon/lat projects onto every profile that crosses it. |
| `--stack FIELD` | stack one field across all sections, one page per orientation. |
| `--nproc N` | plot sections in parallel. |

Colormaps (match the maps): Viso/Vsv = `cmapx` rainbow (slow red → fast blue); ξ = **`RdBu`**
general blue-red by default (red < 1 < blue, natural white centre; `--xi-cmap rwb` for the old
tight-white custom map); Vp/Vs = `rgb`; CCP RF = soft red-white-blue (blue +, red −). Colorbar
ticks snap to round values. CCP RF has no "variation" (it's an amplitude image), so dViso/dlnξ
apply to the model fields only.

---------------------------------------------------------------------------------------
## Key flags (maps)

| Flag | What it does |
|---|---|
| `--cut-poly FILE` | keep only the SW side of a lon/lat polyline (`data/masks/wa_array_cut.txt`); clips **field AND stations**. Errors loudly if the file isn't found. |
| `--draw-cut` | draw the cut line (magenta dashed) to see which stations are in/out. |
| `--moho-grid FINAL.mat` | CCP Moho pick (`moho_refined`: lat2/lon2/z_moho/amp_moho/stn_mask). |
| `--moho-field z_moho\|amp_moho` | Moho depth, or the picked (flat) amplitude. |
| `--moho-amp PMAT` | physical Moho amplitude = peak of the pvh.6 cube within `--amp-win` km of the Moho. |
| `--moho-fill-hk HK` | fill the Moho north/gaps (beyond the CCP grid, which stops ~−22.5°N) with the H-K station Moho (`h_new`), keeping continuous CCP where it exists. **Moho only** (stations have `h_new`; Vsv/ξ don't). |
| `--no-stn-mask` | ignore the CCP `stn_mask`; show data wherever it has a value (holes stay blank). |
| `--vsavg NPZ --avg-window A B` | crustal-average Vsv over a depth window (`moho` = that node's Moho; `0 moho` = whole crust). |
| `--xiavg NPZ --avg-window A B` | same for ξ. Crustal → clim 0.96-1.04; sub-Moho → symmetric about 1.0. |
| `--clip-stations HK --clip-dist 0.8` | mask any field to within 0.8° of the stations, so CCP maps match the Vp/Vs footprint. |
| `--hk HK [--hk-field k_new\|h_new\|vp\|vs]` | station-kriged map (Vp/Vs, H-K Moho, Vp, Vs). k_new → rgb; h_new → parula_r. |
| `--yilgarn` | overlay the parsed GSWA Yilgarn geology (`data/geology/`). |
| `--yilgarn-fill greenstone\|all\|none` | fill rock types (greenstone default) or none. |
| `--yilgarn-bounds craton\|all\|none` | red craton outline only / + grey terrane-domain lines / none. |
| `--deposits gold,copper,lithium,nickel` | mineral-deposit markers via `plotting/deposits.py` + legend. |
| `--field-style contourf\|pcolor\|smooth` | default **contourf** for all maps now. |
| `--field-ncolors N` | colour bands (default 16). `--field-clim LO HI`, `--field-cmap NAME` to override. |

Colormaps: `parula`/`parula_r`, `rwb` (red-white-blue, ξ), `rgb` (red→green→blue, Vp/Vs),
`turbo`/`turbo_r`, any matplotlib name, or a `.mat` file (e.g. `cmapx.mat`). Colorbar ticks
snap to round values automatically.

---------------------------------------------------------------------------------------
## Data added this round

- `data/geology/yilgarn_units.npz` — 867 GSWA Yilgarn rock-type polygons (greenstone/
  granite/gneiss/sedimentary), parsed from the GSWA shapefiles (Web-Mercator → lon/lat).
- `data/geology/yilgarn_boundaries.npz` — craton (red) + terrane/domain (grey) boundary lines.
- `data/geology/yilgarn_contours.mat` / `yilgarn_craton_outline.txt` — the craton outline for
  MATLAB (`[cx,cy]=wa_albers(Y.craton_lon,Y.craton_lat); plot(ax,cx,cy,'r-')`).
- `data/masks/wa_array_cut.txt` — the agreed WA Array cut boundary (edit the lon/lat rows to nudge).
- `data/geology/wa_smnd_tdm2.npz` — 1506 **felsic** GSWA Sm-Nd samples (lon/lat + TDM2 Ma,
  classified SYM_TDM2, TECT_UNIT, official `bins`), from `extract/make_smnd.py` on the
  classified GEOCHRON Sm-Nd shapefile. Feeds `--smnd-tdm2`. (`--all-rocks` → all 2390.)

## Notes / gotchas

- The CCP Moho/Vsv/ξ grid stops at ~**−22.5°N** — the flat northern edge is the data limit,
  not a mask. Only the Moho can fill the north (via `--moho-fill-hk`); Vsv/ξ need a CCP re-grid.
- Deposits live in `plotting/deposits.py` (+ the `.mat` files in `data/deposits/`). `plot_wa_map.py`
  imports it lazily — if it's missing, `--deposits` is silently disabled.
- Deploy a downloaded script with: `mv -f "$(ls -t ~/Downloads/plot_wa_map*.py | head -1)" plotting/plot_wa_map.py`.
