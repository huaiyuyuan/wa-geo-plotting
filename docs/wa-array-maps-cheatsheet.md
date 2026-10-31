# WA Array focus-area model maps — cheatsheet

Smooth (contourf) model maps for the **WA Array focus area**, all sharing one
footprint, one cut boundary, and the red Yilgarn craton contour. Built with
`plotting/plot_wa_map.py` on the GSWA 2022 tectonic basemap. Run from the repo root.

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

---------------------------------------------------------------------------------------
## Key flags (this workflow)

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

## Notes / gotchas

- The CCP Moho/Vsv/ξ grid stops at ~**−22.5°N** — the flat northern edge is the data limit,
  not a mask. Only the Moho can fill the north (via `--moho-fill-hk`); Vsv/ξ need a CCP re-grid.
- Deposits live in `plotting/deposits.py` (+ the `.mat` files in `data/deposits/`). `plot_wa_map.py`
  imports it lazily — if it's missing, `--deposits` is silently disabled.
- Deploy a downloaded script with: `mv -f "$(ls -t ~/Downloads/plot_wa_map*.py | head -1)" plotting/plot_wa_map.py`.
