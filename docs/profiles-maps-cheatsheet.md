# Profiles & Maps cheat sheet (WA Array)

Companion to `wa-geo-plotting CHEATSHEET.md`. Covers the tools built for the
**picked-transect** workflow: convert the MATLAB `prof` struct → plot Vsv models
along those lines → plot dispersion along those lines → draw the locator map.
Run everything **from the repo root** on d204403. Set up once:

```bash
cd $WGP                       # repo root (/data/25/yuan/github/wa-geo-plotting)
source env_stage6.sh          # sets $ST, $FIG, $XS, etc.
A=/workspace/WA.Array/Year.3/noise.processing/ASDF.processing
PROF=$FIG/profiles/profiles.latlon.phases.1.2.3.polylines.npz
```

The pipeline: `prof.mat → mat_to_profiles.py → {sections.txt, polylines.npz}`,
then feed `polylines.npz` to the profile plotters and the map.

---

## 1. Convert the prof struct — `tools/mat_to_profiles.py`

The MATLAB `prof` (1×N struct: `name, short, latlon[M×2], colour`) → two files.

```bash
python3 tools/mat_to_profiles.py \
    /mnt/data64/workspace/CCP/CCP.WA.Array.2027.phase.3/profiles.latlon.phases.1.2.3.mat \
    --out-dir $FIG/profiles
# writes  <stem>.sections.txt    (endpoints: lat1 lon1 lat2 lon2 label)
#         <stem>.polylines.npz   (full picked paths + short/name/colour/group/order)
```

- lat/lon **column order is auto-detected** from WA value ranges — either way round works.
- Groups assigned from `short`: `WE*`→we, `SENW*`/`WEST`→senw, else→ns. (WEST is the
  coastal SE-NW line, grouped senw, NOT we.)
- Polylines keep every picked vertex, so the curved WEST (Darling-Fault) path survives.
- `--var prof` (struct name), `--stem NAME` (output stem) if needed.
- Check the console: it prints the group split and flags any lat/lon ambiguity.

---

## 2. Vsv along the profiles — `plotting/plot_profiles_models.py`

One landscape page per orientation (WE/NS/SENW); **rows = profiles, columns = models**.

```bash
python3 plotting/plot_profiles_models.py \
    --profiles $PROF \
    --models $ST/Fvs.iter.10.Z.npz:"v2 iter10" \
             $A/v3c/Fvs.v3c.iter.3.Z.npz:"v3c iter3" \
    --field vsv --d-max 60 --out-dir $FIG/profiles
# -> profiles.vsv.v2-iter10_v3c-iter3.{we,ns,senw}.png (+ .pdf)
```

- `--models PATH:"LABEL"[:VEL]` repeatable → one column each. `VEL` = `auto`
  (default; xi/ZT model shown as Vsv is auto-converted to true Vsv), `model`, `vsv_true`.
- `--field vsv | dvsv | viso | xi | vpvs`. `dvsv` = % vs each model's own depth-mean.
- Colour is **shared across the page** (median ±1.5σ for abs) for fair comparison;
  pin it with `--vlim 3.3 4.4`.
- `--max-node-dist 0.5` (deg) blanks points far from a node — SW-core models (v3/v3c)
  go blank on the long WE/SENW ends; that gap *is* the coverage, shown on purpose.
- `--tag NAME` overrides the model-label filename tag. `--straight` = endpoint chords.
  `--no-strips` drops geology strips (strips need `wa_tectonics.npz`). No Moho by default.

---

## 3. Dispersion along the profiles — `plotting/plot_profiles_dispersion.py`

Same profiles, but **y = period (log)**; colour = surface-wave velocity. Reads the node
dispersion files (observed `Data_SWD.in.*`, synthetic `data_bestd.out*`).

```bash
python3 plotting/plot_profiles_dispersion.py \
    --profiles $PROF \
    --data "v2 obs"    "$ST/IN/Data_SWD.in.*" \
    --syn  "v2 iter10" "$ST/OUT.iter.10.Z/data_bestd.out*" \
    --data "v3 obs"    "$A/v3c/IN_v3nodes/Data_SWD.in.*" \
    --syn  "v3c iter3" "$A/v3c/OUT.iter.3.Z/data_bestd.out*" \
    --rayleigh --colour both --resid --pmax 44 --out-dir $FIG/profiles/dispersion
# -> profiles.disp.{cR,uR}.{abs,dv,resid}.<tag>.{we,ns,senw}.png (+ .pdf)
```

- `--data LABEL GLOB` (observed, cols `p cR uR cL uL`), `--syn LABEL GLOB` (synthetic,
  cols `cR uR cL uL p`) — both repeatable, each a column; quote the globs. Node position
  read from `x.<lon>.y.<lat>` in each filename.
- `--comp cR uR cL uL` (one page-set each) or `--rayleigh` = `cR uR`.
- `--colour abs | dv | both` — abs (Spectral), dv (% vs first observed mean, RdBu).
- `--resid` adds syn−obs columns, **paired per version** (1st syn↔1st obs, 2nd↔2nd).
- `--diff "LABEL_A" "LABEL_B"` → a page of A−B along the profiles (e.g. `"v2 obs" "v3 obs"`
  to see where the input data differ). Repeatable.
- `--pmax 44` keeps the period axis tight (group uR data stops ~40 s; no extrapolation).
- **Just side-by-side** (no differencing): drop `--diff`/`--resid`, use `--colour abs`.

---

## 4. Locator / station map — `plotting/plot_wa_map.py`

Standalone WA map (also what `plot_xsection.py --index-map` calls). **No Vs model needed.**

```bash
P4="$HOME/Downloads/New.WA.Seismic.2026- Phase 4.csv"       # names have spaces — quote
PERM="$HOME/Downloads/New.WA.Seismic.2026- Perm-phase-0.csv"

python3 plotting/plot_wa_map.py \
    --sections $FIG/profiles/profiles.latlon.phases.1.2.3.sections.txt \
    --load-polylines $PROF \
    --stations $ST/stations.phases.1.2.3.txt \
    --map-stations "$P4"   lime s "Phase 4" \
    --map-stations "$PERM" red  s Permanent black \
    --label-stations Permanent \
    --extent 112 129 -36 -13 --out-dir $FIG/profiles
# -> xsection_index_map.png AND .pdf (editable Type-42 fonts)
```

- Basemap: GSWA 2022 tectonic units, crustal boundaries, unit labels/legend.
- Profile lines: `--sections` (straight) + `--load-polylines` (true picked paths, distance
  ticks, **open-circle ends**); red line labels with white halo.
- Stations (all squares, differ by colour/edge):
  - `--stations FILE` = **Available**, red square white edge (defaults to `config.STATIONS`;
    `--stations-label` renames; `--no-stations` omits). `--stations $ST/stations.phases.1.2.3.txt`
    for phases 1–3.
  - `--map-stations CSV COLOUR MARKER LABEL [EDGE]` repeatable; CSV needs Latitude/Longitude
    columns; optional 5th token = marker **edge** colour. Convention: Phase 4 = `lime s`,
    Permanent = `red s … black` (red square, black edge).
  - `--label-stations LABEL …` annotates named layer(s) with station codes (from a
    Code/Station/Sta/Site_ID/Site/Name column), e.g. `--label-stations Permanent`.
- `--extent lon0 lon1 lat0 lat1` sets the frame (no model needed); else from sections/stations.
- `--footprint`, `--style map2022|gswa|craton|plain`, `--proj albers|plate`, `--tick 100`.
- Station legend is **lower-right**. The **.pdf** is vector with editable text (Illustrator/Inkscape).

---

## 5. `plot_xsection.py` in one view

Same script, three modes by which flags you pass. Lines come from `--load-sections` /
`--sections` / `--start`+`--end` / `--ginput` (+`--save-sections`). Model: `--fvs model.npz`
(required; `--vel vsv_true` for xi models). Common: `--d-max`, `--moho`, `--strips`,
`--ref-domain`, `--ref-fvs`.

| mode | flag | output |
|---|---|---|
| **Index map** | `--index-map` (`--index-only` to stop there) | locator map — delegates to `plot_wa_map.plot_map`; all §4 flags apply |
| **Stacked** | `--stack vsv,dvsv,xi` | one page per orientation (`.ns`/`.ew`), rows=sections, cols=fields; `--stack-split none` = one page |
| **Per-section** | *(neither)* | one figure per line; `--nproc N` to parallelize |

Stack/section fields: `vsv dvsv xi dxi vpvs dvpvs vdiff err`.

---

## Gotchas (learned the hard way)

- **Push the coupled pair together.** `plot_wa_map.py` and `plot_xsection.py` share
  `_load_station_csv` and the `--index-map` call — a change to one usually needs the other.
- **CSV filenames have spaces** → always quote `"$P4"`, `"$PERM"`.
- **Overwrite downloads, don't pile up:** `mv -f "$(ls -t ~/Downloads/plot_wa_map*.py | head -1)" plotting/plot_wa_map.py`.
- **Coverage blanking** (`--max-node-dist`) is expected on SW-core models at the far ends —
  not a bug.
- **Geology strips / tectonic basemap** need the `wa_*.npz` caches (present on d204403, not
  on every machine); the code falls back without them.
- **Run from the repo root** — `plotting/…` paths resolve relative to cwd.
