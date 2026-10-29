# data/stations/

WA Array station coordinate files, consolidated here so the plotters don't depend
on `/workspace` or `~/Downloads`. `config.py` points at these by default (each with
a `$WA_*` env override).

| file | what | config constant |
|------|------|-----------------|
| `stations_v2.txt`                        | current deployment ("Available", red squares) — `code lat lon`, cols (lon=2, lat=1) | `STATIONS` / `$WA_STATIONS` |
| `New.WA.Seismic.2026.Phase.4.csv`        | Phase 4 grid (green squares) — GSWA export, Latitude/Longitude columns auto-detected | `STATIONS_PHASE4` / `$WA_STATIONS_PHASE4` |
| `New.WA.Seismic.2026.Perm.phase.0.csv`   | permanent stations — GSWA export                                                     | `STATIONS_PERM` / `$WA_STATIONS_PERM` |

These are deployment snapshots — refresh them when the GSWA exports change. The
loaders accept either the whitespace `code lat lon` format or a GSWA CSV with
named Latitude/Longitude columns (any order).

Add a Phase 4 layer to any map tool with, e.g.:
```
--map-stations data/stations/New.WA.Seismic.2026.Phase.4.csv green s "Phase 4"
```
