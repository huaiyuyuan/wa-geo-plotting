#!/usr/bin/env bash
# Consolidate WA-geo-plotting data into the repo data/ tree:
#   data/geology/   wa_tectonics.npz  wa_lithology.npz  wa_crustal_boundaries.npz
#   data/stations/  stations_v2.txt   New.WA.Seismic.2026.Phase.4.csv  ...Perm.phase.0.csv
#   data/deposits/  gold/copper/lithium/nickel_deposits.mat  greenstones_sw_yilgarn.mat
#
# Run from anywhere inside the repo on basin, AFTER deploying the updated
# config.py / deposits.py / plot_tectonic_map_phase3.m / draw_deposits.m.
# Idempotent (git mv -k skips what's already moved). Edit the SOURCE paths below
# if yours differ, then review `git status` and commit.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p data/geology data/stations data/deposits

echo "== moving tracked geology npz + deposit mats =="
for f in wa_tectonics.npz wa_lithology.npz wa_crustal_boundaries.npz; do
    [ -e "data/$f" ] && git mv -k "data/$f" "data/geology/$f" && echo "  data/$f -> data/geology/"
done
for f in gold copper lithium nickel; do
    [ -e "matlab/${f}_deposits.mat" ] && git mv -k "matlab/${f}_deposits.mat" "data/deposits/" && echo "  matlab/${f}_deposits.mat -> data/deposits/"
done
[ -e matlab/greenstones_sw_yilgarn.mat ] && git mv -k matlab/greenstones_sw_yilgarn.mat data/deposits/ && echo "  matlab/greenstones_sw_yilgarn.mat -> data/deposits/"

echo "== copying external station files in (edit these if your paths differ) =="
AVAIL='/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/pathqc.v2/stations_v2.txt'
PH4="$HOME/Downloads/New.WA.Seismic.2026- Phase 4.csv"
PERM="$HOME/Downloads/New.WA.Seismic.2026- Perm-phase-0.csv"
[ -e "$AVAIL" ] && cp -v "$AVAIL" data/stations/stations_v2.txt                               || echo "  (available stations not found: $AVAIL)"
[ -e "$PH4"   ] && cp -v "$PH4"   "data/stations/New.WA.Seismic.2026.Phase.4.csv"             || echo "  (phase 4 csv not found: $PH4)"
[ -e "$PERM"  ] && cp -v "$PERM"  "data/stations/New.WA.Seismic.2026.Perm.phase.0.csv"        || echo "  (permanent csv not found: $PERM)"

# Optional: make the two symlinked geology npz self-contained (deref into real files).
# Uncomment to copy the real bytes so a fresh clone works off-basin too:
# for f in wa_tectonics.npz wa_lithology.npz; do
#     if [ -L "data/geology/$f" ]; then cp -L "data/geology/$f" "data/geology/$f.real" && mv "data/geology/$f.real" "data/geology/$f"; fi
# done

git add data/stations data/geology data/deposits
echo
echo "Done. Review with 'git status', then:"
echo "  git add -A"
echo "  git commit -m 'Consolidate data into data/{geology,stations,deposits}'"
echo "  git push origin main"
