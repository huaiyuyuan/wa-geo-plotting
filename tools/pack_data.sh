#!/bin/bash
# pack_data.sh - bundle everything wa-geo-plotting needs besides the code (models, sections,
# Moho, stations, footprint, GSWA basemap caches, optionally Natural Earth data) into ONE
# tarball for another machine (basin, mantle, laptop). Run on d204403:
#
#   tools/pack_data.sh                         # iter 2 8 10 11 -> ~/wa_plot_data.tgz
#   tools/pack_data.sh -i "8 11" -o /tmp/x.tgz # only these models
#   tools/pack_data.sh --cartopy               # + Natural Earth (for machines offline)
#
# On the other machine:  tar xzf wa_plot_data.tgz -C ~      (-> ~/wa_data)
#   export WA_DATA=~/wa_data; source <repo>/env_stage6.sh
set -e
ST=${WA_STAGE6:-/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/stage6}
ASDF=$(dirname "$ST")
ITERS="2 8 10 11"; OUT=~/wa_plot_data.tgz; CARTO=0
while [ $# -gt 0 ]; do
  case $1 in
    -i) ITERS=$2; shift 2;;
    -o) OUT=$2; shift 2;;
    --cartopy) CARTO=1; shift;;
    *) echo "usage: $0 [-i \"2 8 10 11\"] [-o out.tgz] [--cartopy]"; exit 1;;
  esac
done
TMP=$(mktemp -d); W=$TMP/wa_data
mkdir -p $W/stage6/figures/xsections $W/moho $W/stations $W/footprint $W/derived
add() { if [ -e "$1" ]; then ln -s "$1" "$2"; echo "  + $1"; else echo "  - missing $1"; fi; }
for i in $ITERS; do for f in $ST/Fvs.iter.$i.*.npz; do add "$f" $W/stage6/; done; done
add $ST/figures/xsections/sections.txt              $W/stage6/figures/xsections/
add /workspace/Others.data/AusMoho2023/AR23-moho-hmp.txt  $W/moho/
add $ASDF/pathqc.v2/stations_v2.txt                 $W/stations/
add $ASDF/footprint/wa_array_footprint.npz          $W/footprint/
add $ASDF/footprint/wa_array_outline.txt            $W/footprint/
for f in wa_tectonics wa_lithology wa_crustal_boundaries; do
  add /workspace/shape.files/derived/$f.npz $W/derived/
done
if [ $CARTO = 1 ]; then
  for c in /data/25/yuan/mfiles/cartopy_data ~/.local/share/cartopy; do
    [ -d $c ] && { add $c $W/cartopy_data; break; }
  done
fi
tar -C $TMP -chzf "$OUT" wa_data            # -h: store the files, not the links
rm -rf $TMP
echo "Packed: $OUT ($(du -h "$OUT" | cut -f1))"
echo "On the other machine:  tar xzf $(basename $OUT) -C ~ ; export WA_DATA=~/wa_data ; source <repo>/env_stage6.sh"
