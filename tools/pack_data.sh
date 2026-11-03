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
CCPDIR=${WA_CCP:-/workspace/CCP/CCP.WA.Array.2027.phase.3}
CMAPXSRC=${WA_CMAPX:-/data/25/yuan/mfiles/mat/cmapx.mat}
ITERS="2 8 10 11"; OUT=~/wa_plot_data.tgz; CARTO=0; CCPCUBE=0
while [ $# -gt 0 ]; do
  case $1 in
    -i) ITERS=$2; shift 2;;
    -o) OUT=$2; shift 2;;
    --cartopy) CARTO=1; shift;;
    --ccp) CCPCUBE=1; shift;;          # also bundle the big CCP RF cube (pvh.4 P+CMP .mat)
    *) echo "usage: $0 [-i \"2 8 10 11\"] [-o out.tgz] [--cartopy] [--ccp]"; exit 1;;
  esac
done
TMP=$(mktemp -d); W=$TMP/wa_data
mkdir -p $W/stage6/figures/xsections $W/moho $W/stations $W/footprint $W/derived
add() { if [ -e "$1" ]; then ln -s "$1" "$2"; echo "  + $1"; else echo "  - missing $1"; fi; }
for i in $ITERS; do for f in $ST/Fvs.iter.$i.*.npz; do add "$f" $W/stage6/; done; done

# ── recent v3c / CCP-Moho models (the current Vsv / xi / v2v3comb deliverables) ──
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p $W/v3c/v2v3comb $W/v3c/smoomoho $W/CCP/Moho.picked $W/deposits
add $ASDF/v3c/Fvs.v3c.ccpmoho.iter.4.ZT.npz                        $W/v3c/              # final xi (ZT)
add $ASDF/v3c/Fvs.v3c.ccpmoho.iter.2.Z.npz                         $W/v3c/              # true Vsv
add $ASDF/v3c/Fvs.v3c.ccpmoho.iter.1.Z.npz                         $W/v3c/              # Vs overlay model
add $ASDF/v3c/v2v3comb/Fvs.v3c.v2v3comb.ccpmoho.iter.1.Z.npz       $W/v3c/v2v3comb/     # v2v3 combined Vsv
add $ASDF/v3c/smoomoho/Fvs.v3c.ccpmoho.smoomoho.iter.4.ZT.npz      $W/v3c/smoomoho/     # smooth xi (ZT)
# CCP refined Moho pick (.mat): red Moho line + --moho-grid for avg/amplitude field maps
add $CCPDIR/Moho.picked/moho_ccp_refined.final.mat                 $W/CCP/Moho.picked/
# MATLAB Vsv colormap (cmapx) for absolute-Vsv panels/maps
add $CMAPXSRC                                                      $W/cmapx.mat
# per-station Vp/Vs (HK, for --vpvs-track) + mineral deposits (gold/greenstone)
add $REPO/data/stations/HK.final.2026.merged.txt                  $W/stations/
for d in gold copper lithium nickel; do add $REPO/data/deposits/${d}_deposits.mat $W/deposits/; done
add $REPO/matlab/greenstones_sw_yilgarn.mat                       $W/deposits/
# TDM2 Sm-Nd age npz lives in the repo (data/geology/) — travels with the clone, not here
# optional: the big CCP RF cube for the CCP panel (--ccp); otherwise drop --ccp-pmat on the laptop
if [ $CCPCUBE = 1 ]; then
  mkdir -p $W/CCP/mat
  for f in $CCPDIR/mat/*pvh.4*.Z.P.mat $CCPDIR/mat/*pvh.4*.Z.CMP.mat; do add "$f" $W/CCP/mat/; done
fi
for c in Z T; do for t in group phase; do        # v3b surface-wave maps (small)
  mkdir -p $W/v3b/$c/${t}_v_tomo
  add $ASDF/v3b/$c/${t}_v_tomo/TransD.region.$t.npz $W/v3b/$c/${t}_v_tomo/
done; done
add $ST/figures/xsections/sections.txt              $W/stage6/figures/xsections/
add /workspace/Others.data/AusMoho2023/AR23-moho-hmp.txt  $W/moho/
add $ASDF/pathqc.v2/stations_v2.txt                 $W/stations/
add $ASDF/footprint/wa_array_footprint.npz          $W/footprint/
add $ASDF/footprint/wa_array_outline.txt            $W/footprint/
for f in wa_tectonics wa_lithology; do
  add /workspace/shape.files/derived/$f.npz $W/derived/
done
# crustal boundaries: derived/ if present, else the copy in the repo
REPO=$(cd "$(dirname "$0")/.." && pwd)
for c in /workspace/shape.files/derived/wa_crustal_boundaries.npz $REPO/data/wa_crustal_boundaries.npz; do
  [ -e $c ] && { add $c $W/derived/; break; }
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
