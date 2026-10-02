# source $WGP/env_stage6.sh  - paths for wa-geo-plotting runs on stage-6 models
export WGP=/data/25/yuan/github/wa-geo-plotting
export OUT=/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/stage6
export FVS_Z=$OUT/Fvs.iter.2.Z.npz
export FVS_ZT=$OUT/Fvs.iter.3.ZT.npz
export SEC=$OUT/figures/xsections/sections.txt
export MOHO=/workspace/Others.data/AusMoho2023/AR23-moho-hmp.txt
for f in $FVS_Z $FVS_ZT $SEC $MOHO; do [ -e "$f" ] || echo "env_stage6: missing $f"; done
