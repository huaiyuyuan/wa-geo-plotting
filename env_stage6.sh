# source /data/25/yuan/github/wa-geo-plotting/env_stage6.sh
# Paths for wa-geo-plotting runs on the stage-6 TransD models (see docs/CHEATSHEET.md).
export WGP=/data/25/yuan/github/wa-geo-plotting
export ST=/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/stage6
export FIG=$ST/figures                            # one sub-folder per model: $FIG/iter8, ...

# models (Fvs npz)
export FVS2=$ST/Fvs.iter.2.Z.npz                  # Rayleigh Vsv, with Moho (Vp/Vs fixed 1.73)
export FVS8=$ST/Fvs.iter.8.ZT.npz                 # radial anisotropy: Vsv pinned to iter2, xi inverted
export FVS10=$ST/Fvs.iter.10.Z.npz                # Vsv, no Moho
export FVS11=$ST/Fvs.iter.11.Z.npz                # Vsv + Vp/Vs inverted (1.67-1.88)

# inputs
export XS=$FIG/xsections/sections.txt             # NS1-7 / EW1-7 (picked 8 Oct 2026; .bak.* = older lists)
export MOHO=/workspace/Others.data/AusMoho2023/AR23-moho-hmp.txt

# legacy names used by older commands
export OUT=$ST
export SEC=$XS
export FVS_Z=$FVS2
export FVS_ZT=$FVS8

for f in $FVS2 $FVS8 $FVS10 $FVS11 $XS $MOHO; do
    [ -e "$f" ] || echo "env_stage6: missing $f"
done
cd $WGP
