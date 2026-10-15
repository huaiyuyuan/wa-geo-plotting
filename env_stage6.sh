# source <repo>/env_stage6.sh
# Paths for wa-geo-plotting runs on the stage-6 TransD models (see docs/CHEATSHEET.md).
#   d204403 : just source it (data under /workspace).
#   elsewhere (basin, mantle, laptop): unpack the bundle from tools/pack_data.sh, then
#             export WA_DATA=~/wa_data ; source <repo>/env_stage6.sh
export WGP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -n "$WA_DATA" ]; then                        # portable bundle
    export ST=$WA_DATA/stage6
    export V3B=$WA_DATA/v3b
    export MOHO=$WA_DATA/moho/AR23-moho-hmp.txt
    export WA_DERIVED=$WA_DATA/derived            # read by config.py
    export WA_STATIONS=$WA_DATA/stations/stations_v2.txt
    export WA_FOOTPRINT_DIR=$WA_DATA/footprint
    [ -d $WA_DATA/cartopy_data ] && export CARTOPY_DATA_DIR=$WA_DATA/cartopy_data
else                                              # d204403
    export ST=/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/stage6
    export V3B=/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/v3b
    export MOHO=/workspace/Others.data/AusMoho2023/AR23-moho-hmp.txt
    export WA_DERIVED=${WA_DERIVED:-/workspace/shape.files/derived}
fi
export WA_MOHO=$MOHO
export WA_TECTONICS_NPZ=$WA_DERIVED/wa_tectonics.npz
export WA_BOUNDARIES_NPZ=$WA_DERIVED/wa_crustal_boundaries.npz
[ -e $WA_BOUNDARIES_NPZ ] || export WA_BOUNDARIES_NPZ=$WGP/data/wa_crustal_boundaries.npz  # copy in the repo
export FIG=${FIG:-$ST/figures}                    # one sub-folder per model: $FIG/iter8, ...

# models (Fvs npz)
export FVS2=$ST/Fvs.iter.2.Z.npz                  # Rayleigh Vsv, with Moho (Vp/Vs fixed 1.73)
export FVS8=$ST/Fvs.iter.8.ZT.npz                 # radial anisotropy: Vsv pinned to iter2, xi inverted
export FVS10=$ST/Fvs.iter.10.Z.npz                # Vsv, no Moho
export FVS11=$ST/Fvs.iter.11.Z.npz                # setup 4: Vsv + Vp/Vs, Rayleigh only; band80, provisional
export FVS12=$ST/Fvs.iter.12.ZT.npz               # Viso + xi (ZT); scripts derive Vsv = Viso/sqrt((2+xi^2)/3)
# v3 (band120, 25 km, 1682 nodes, SW core only): current model = iter2.sm (16 Oct 2026),
# warm Vsv-only (setup 4) iteration from the smooth, kink-free REF; supersedes FVS11 and the
# first v3 runs Fvs.v3.iter.1/2.Z.npz (AK135 REF, crustal kinks). Depth 0-80 km.
export V3=${V3:-$(dirname $ST)/v3}
export FVSV3=${FVSV3:-$V3/Fvs.v3.iter.2.sm.Z.npz}

# surface-wave maps (v3b band120 iter-2): $V3B/{Z,T}/{group,phase}_v_tomo/TransD.region.*.npz
# coverage mask: footprint npz key 'density' (config.FOOTPRINT_NPZ)

# inputs
export XS=$ST/figures/xsections/sections.txt      # NS1-7 / EW1-7 (picked 8 Oct 2026; .bak.* = older lists)

# legacy names used by older commands
export OUT=$ST
export SEC=$XS
export FVS_Z=$FVS2
export FVS_ZT=$FVS8

for f in $FVS2 $FVS8 $FVS10 $FVS11 $FVS12 $FVSV3 $XS $MOHO $WA_TECTONICS_NPZ $WA_BOUNDARIES_NPZ; do
    [ -e "$f" ] || echo "env_stage6: missing $f"
done
cd $WGP
