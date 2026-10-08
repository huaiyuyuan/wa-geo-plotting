#!/usr/bin/env python3
"""
make_depth_maps.py - the full depth-slice set in one go (plot_vsv_tectonic.py):
every parameter that varies in each model, absolute and dln, at 10/20/30/45 km
(45 not 40: 40 km sits on the Moho artefact).

  ZT/xi model : Vsv (= Viso/sqrt((2+Xi^2)/3)), Viso, Xi   (abs + rel)
  Z model     : Vsv                (abs + rel)
  any model   : Vp/Vs only if it actually varies (fixed at 1.73 in current Z runs)

  python3 make_depth_maps.py --fvs-zt $FVS_ZT --fvs-z $FVS_Z --out-dir $OUT/figures/depth_maps
  # footprint focus is ON by default (--no-footprint for the full WA view)
  # extra options are passed to plot_vsv_tectonic.py, e.g.  -- --overlay filled

Outputs: <out-dir>/<tag>.<param>.<abs|rel>.<fp|wa>.png - fp = footprint focus (default),
wa = whole WA (--no-footprint); e.g. --tag-zt iter8 -> iter8.xi.rel.fp.png, iter8.xi.rel.wa.png
"""
import argparse, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def varies(d, key):
    return key in d.files and np.nanstd(d[key]) > 1e-4


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fvs-zt', help='xi (ZT) model Fvs npz')
    ap.add_argument('--fvs-z', help='Z model Fvs npz')
    ap.add_argument('--out-dir', default='figures/depth_maps')
    ap.add_argument('--tag-zt', default='ZT', help='filename prefix for the ZT model (e.g. iter8)')
    ap.add_argument('--tag-z', default='Z', help='filename prefix for the Z model')
    ap.add_argument('--depths', nargs='+', type=float, default=[10, 20, 30, 45])
    ap.add_argument('--modes', nargs='+', default=['abs', 'rel'], choices=['abs', 'rel'])
    ap.add_argument('--no-footprint', action='store_true', help='full WA view, no focus')
    ap.add_argument('--overlay', default='boundaries',
                    choices=['boundaries', 'outlines', 'both', 'filled', 'none'])
    ap.add_argument('--div-gap', type=float, default=0.07,
                    help='diverging maps skip +/- this band around white (0 = classic)')
    ap.add_argument('--div-white', type=int, default=2,
                    help='neutral colour levels at the centre of diverging maps (0 = none)')
    ap.add_argument('extra', nargs=argparse.REMAINDER,
                    help='after "--": further options for plot_vsv_tectonic.py')
    a = ap.parse_args()
    if not (a.fvs_zt or a.fvs_z):
        sys.exit('give --fvs-zt and/or --fvs-z')
    extra = [x for x in a.extra if x != '--']
    os.makedirs(a.out_dir, exist_ok=True)

    jobs = []
    for tag, path in ((a.tag_zt, a.fvs_zt), (a.tag_z, a.fvs_z)):
        if not path:
            continue
        d = np.load(path, allow_pickle=True)
        xi_model = varies(d, 'Xi')
        params = [('vsv_true', 'vsv'), ('vsv', 'viso'), ('xi', 'xi')] if xi_model \
            else [('vsv', 'vsv')]
        if varies(d, 'Vpvs'):
            params.append(('vpvs', 'vpvs'))
        skipped = [k for k in ('Xi', 'Vpvs') if k in d.files and not varies(d, k)]
        print(f"{tag}: {os.path.basename(path)} -> {', '.join(n for _, n in params)}"
              + (f"  (fixed, skipped: {', '.join(skipped)})" if skipped else ''))
        for field, name in params:
            for mode in a.modes:
                jobs.append((tag, path, field, name, mode))

    done, failed = [], []
    for tag, path, field, name, mode in jobs:
        extent = 'wa' if a.no_footprint else 'fp'      # whole WA vs footprint: never collide
        out = os.path.join(a.out_dir, f"{tag}.{name}.{mode}.{extent}.png")
        cmd = [sys.executable, os.path.join(HERE, 'plot_vsv_tectonic.py'), '--fvs', path,
               '--field', field, '--mode', mode, '--overlay', a.overlay, '--major', '--smooth',
               '--depths', *map(str, a.depths), '--out', out]
        cmd += ['--div-gap', str(a.div_gap), '--div-white', str(a.div_white)]
        if not a.no_footprint:
            cmd.append('--footprint')
        cmd += extra
        r = subprocess.run(cmd, capture_output=True, text=True)
        (done if r.returncode == 0 else failed).append(out)
        status = 'ok ' if r.returncode == 0 else 'ERR'
        print(f"  {status} {os.path.basename(out)}")
        if r.returncode:
            print('     ' + (r.stderr.strip().splitlines() or ['?'])[-1])
    print(f"{len(done)} maps in {a.out_dir}" + (f", {len(failed)} failed" if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
