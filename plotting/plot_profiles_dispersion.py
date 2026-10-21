#!/usr/bin/env python3
"""
plot_profiles_dispersion.py — dispersion sections along the picked `prof` transects.
Same profiles as plot_profiles_models.py, but the y axis is PERIOD, not depth:

    rows    = profiles in that group (WE / NS / SENW, map order)
    columns = data sets  (observed, each model's synthetics, optional residuals)
    one landscape page per orientation PER COMPONENT (cR / uR / cL / uL).

Each panel is distance (km, along the line) x period (s, log), coloured by surface-wave
velocity. It reuses plot_site_period_maps.load_set / plot_site_fits.read_table to read the
node files, and plot_xsection.py for the geometry, colours and geology strips — so the node
values are interpolated onto each profile exactly as the Vsv maps are.

Inputs (globs, quote them; node position from 'x.<lon>.y.<lat>' in each path):
    --data LABEL GLOB   observed  Data_SWD.in.*   (columns 'p cR uR cL uL')
    --syn  LABEL GLOB   synthetic data_bestd.out* (columns 'cR uR cL uL p')
both repeatable; --data-order / --syn-order change the column layout.

Typical use (v2 iter10 and v3c iter3, each vs its own observed; HY, Oct 2026):

    source env_stage6.sh
    A=/workspace/WA.Array/Year.3/noise.processing/ASDF.processing
    python3 plotting/plot_profiles_dispersion.py \
        --profiles $FIG/profiles/profiles.latlon.phases.1.2.3.polylines.npz \
        --data "v2 obs"    "$ST/IN/Data_SWD.in.*" \
        --syn  "v2 iter10" "$ST/OUT.iter.10.Z/data_bestd.out*" \
        --data "v3 obs"    "$A/v3c/IN_v3nodes/Data_SWD.in.*" \
        --syn  "v3c iter3" "$A/v3c/OUT.iter.3.Z/data_bestd.out*" \
        --comp cR --resid --colour both --out-dir $FIG/profiles/dispersion

Writes  profiles.disp.<comp>.<abs|dv[|resid]>.<tag>.<we|ns|senw>.png (+ .pdf).
Points > --max-node-dist deg from a node are left blank (the coverage), as on the maps.
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))
import plot_xsection as px                                    # geometry, colours, strips
from plot_site_period_maps import load_set, NAME             # node dispersion -> V[node,period]

GROUP_TITLE = {'we': 'West-East', 'ns': 'North-South', 'senw': 'SE-NW'}
GROUP_ENDS  = {'we': ('W', 'E'), 'ns': ('S', 'N'), 'senw': ('SE', 'NW')}


def _slug(s):
    import re
    return re.sub(r'[^a-z0-9]+', '-', str(s).strip().lower()).strip('-') or 'set'


def polyline_path(lons, lats, ds_deg):
    lons = np.asarray(lons, float); lats = np.asarray(lats, float)
    if len(lons) < 2:
        raise ValueError("profile has < 2 vertices")
    PL, PO, DS, base = [], [], [], 0.0
    for i in range(len(lons) - 1):
        la, lo, ds = px._great_circle_path(lats[i], lons[i], lats[i + 1], lons[i + 1], ds_deg)
        if i > 0:
            la, lo, ds = la[1:], lo[1:], ds[1:]
        PL.append(la); PO.append(lo); DS.append(ds + base)
        base = DS[-1][-1] if len(DS[-1]) else base
    return np.concatenate(PL), np.concatenate(PO), np.concatenate(DS)


def load_profiles(path, straight=False):
    if path.endswith('.npz'):
        d = np.load(path, allow_pickle=True)
        profs = []
        for i in range(len(d['short'])):
            lo = np.asarray(d['lon'][i], float); la = np.asarray(d['lat'][i], float)
            if straight:
                lo = np.array([lo[0], lo[-1]]); la = np.array([la[0], la[-1]])
            profs.append(dict(short=str(d['short'][i]), name=str(d['name'][i]),
                              group=str(d['group'][i]), colour=np.asarray(d['colour'][i], float),
                              lon=lo, lat=la, order=int(d['order'][i])))
        return profs
    secs = px._load_sections(path)
    profs = []
    for la1, lo1, la2, lo2, lab in secs:
        u = lab.upper()
        g = ('senw' if u.startswith('SENW') or u == 'WEST'
             else 'we' if u.startswith('WE') else 'ns')
        profs.append(dict(short=lab, name=lab, group=g, colour=np.array([0.2, 0.2, 0.2]),
                          lon=np.array([lo1, lo2]), lat=np.array([la1, la2]), order=len(profs)))
    return profs


def sample_set(S, comp, periods, plon, plat):
    """Interpolate a loaded node set's V[node,period] onto a profile -> [nprof, nperiod]."""
    return px._sample_profile(S['lon'], S['lat'], S['V'], plon, plat)


def plot_group(group, profs, columns, comp, periods, mode, out_dir, args):
    """One landscape page. columns = list of dict(label, kind, data=[nprof,nper] per profile).
    mode in {'abs','dv','resid'} sets the colour field and map."""
    import matplotlib.pyplot as plt
    profs = sorted(profs, key=lambda p: p['order'])
    n = len(profs); ncol = len(columns)
    lp = np.log(np.asarray(periods, float))
    pmin, pmax = periods[0], periods[-1]

    if mode == 'abs':
        cmap = px._cmap(px.CMAP_ABS, args.ncolors); clab = f'{NAME[comp]} (km/s)'; ext = 'both'
    elif mode == 'dv':
        cmap = px._cmap('RdBu', args.ncolors); clab = f'{NAME[comp]} dV (%)'; ext = 'both'
    elif mode == 'diff':
        cmap = px._cmap('RdBu', args.ncolors); clab = f'{NAME[comp]} difference (km/s)'; ext = 'both'
    else:  # resid
        cmap = px._cmap('RdBu', args.ncolors); clab = f'{NAME[comp]} syn-obs (km/s)'; ext = 'both'

    # colour range (shared across the page)
    allv = np.concatenate([c['data'][i][np.isfinite(c['data'][i])].ravel()
                           for c in columns for i in range(n)
                           if np.isfinite(c['data'][i]).any()]) if columns else np.array([])
    if args.vlim and mode == 'abs':
        vmn, vmx = args.vlim
    elif mode == 'dv':
        vmn, vmx = -args.clim_rel, args.clim_rel
    elif mode == 'diff':
        vmn, vmx = -args.clim_diff, args.clim_diff
    elif mode == 'resid':
        vmn, vmx = -args.clim_resid, args.clim_resid
    elif allv.size:
        vmn, vmx = np.nanpercentile(allv, 2), np.nanpercentile(allv, 98)
    else:
        vmn, vmx = None, None

    # layout (period axis is fixed height per row; x shares one km/inch)
    maxd = max(p['_dist'][-1] for p in profs)
    col_w = args.col_w
    km_per_in = maxd / col_w
    row_h_in = args.row_h

    strip_h = px._strip_height_in(px._GEO)
    gap_in = args.gap + strip_h
    top_in = 1.1 + strip_h + 0.25
    left_in = 1.0
    fig_h = n * row_h_in + (n - 1) * gap_in + top_in + 0.9
    pitch = maxd / km_per_in + 1.15
    fig_w = left_in + ncol * pitch + 0.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    endL, endR = GROUP_ENDS[group]
    fig.suptitle(f'{GROUP_TITLE[group]} dispersion — {clab}   '
                 f'[{n} lines x {ncol} sets, period {pmin:g}-{pmax:g} s (log), '
                 f'col width {maxd:.0f} km]', fontsize=11, fontweight='bold')

    rh = row_h_in / fig_h
    geo_res = []
    for j, c in enumerate(columns):
        x0_in = left_in + j * pitch
        fig.text((x0_in + col_w / 2) / fig_w, 1 - 0.78 / fig_h, c['label'],
                 ha='center', va='top', fontsize=12, fontweight='bold')
        for i, p in enumerate(profs):
            dist = p['_dist']; data = c['data'][i]
            w_in = dist[-1] / km_per_in
            y0 = 1 - (top_in / fig_h) - (i + 1) * rh - i * (gap_in / fig_h)
            ax = fig.add_axes([x0_in / fig_w, y0, w_in / fig_w, rh])
            D, P = np.meshgrid(dist, periods, indexing='ij')
            im = ax.pcolormesh(D, P, data, cmap=cmap, vmin=vmn, vmax=vmx, shading='auto')
            ax.set_yscale('log')
            ax.set_ylim(pmax, pmin)                       # short period on top (like depth 0)
            ax.set_xlim(0, dist[-1]); ax.tick_params(labelsize=6)
            ax.set_xticks(np.arange(0, dist[-1] + 1, 200))
            from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
            ax.yaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 30, 50, 80, 120]))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax.yaxis.set_minor_formatter(NullFormatter())
            if j == 0:
                ax.set_ylabel('Period (s)', fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.text(0.004, 0.04, f"{p['short']} [{dist[-1]:.0f} km]", transform=ax.transAxes,
                    fontsize=8, fontweight='bold', ha='left', va='bottom', zorder=20, color='white',
                    bbox=dict(fc=tuple(np.clip(p['colour'], 0, 1)), ec='none', alpha=0.95, pad=1.8))
            if np.isfinite(data).sum() == 0:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center',
                        fontsize=7, color='0.4', style='italic')
            if i == n - 1:
                ax.set_xlabel('Distance (km)', fontsize=8)
                ax.text(-0.005, -0.02, endL, transform=ax.transAxes, ha='right', va='top',
                        fontsize=8, fontweight='bold', color='0.3')
                ax.text(1.005, -0.02, endR, transform=ax.transAxes, ha='left', va='top',
                        fontsize=8, fontweight='bold', color='0.3')
            cax = fig.add_axes([(x0_in + w_in + 0.10) / fig_w, y0, 0.12 / fig_w, rh])
            cb = fig.colorbar(im, cax=cax, extend=ext); cb.ax.tick_params(labelsize=6)
            if i == 0:
                cb.set_label(clab, fontsize=8)
            if px._GEO is not None:
                geo_res.append(px._add_geology(ax, [ax], p['_plon'], p['_plat'], dist,
                                               names=None if j == 0 else False))

    if geo_res and any(g for g in geo_res):
        try:
            px._geology_legend(fig, geo_res, ax, below_in=0.55)
        except Exception:
            pass
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or getattr(args, '_settag', '') or \
        '_'.join(_slug(c['label']) for c in columns)
    out = os.path.join(out_dir, f'profiles.disp.{comp}.{mode}.{tag}.{group}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  {group}/{comp}/{mode}: {out} (+ .pdf)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', required=True, help='polylines npz (or a stock sections.txt)')
    ap.add_argument('--data', nargs=2, action='append', metavar=('LABEL', 'GLOB'), default=[],
                    help='observed set: label and a glob of Data_SWD.in.* (repeatable)')
    ap.add_argument('--syn', nargs=2, action='append', metavar=('LABEL', 'GLOB'), default=[],
                    help='synthetic set: label and a glob of data_bestd.out* (repeatable)')
    ap.add_argument('--data-order', default='p cR uR cL uL')
    ap.add_argument('--syn-order', default='cR uR cL uL p')
    ap.add_argument('--comp', nargs='+', default=['cR'], choices=list(NAME),
                    help='one or more of cR uR cL uL (one page set each)')
    ap.add_argument('--rayleigh', action='store_true',
                    help='shortcut for --comp cR uR (Rayleigh phase + group only)')
    ap.add_argument('--diff', nargs=2, action='append', metavar=('LABEL_A', 'LABEL_B'), default=[],
                    help='difference two loaded sets along the profiles: a page of A - B (km/s), '
                         'e.g. --diff "v2 obs" "v3 obs" to see where the input data differ. '
                         'Repeatable (each pair is a column); labels must match --data/--syn labels')
    ap.add_argument('--periods', nargs='+', type=float, default=None,
                    help='explicit period axis (s); default: log-spaced --pmin..--pmax')
    ap.add_argument('--pmin', type=float, default=2.5)
    ap.add_argument('--pmax', type=float, default=50.0)
    ap.add_argument('--nper', type=int, default=40, help='period samples when --periods not given')
    ap.add_argument('--colour', default='both', choices=['abs', 'dv', 'both'])
    ap.add_argument('--resid', action='store_true', help='add syn-minus-obs columns')
    ap.add_argument('--ref-domain', default=None,
                    help='dV reference = this domain mean at each period (default: all nodes)')
    ap.add_argument('--groups', nargs='+', default=['we', 'ns', 'senw'],
                    choices=['we', 'ns', 'senw'])
    ap.add_argument('--straight', action='store_true')
    ap.add_argument('--ds', type=float, default=0.08)
    ap.add_argument('--col-w', type=float, default=7.0, dest='col_w')
    ap.add_argument('--row-h', type=float, default=1.6, dest='row_h',
                    help='inches per profile row (period panels)')
    ap.add_argument('--gap', type=float, default=0.55)
    ap.add_argument('--ncolors', type=int, default=16)
    ap.add_argument('--clim-rel', type=float, default=6.0, dest='clim_rel')
    ap.add_argument('--clim-resid', type=float, default=0.15, dest='clim_resid')
    ap.add_argument('--clim-diff', type=float, default=0.20, dest='clim_diff',
                    help='+-km/s colour range for --diff pages')
    ap.add_argument('--vlim', nargs=2, type=float, default=None, metavar=('VMIN', 'VMAX'))
    ap.add_argument('--max-node-dist', type=float, default=px.MAX_NODE_DIST, dest='max_node_dist')
    ap.add_argument('--div-gap', type=float, default=0.20)
    ap.add_argument('--div-white', type=float, default=0.5)
    ap.add_argument('--strips', nargs='?', const='domain,boundaries,names',
                    default='domain,boundaries,names')
    ap.add_argument('--no-strips', action='store_true')
    ap.add_argument('--litho-npz', default=None)
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--bnd-npz', default=None)
    ap.add_argument('--bnd-lines', action='store_true')
    ap.add_argument('--domain-colours', default='map2022')
    ap.add_argument('--out-dir', default='.', dest='out_dir')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    if not args.data and not args.syn:
        sys.exit("give at least one --data or --syn set")
    px.MAX_NODE_DIST = args.max_node_dist
    px.DIV_GAP = args.div_gap; px.DIV_WHITE = args.div_white

    px._GEO = None
    if not args.no_strips and args.strips:
        try:
            px._GEO = px._load_geology(args.strips, args.litho_npz, args.tect_npz,
                                       args.bnd_npz, args.bnd_lines, args.domain_colours)
        except SystemExit as e:
            sys.exit(str(e))
        except Exception as e:
            print(f"  strips unavailable ({e}); drawing without them")

    periods = (np.asarray(args.periods, float) if args.periods
               else np.logspace(np.log10(args.pmin), np.log10(args.pmax), args.nper))
    periods = np.sort(periods)

    # one filename tag for every page of this run (from the set labels)
    args._settag = '_'.join(_slug(l) for l, _ in (args.data + args.syn))

    # load every set once per component (node V on the period grid)
    profs = load_profiles(args.profiles, straight=args.straight)
    have = sorted({p['group'] for p in profs})
    print(f"Profiles: {len(profs)} in groups {have}")

    # precompute profile paths
    for p in profs:
        plat, plon, dist = polyline_path(p['lon'], p['lat'], args.ds)
        p['_plat'], p['_plon'], p['_dist'] = plat, plon, dist

    comps = ['cR', 'uR'] if args.rayleigh else args.comp

    made = []
    for comp in comps:
        print(f"Component {comp} ({NAME[comp]}):")
        obs_sets = [load_set(l, g, args.data_order.split(), comp, periods, 'observed')
                    for l, g in args.data]
        syn_sets = [load_set(l, g, args.syn_order.split(), comp, periods, 'synthetic')
                    for l, g in args.syn]
        sets = obs_sets + syn_sets
        by_label = {S['label']: S for S in sets}
        for a_lab, b_lab in args.diff:                     # validate --diff labels up front
            for lab in (a_lab, b_lab):
                if lab not in by_label:
                    sys.exit(f"--diff: no set labelled '{lab}'. Loaded: {list(by_label)}")

        # build per-group columns (sampled onto each profile)
        for g in args.groups:
            gp = [p for p in profs if p['group'] == g]
            if not gp:
                continue
            cols_abs = []
            for S in sets:
                data = [sample_set(S, comp, periods, p['_plon'], p['_plat']) for p in gp]
                cols_abs.append(dict(label=S['label'], kind=S['kind'], data=data))

            if args.colour in ('abs', 'both'):
                made.append(plot_group(g, gp, cols_abs, comp, periods, 'abs', args.out_dir, args))

            if args.colour in ('dv', 'both'):
                # reference at each period: mean over the first observed set's nodes (or ref-domain)
                refset = obs_sets[0] if obs_sets else sets[0]
                ref = np.nanmean(refset['V'], axis=0)
                cols_dv = []
                for c in cols_abs:
                    dv = [(np.asarray(d) / ref[None, :] - 1.0) * 100.0 for d in c['data']]
                    cols_dv.append(dict(label=c['label'], kind=c['kind'], data=dv))
                made.append(plot_group(g, gp, cols_dv, comp, periods, 'dv', args.out_dir, args))

            if args.resid and obs_sets and syn_sets:
                # syn - obs along each profile; pair the k-th syn with the k-th observed set
                # (v2 syn vs v2 obs, v3 syn vs v3 obs), falling back to the first observed.
                obs_sampled = {}
                def _obs(S):
                    if id(S) not in obs_sampled:
                        obs_sampled[id(S)] = [sample_set(S, comp, periods, p['_plon'], p['_plat'])
                                              for p in gp]
                    return obs_sampled[id(S)]
                cols_r = []
                for k, S in enumerate(syn_sets):
                    O = obs_sets[k] if k < len(obs_sets) else obs_sets[0]
                    ob = _obs(O)
                    sy = [sample_set(S, comp, periods, p['_plon'], p['_plat']) for p in gp]
                    rr = [sy[i] - ob[i] for i in range(len(gp))]
                    cols_r.append(dict(label=f"{S['label']} - {O['label']}", kind='resid', data=rr))
                made.append(plot_group(g, gp, cols_r, comp, periods, 'resid', args.out_dir, args))

            if args.diff:
                # difference two named sets along each profile (e.g. v2 obs - v3 obs):
                # where the input data themselves differ between versions.
                diff_sampled = {}
                def _samp(S):
                    if id(S) not in diff_sampled:
                        diff_sampled[id(S)] = [sample_set(S, comp, periods, p['_plon'], p['_plat'])
                                               for p in gp]
                    return diff_sampled[id(S)]
                cols_d = []
                for a_lab, b_lab in args.diff:
                    A, B = by_label[a_lab], by_label[b_lab]
                    sa, sb = _samp(A), _samp(B)
                    dd = [sa[i] - sb[i] for i in range(len(gp))]
                    cols_d.append(dict(label=f"{a_lab} - {b_lab}", kind='diff', data=dd))
                made.append(plot_group(g, gp, cols_d, comp, periods, 'diff', args.out_dir, args))

    if not made:
        sys.exit("No pages made.")
    print(f"Done: {len(made)} page(s).")


if __name__ == '__main__':
    main()
