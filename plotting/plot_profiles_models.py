#!/usr/bin/env python3
"""
plot_profiles_models.py — cross-sections along the picked `prof` transects, with
one landscape page per orientation (WE / NS / SENW) and ONE MODEL PER COLUMN.

    rows    = profiles in that group (map order)
    columns = models (--models), same field sampled the same way
    one page per group -> landscape figure, easy to drop a third model column later.

It reuses plot_xsection.py's sampling, colour rules, scaling and geology strips, so
the panels match the rest of the figure set. Profiles come from the polyline npz that
mat_to_profiles.py writes (true picked paths, so the curved WEST line plots honestly);
with --straight it uses the endpoints only.

Typical use (HY, 20 Oct 2026): v2 iter10 and v3c iter3, Vsv, no Moho ->

    source env_stage6.sh
    python3 tools/mat_to_profiles.py \
        /mnt/data64/workspace/CCP/CCP.WA.Array.2027.phase.3/profiles.latlon.phases.1.2.3.mat \
        --out-dir $FIG/profiles
    python3 plotting/plot_profiles_models.py \
        --profiles $FIG/profiles/profiles.latlon.phases.1.2.3.polylines.npz \
        --models $ST/Fvs.iter.10.Z.npz:"v2 iter10" \
                 /workspace/.../v3c/Fvs.v3c.iter.3.Z.npz:"v3c iter3" \
        --field vsv --d-max 60 --out-dir $FIG/profiles

Each group writes  profiles.<field>.<we|ns|senw>.png (+ .pdf).
v3c iter3 is SW-core only, so on the long WE/SENW lines its column blanks beyond
~0.5 deg from a node (--max-node-dist) while a statewide model fills — that gap is
the coverage, shown on purpose.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_xsection as px

def _slug(s):
    import re
    return re.sub(r'[^a-z0-9]+', '-', str(s).strip().lower()).strip('-') or 'model'


GROUP_TITLE = {'we': 'West-East', 'ns': 'North-South', 'senw': 'SE-NW'}
GROUP_ENDS  = {'we': ('W', 'E'), 'ns': ('S', 'N'), 'senw': ('SE', 'NW')}
FIELD_ARR   = {'vsv': 'Vsv', 'dvsv': 'Vsv', 'viso': 'Vsv', 'xi': 'Xi', 'vpvs': 'Vpvs'}


# ── model specs ───────────────────────────────────────────────────────────────
def parse_model(spec):
    """PATH[:LABEL[:VEL]] -> dict(path,label,vel). VEL in {auto,model,vsv_true}."""
    parts = spec.split(':')
    path = parts[0]
    label = parts[1] if len(parts) > 1 and parts[1] else None
    vel = parts[2] if len(parts) > 2 and parts[2] else 'auto'
    if label is None:
        import re
        m = re.search(r'iter\.?(\d+)', os.path.basename(path))
        label = f'iter{m.group(1)}' if m else os.path.basename(path).replace('.npz', '')
    return dict(path=path, label=label, vel=vel)


def moho_name(path):
    """Short caption for a Moho file: AR23 / CCP (merged) / the file stem."""
    b = os.path.basename(path).lower()
    if 'ar23' in b and 'ccp' not in b:
        return 'AusMoho AR23'
    if 'ccp' in b:
        return 'CCP pick' + (' (final merged)' if 'final' in b else '')
    return os.path.splitext(os.path.basename(path))[0]


def load_model(m, field):
    """Load one model, resolving 'auto' vel: an xi (ZT) model shown as vsv becomes
    true Vsv; otherwise the stored arrays are used as-is."""
    vel = m['vel']
    if vel == 'auto':
        raw = np.load(m['path'], allow_pickle=True)
        has_xi = 'Xi' in raw.files and np.nanstd(raw['Xi']) > 1e-4
        vel = 'vsv_true' if (field in ('vsv', 'dvsv') and has_xi) else 'model'
    d = px._load_fvs(m['path'], vel)
    d['_label'] = m['label']
    d['_vel'] = d.get('_vel', vel)
    return d


# ── geometry: walk a polyline as joined great-circle segments ─────────────────
def polyline_path(lons, lats, ds_deg):
    """(plat, plon, dist_km) along the polyline through all vertices."""
    lons = np.asarray(lons, float); lats = np.asarray(lats, float)
    if len(lons) < 2:
        raise ValueError("profile has < 2 vertices")
    PL, PO, DS = [], [], []
    base = 0.0
    for i in range(len(lons) - 1):
        la, lo, ds = px._great_circle_path(lats[i], lons[i], lats[i + 1], lons[i + 1], ds_deg)
        if i > 0:                      # drop the vertex shared with the previous segment
            la, lo, ds = la[1:], lo[1:], ds[1:]
        PL.append(la); PO.append(lo); DS.append(ds + base)
        base = DS[-1][-1] if len(DS[-1]) else base
    return np.concatenate(PL), np.concatenate(PO), np.concatenate(DS)


# ── profile list ──────────────────────────────────────────────────────────────
def load_profiles(path, straight=False):
    """From the polylines npz -> list of dict(short,name,group,colour,lon,lat,order).
    straight=True collapses each polyline to its two endpoints."""
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
    # fall back to a stock 5-col sections.txt (always straight, no colours/groups)
    secs = px._load_sections(path)
    profs = []
    for la1, lo1, la2, lo2, lab in secs:
        g = ('we' if lab.upper().startswith('WE') and lab.upper() != 'WEST'
             else 'senw' if lab.upper().startswith('SENW') or lab.upper() == 'WEST'
             else 'ns')
        profs.append(dict(short=lab, name=lab, group=g, colour=np.array([0.2, 0.2, 0.2]),
                          lon=np.array([lo1, lo2]), lat=np.array([la1, la2]), order=len(profs)))
    return profs


# ── one landscape page for a group ────────────────────────────────────────────
def plot_group(group, profs, models, field, z_by_model, out_dir, args):
    import matplotlib.pyplot as plt
    profs = sorted(profs, key=lambda p: p['order'])
    n = len(profs); ncol = len(models)
    arrname = FIELD_ARR[field]
    rel = field == 'dvsv'

    # sample every profile in every model, on that model's own depth grid (<= d_max)
    cell = {}          # (row i, col j) -> dict(dist, z, data)
    for i, p in enumerate(profs):
        plat, plon, dist = polyline_path(p['lon'], p['lat'], args.ds)
        p['_dist'] = dist; p['_plat'] = plat; p['_plon'] = plon
        for j, (m, d) in enumerate(zip(models, z_by_model)):
            z = d['z']
            if arrname not in d:
                cell[(i, j)] = dict(dist=dist, z=z, data=np.full((len(dist), len(z)), np.nan))
                continue
            arr = d[arrname]
            samp = px._sample_profile(d['Lon'], d['Lat'], arr, plon, plat)
            if rel:
                ref = np.nanmean(arr, axis=0)
                samp = (samp - ref[None, :]) / ref[None, :] * 100.0
            cell[(i, j)] = dict(dist=dist, z=z, data=samp)

    # absolute-velocity colormap: cmapx (HY's red-green-blue) or Spectral
    abs_cmap = px.cmapx(args.ncolors) if args.cmap == 'cmapx' \
        else px._cmap(px.CMAP_ABS, args.ncolors)
    # colour scale
    if field == 'vsv':
        cmap = abs_cmap; clab = 'Vsv (km/s)'; ext = 'both'
    elif field == 'dvsv':
        cmap = px._cmap('RdBu', args.ncolors); clab = 'dVsv (%)'; ext = 'both'
    elif field == 'vpvs':
        cmap = px._cmap(px.CMAP_VPVS, args.ncolors); clab = 'Vp/Vs'; ext = 'both'
    elif field == 'xi':
        cmap = px._cmap('RdBu', args.ncolors); clab = 'xi'; ext = 'both'
    else:
        cmap = abs_cmap; clab = field; ext = 'both'

    allvals = np.concatenate([c['data'][np.isfinite(c['data'])].ravel() for c in cell.values()
                              if np.isfinite(c['data']).any()]) if cell else np.array([])
    if args.vlim:
        vmn, vmx = args.vlim
    elif rel:
        vmn, vmx = -args.clim_rel, args.clim_rel
    elif field == 'xi':
        vmn, vmx = px.XI_LIM
    elif allvals.size:
        med, sd = np.nanmedian(allvals), np.nanstd(allvals)
        vmn, vmx = med - args.sigma_vsv * sd, med + args.sigma_vsv * sd
    else:
        vmn, vmx = None, None

    # layout: shared km/inch across the whole page (longest profile fills a column)
    maxd = max(p['_dist'][-1] for p in profs)
    zbot = max(d['z'][-1] for d in z_by_model)
    col_w = args.col_w
    km_per_in, row_h_in = px._section_scale(maxd, zbot, args.ve, col_w)

    strip_h = px._strip_height_in(px._GEO)
    gap_in = args.gap + strip_h
    top_in = 1.1 + strip_h + 0.25
    left_in = 1.0
    fig_h = n * row_h_in + (n - 1) * gap_in + top_in + 0.9
    pitch = maxd / km_per_in + 1.15                 # panel + colourbar + gap between columns
    fig_w = left_in + ncol * pitch + 0.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    endL, endR = GROUP_ENDS[group]
    scale_note = (f'fixed {vmn:.2f}-{vmx:.2f}' if args.vlim else
                  'shared median+-{:.0f}sd'.format(args.sigma_vsv) if field in ('vsv', 'vpvs')
                  else 'fixed')
    fig.suptitle(f'{GROUP_TITLE[group]} profiles — {clab}   '
                 f'[{n} lines x {ncol} models, depth 0-{zbot:.0f} km, '
                 f'col width {maxd:.0f} km, VE {args.ve:.0f}x, colour {scale_note}]',
                 fontsize=11, fontweight='bold')

    rh = row_h_in / fig_h
    geo_res = []
    im0 = None
    for j, (m, d) in enumerate(zip(models, z_by_model)):
        x0_in = left_in + j * pitch
        fig.text((x0_in + col_w / 2) / fig_w, 1 - 0.78 / fig_h, m['label'],
                 ha='center', va='top', fontsize=12, fontweight='bold')
        if m.get('_moho_name'):
            fig.text((x0_in + col_w / 2) / fig_w, 1 - 1.02 / fig_h, f"Moho: {m['_moho_name']}",
                     ha='center', va='top', fontsize=8, color='0.3')
        for i, p in enumerate(profs):
            c = cell[(i, j)]
            dist, z, data = c['dist'], c['z'], c['data']
            w_in = dist[-1] / km_per_in
            y0 = 1 - (top_in / fig_h) - (i + 1) * rh - i * (gap_in / fig_h)
            ax = fig.add_axes([x0_in / fig_w, y0, w_in / fig_w, rh])
            D, Z = np.meshgrid(dist, z, indexing='ij')
            im = ax.pcolormesh(D, Z, data, cmap=cmap, vmin=vmn, vmax=vmx, shading='auto')
            im0 = im
            ax.set_ylim(zbot, 0); ax.set_xlim(0, dist[-1])
            ax.tick_params(labelsize=6)
            ax.set_xticks(np.arange(0, dist[-1] + 1, 200))
            if j == 0:
                ax.set_ylabel('Depth', fontsize=7)
            else:
                ax.set_yticklabels([])
            # profile name chip in the map colour
            ax.text(0.004, 0.04, f"{p['short']} [{dist[-1]:.0f} km]", transform=ax.transAxes,
                    fontsize=8, fontweight='bold', ha='left', va='bottom', zorder=20,
                    color='white',
                    bbox=dict(fc=tuple(np.clip(p['colour'], 0, 1)), ec='none', alpha=0.95, pad=1.8))
            if np.isfinite(data).sum() == 0:
                ax.text(0.5, 0.5, 'no data in this model', transform=ax.transAxes,
                        ha='center', va='center', fontsize=7, color='0.4', style='italic')
            # geographic end labels on the bottom row
            if i == n - 1:
                ax.set_xlabel('Distance (km)', fontsize=8)
                ax.text(-0.005, -0.02, endL, transform=ax.transAxes, ha='right', va='top',
                        fontsize=8, fontweight='bold', color='0.3')
                ax.text(1.005, -0.02, endR, transform=ax.transAxes, ha='left', va='top',
                        fontsize=8, fontweight='bold', color='0.3')
            mi = m.get('_moho')
            if mi is not None:
                mp = np.array([float(np.ravel(mi(a, o))[0]) for a, o in zip(p['_plat'], p['_plon'])])
                px._draw_moho(ax, dist, mp, zbot, None if args.no_moho_mask else args.moho_mask)
            # per-panel colourbar
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
    mtag = args.tag or '_'.join(_slug(m['label']) for m in models)
    out = os.path.join(out_dir, f'profiles.{field}.{mtag}.{group}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  {group}: {out} (+ .pdf)  [{n} lines x {ncol} models]")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', required=True,
                    help='polylines npz from mat_to_profiles.py (or a stock sections.txt)')
    ap.add_argument('--models', nargs='+', required=True,
                    help='one or more PATH[:LABEL[:VEL]] — each becomes a column')
    ap.add_argument('--field', default='vsv', choices=list(FIELD_ARR),
                    help='vsv (default), dvsv (%% vs model mean), viso, xi, vpvs')
    ap.add_argument('--groups', nargs='+', default=['we', 'ns', 'senw'],
                    choices=['we', 'ns', 'senw'], help='which pages to make')
    ap.add_argument('--straight', action='store_true',
                    help='endpoints only (ignore the picked polyline vertices)')
    ap.add_argument('--d-max', type=float, default=60.0, dest='d_max')
    ap.add_argument('--ds', type=float, default=0.08, help='profile sampling step (deg)')
    ap.add_argument('--ve', type=float, default=px.DEFAULT_VE, help='vertical exaggeration')
    ap.add_argument('--col-w', type=float, default=7.0, dest='col_w',
                    help='inches the longest profile spans in each model column')
    ap.add_argument('--gap', type=float, default=0.55, help='inches between rows')
    ap.add_argument('--ncolors', type=int, default=16)
    ap.add_argument('--cmap', default='cmapx', choices=['cmapx', 'spectral'],
                    help="absolute-velocity colormap: cmapx (HY's red-green-blue, warm=slow; "
                         "default) or spectral")
    ap.add_argument('--sigma-vsv', type=float, default=1.5, dest='sigma_vsv',
                    help='abs colour range = median +- this*std over the page')
    ap.add_argument('--clim-rel', type=float, default=6.0, dest='clim_rel')
    ap.add_argument('--vlim', nargs=2, type=float, default=None, metavar=('VMIN', 'VMAX'),
                    help='force one colour range for every panel on every page')
    ap.add_argument('--max-node-dist', type=float, default=px.MAX_NODE_DIST, dest='max_node_dist',
                    help='blank profile points > this many deg from a model node (0 = off)')
    ap.add_argument('--moho', default=None, help='Moho file drawn on every column (default: no Moho)')
    ap.add_argument('--mohos', nargs='+', default=None, metavar='FILE|none',
                    help="one Moho file per --models entry, in the same order ('none' = no line); "
                         "overrides --moho, e.g. --mohos $MOHO $CMOHO")
    ap.add_argument('--no-moho-mask', action='store_true')
    ap.add_argument('--moho-mask', type=float, default=0.5, dest='moho_mask')
    ap.add_argument('--div-gap', type=float, default=0.20)
    ap.add_argument('--div-white', type=float, default=0.5)
    ap.add_argument('--strips', nargs='?', const='domain,boundaries,names',
                    default='domain,boundaries,names',
                    help='geology strips above each row (default domain,boundaries,names)')
    ap.add_argument('--no-strips', action='store_true')
    ap.add_argument('--litho-npz', default=None)
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--bnd-npz', default=None)
    ap.add_argument('--bnd-lines', action='store_true')
    ap.add_argument('--domain-colours', default='map2022')
    ap.add_argument('--out-dir', default='.', dest='out_dir')
    ap.add_argument('--tag', default=None,
                    help='model tag in the file name (default: the model labels slugged '
                         'and joined, e.g. v2-iter10_v3c-iter3)')
    args = ap.parse_args()

    px.MAX_NODE_DIST = args.max_node_dist
    px.DIV_GAP = args.div_gap; px.DIV_WHITE = args.div_white

    # geology strips (quietly skipped if the npz caches are not on this machine)
    px._GEO = None
    if not args.no_strips and args.strips:
        try:
            px._GEO = px._load_geology(args.strips, args.litho_npz, args.tect_npz,
                                       args.bnd_npz, args.bnd_lines, args.domain_colours)
        except SystemExit as e:
            sys.exit(str(e))
        except Exception as e:
            print(f"  strips unavailable ({e}); drawing without them")
            px._GEO = None

    models = [parse_model(s) for s in args.models]

    # Moho per model column: --mohos (one per model, 'none' = no line) else --moho for all
    px.plot_section._moho_interp = None
    if args.mohos and len(args.mohos) != len(models):
        sys.exit(f"--mohos needs one entry per model ({len(models)}), got {len(args.mohos)}")
    cache = {}
    for j, m in enumerate(models):
        f = args.mohos[j] if args.mohos else args.moho
        m['_moho'] = m['_moho_name'] = None
        if not f or f.lower() == 'none':
            continue
        if f not in cache:
            try:
                cache[f] = px.load_moho_interp(f)
            except Exception as e:
                print(f"  Moho load failed ({f}): {e}")
                cache[f] = None
        m['_moho'] = cache[f]
        if cache[f] is not None:
            m['_moho_name'] = moho_name(f)
            print(f"  {m['label']}: Moho {f}")
    print("Models (columns):")
    loaded = []
    for m in models:
        d = load_model(m, args.field)
        z = np.asarray(d['z'], float)
        keep = z <= args.d_max
        nz = len(z)
        for k in list(d.keys()):                      # trim every depth-indexed array
            v = d[k]
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == nz:
                d[k] = v[:, keep]
        d['z'] = z[keep]
        loaded.append(d)
        velnote = f" ({d['_vel']})" if d.get('_vel') not in (None, 'model') else ''
        print(f"  {m['label']}{velnote} <- {m['path']}  "
              f"[{len(d['Lon'])} nodes, {FIELD_ARR[args.field]} "
              f"{'present' if FIELD_ARR[args.field] in d else 'MISSING'}]")

    profs = load_profiles(args.profiles, straight=args.straight)
    have = sorted({p['group'] for p in profs})
    print(f"Profiles: {len(profs)} in groups {have} (from {os.path.basename(args.profiles)})")

    made = []
    for g in args.groups:
        gp = [p for p in profs if p['group'] == g]
        if not gp:
            print(f"  {g}: no profiles — skipped")
            continue
        made.append(plot_group(g, gp, models, args.field, loaded, args.out_dir, args))
    if not made:
        sys.exit("No pages made (no profiles matched the requested groups).")
    print(f"Done: {len(made)} page(s).")


if __name__ == '__main__':
    main()
