#!/usr/bin/env python3
"""
xi_endowment_stats.py  --  quantitative test of the xi / gold-endowment association.

Samples the SAME depth-window-averaged radial-anisotropy field xi = (Vsh/Vsv)^2 that
plot_wa_map.py --xiavg draws (same _avg_model_window, same Moho surface, same 0.8-node
smoothing), then compares xi between geological populations and reports separability.

Populations (all restricted to the data footprint = finite masked Moho, and to the SW
side of the --cut-poly, so the northern Pilbara/Hamersley gold is excluded as on the maps):

  GOLD        xi sampled at gold-deposit coordinates (matlab/gold_deposits.mat)
  GREENSTONE  grid nodes inside greenstone polygons (yilgarn_units.npz)
  GRANITE     Yilgarn grid nodes NOT in greenstone (granite/gneiss/sedimentary)  [background]
  CRATON      all grid nodes inside the Yilgarn (craton outline, or union of units)
  MARGIN      footprint grid nodes OUTSIDE the Yilgarn (Proterozoic orogens etc.)

For each depth window and each comparison it prints n, mean, median, and the
separability: Mann-Whitney U -> AUC (prob. a random A-point has HIGHER xi than a random
B-point; 0.5 = no separation, >0.5 = A higher), plus a two-sample KS statistic and p.

It renders nothing (no cartopy / display needed) and writes a CSV of every comparison.

Run on the workstation (paths as in docs/wa-array-maps-cheatsheet.md), from the repo root:

  A=/workspace/WA.Array/Year.3/noise.processing/ASDF.processing
  CCP=/workspace/CCP/CCP.WA.Array.2027.phase.3
  python3 plotting/xi_endowment_stats.py \
      --xi   $A/v3c/Fvs.v3c.ccpmoho.iter.4.ZT.npz \
      --moho $CCP/Moho.picked/moho_ccp_refined.final.mat \
      --cut-poly data/masks/wa_array_cut.txt \
      --windows "0,moho" "20,moho" "moho,60" "0,10" \
      --out figures/xi-endowment/xi_endowment_stats.csv
"""
import os
import sys
import argparse
import numpy as np

# reuse the map/section internals so the sampling is identical to the figures
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from plot_wa_map import _avg_model_window, _read_moho_mat, _read_cut_poly, _keep_poly  # noqa: E402


# ----------------------------------------------------------------------------- loaders
def _load_gold(deposit_dir, gold_file=None):
    """Gold deposit lon/lat (+endowment size). Same source as deposits.py / the maps:
    <deposit_dir>/gold_deposits.mat, struct 'gold', fields lon/lat/size. Deposits moved
    to data/deposits/ (29 Oct 2026, was matlab/) so we search deposit_dir, then
    data/deposits, then matlab, relative to the repo root."""
    from scipy.io import loadmat
    if gold_file and os.path.isfile(gold_file):
        g = loadmat(gold_file, squeeze_me=True, struct_as_record=False)
        g = g.get('gold', next(v for k, v in g.items() if not k.startswith('__')))
    else:
        cands = []
        for dd in (deposit_dir, os.path.join(_HERE, '..', 'data', 'deposits'),
                   os.path.join(_HERE, '..', 'matlab')):
            if dd:
                cands.append(os.path.join(dd, 'gold_deposits.mat'))
        path = next((p for p in cands if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError('gold deposits not found; looked in: '
                                    + ', '.join(cands) + '  (set --deposit-dir or --gold-file)')
        g = loadmat(path, squeeze_me=True, struct_as_record=False)['gold']
        print(f"  gold .mat: {path}")
    lon = np.asarray(g.lon, float).ravel()
    lat = np.asarray(g.lat, float).ravel()
    sz = np.asarray(g.size, float).ravel() if hasattr(g, 'size') else np.ones_like(lon)
    ok = np.isfinite(lon) & np.isfinite(lat)
    return lon[ok], lat[ok], sz[ok]


def _unit_paths(units_npz, rocktypes):
    """Compound matplotlib Paths for the requested rocktypes in yilgarn_units.npz,
    grouped by rec_idx (outer ring + holes), exactly as _yilgarn_units_draw groups them."""
    from matplotlib.path import Path
    from collections import defaultdict
    d = np.load(units_npz, allow_pickle=True)
    rings = d['rings']
    rec = d['rec_idx']
    rock = d['rocktype'].astype(str)
    want = set(rocktypes)
    groups = defaultdict(list)
    for i in range(len(rings)):
        if rock[i] in want:
            groups[int(rec[i])].append(np.asarray(rings[i], float))
    paths = []
    for rr in groups.values():
        paths.append(Path.make_compound_path(*[Path(np.asarray(r, float)) for r in rr]))
    return paths


def _in_any(paths, pts):
    """Boolean: is each (lon,lat) point inside ANY of the compound paths (holes respected
    per compound path via the even-odd rule)."""
    pts = np.asarray(pts, float)
    inside = np.zeros(len(pts), bool)
    for p in paths:
        if inside.all():
            break
        inside |= p.contains_points(pts)
    return inside


def _craton_path(bnd_npz):
    """Try to build a single closed craton polygon by greedily stitching the 'craton'
    boundary segments end-to-end. Returns a matplotlib Path or None if it can't close."""
    from matplotlib.path import Path
    try:
        d = np.load(bnd_npz, allow_pickle=True)
    except Exception:
        return None
    if 'lines' not in d.files or 'kind' not in d.files:
        return None
    lines = [np.asarray(l, float) for l, k in zip(d['lines'], d['kind'].astype(str))
             if k == 'craton']
    if not lines:
        return None
    segs = [l for l in lines if len(l) >= 2]
    if not segs:
        return None
    chain = [segs.pop(0)]

    def _end(c):
        return c[-1][-1]
    tol = 0.25  # deg; stitch tolerance
    while segs:
        tail = chain[-1][-1]
        best, rev, dmin = None, False, np.inf
        for j, s in enumerate(segs):
            for r, pt in ((False, s[0]), (True, s[-1])):
                dd = np.hypot(*(pt - tail))
                if dd < dmin:
                    dmin, best, rev = dd, j, r
        if best is None or dmin > tol:
            break
        s = segs.pop(best)
        chain.append(s[::-1] if rev else s)
    ring = np.vstack(chain)
    # closed enough?
    if np.hypot(*(ring[0] - ring[-1])) > 1.0:
        return None
    return Path(np.vstack([ring, ring[0]]))


# ------------------------------------------------------------------------------- stats
def _auc_mwu(a, b):
    """AUC = P(random a > random b) via Mann-Whitney U; plus two-sided p. KS too."""
    from scipy.stats import mannwhitneyu, ks_2samp
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return dict(n_a=a.size, n_b=b.size, auc=np.nan, p_mwu=np.nan, ks=np.nan, p_ks=np.nan,
                    mean_a=np.nanmean(a) if a.size else np.nan,
                    mean_b=np.nanmean(b) if b.size else np.nan,
                    med_a=np.nanmedian(a) if a.size else np.nan,
                    med_b=np.nanmedian(b) if b.size else np.nan)
    U, p = mannwhitneyu(a, b, alternative='two-sided')
    auc = U / (a.size * b.size)                       # P(a > b)
    ks, pks = ks_2samp(a, b)
    return dict(n_a=a.size, n_b=b.size, auc=auc, p_mwu=p, ks=ks, p_ks=pks,
                mean_a=a.mean(), mean_b=b.mean(), med_a=np.median(a), med_b=np.median(b))


def _parse_window(s):
    p = lambda t: np.inf if str(t).strip().lower().startswith('moho') else float(t)
    a, b = s.split(',')
    return (p(a), p(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xi', required=True, help='xi model ZT npz (field Xi), e.g. Fvs.v3c.ccpmoho.iter.4.ZT.npz')
    ap.add_argument('--moho', required=True, help='moho_ccp_refined.final.mat (Moho surface + stn_mask)')
    ap.add_argument('--units-npz', default='data/geology/yilgarn_units.npz',
                    help='Yilgarn rock-type polygons (greenstone/granite/...)')
    ap.add_argument('--bnd-npz', default='data/geology/yilgarn_boundaries.npz',
                    help='Yilgarn boundaries (for the craton outline)')
    ap.add_argument('--deposit-dir', default='data/deposits',
                    help='dir with gold_deposits.mat (default data/deposits, falls back to matlab/)')
    ap.add_argument('--gold-file', default=None, help='explicit gold .mat/.npz (else <deposit-dir>/gold_deposits.mat)')
    ap.add_argument('--cut-poly', default=None, help='keep only the SW side of this cut line (as the maps do)')
    ap.add_argument('--windows', nargs='+', default=['0,moho', '20,moho', 'moho,60', '0,10'],
                    help='depth windows "A,B"; A/B a number (km) or "moho". Default: 0,moho 20,moho moho,60 0,10')
    ap.add_argument('--field', default='Xi', help='npz field (default Xi)')
    ap.add_argument('--smooth', type=float, default=0.8, help='node smoothing (default 0.8, same as the maps)')
    ap.add_argument('--out', default='figures/xi-endowment/xi_endowment_stats.csv', help='CSV output path')
    args = ap.parse_args()

    _rel = lambda p: p if os.path.isabs(p) or os.path.exists(p) else os.path.join(_HERE, '..', p)

    # --- geology -------------------------------------------------------------------
    units = _rel(args.units_npz)
    gs_paths = _unit_paths(units, ['greenstone'])
    all_paths = _unit_paths(units, ['greenstone', 'granite', 'gneiss', 'sedimentary', 'other'])
    craton = _craton_path(_rel(args.bnd_npz))
    craton_src = 'craton outline (stitched)' if craton is not None else 'union of Yilgarn units'
    print(f"  greenstone polygons: {len(gs_paths)}   all-unit polygons: {len(all_paths)}")
    print(f"  craton mask: {craton_src}")

    glon, glat, gsz = _load_gold(_rel(args.deposit_dir), args.gold_file)
    print(f"  gold deposits: {glon.size}")

    # --- Moho grid (footprint) + optional SW cut -----------------------------------
    LON2, LAT2, ZC = _read_moho_mat(_rel(args.moho), 'z_moho', apply_mask=True)
    gl = LON2.ravel(); ga = LAT2.ravel(); gz = ZC.ravel()
    node_xy = np.column_stack([gl, ga])

    keep_node = np.isfinite(gz)
    keep_gold = np.ones(glon.size, bool)
    if args.cut_poly:
        kp = _keep_poly(_read_cut_poly(_rel(args.cut_poly)))
        keep_node &= kp.contains_points(node_xy)
        keep_gold &= kp.contains_points(np.column_stack([glon, glat]))
        print(f"  cut-poly: {keep_node.sum()} footprint nodes, {keep_gold.sum()} gold kept (SW side)")

    # --- classify grid nodes -------------------------------------------------------
    in_gs = _in_any(gs_paths, node_xy)
    in_yil = _in_any(all_paths, node_xy)
    if craton is not None:
        in_craton = craton.contains_points(node_xy)
    else:
        in_craton = in_yil
    foot = keep_node
    m_gs = foot & in_gs
    m_gran = foot & in_craton & ~in_gs            # craton, non-greenstone background
    m_craton = foot & in_craton
    m_margin = foot & ~in_craton                  # footprint outside craton

    print(f"  nodes -> footprint {foot.sum()}, craton {m_craton.sum()}, margin {m_margin.sum()}, "
          f"greenstone {m_gs.sum()}, granite-bkg {m_gran.sum()}")

    # --- gold -> nearest finite node per window (sample the SAME smoothed field) ----
    from scipy.spatial import cKDTree

    rows = []
    for wstr in args.windows:
        win = _parse_window(wstr)
        V = _avg_model_window(_rel(args.xi), args.field, LON2, LAT2, ZC, win, smooth=args.smooth)
        v = V.ravel()
        finite = foot & np.isfinite(v)
        if finite.sum() < 10:
            print(f"  [{wstr}] too few finite nodes ({finite.sum()}); skipped")
            continue
        tree = cKDTree(node_xy[finite])
        vfin = v[finite]
        # gold xi = nearest finite footprint node
        dd, ii = tree.query(np.column_stack([glon[keep_gold], glat[keep_gold]]))
        gold_xi = np.where(dd < 0.5, vfin[ii], np.nan)   # drop gold >0.5deg off any node

        pops = {
            'GOLD': gold_xi,
            'GREENSTONE': v[m_gs & np.isfinite(v)],
            'GRANITE': v[m_gran & np.isfinite(v)],
            'CRATON': v[m_craton & np.isfinite(v)],
            'MARGIN': v[m_margin & np.isfinite(v)],
        }
        comparisons = [
            ('GOLD', 'MARGIN'),       # is gold xi different from barren margins?
            ('GOLD', 'GRANITE'),      # is gold xi different from craton granite background?
            ('CRATON', 'MARGIN'),     # craton-vs-margin (the first-order claim)
            ('GREENSTONE', 'GRANITE'),# belt-vs-dome (the shallow claim)
        ]
        print(f"\n=== window {wstr}  ({_avg_window_desc(win)}) ===")
        print(f"  {'population':<11s} {'n':>5s} {'mean':>8s} {'median':>8s}")
        for name, vals in pops.items():
            vv = np.asarray(vals, float); vv = vv[np.isfinite(vv)]
            if vv.size:
                print(f"  {name:<11s} {vv.size:>5d} {vv.mean():>8.4f} {np.median(vv):>8.4f}")
        print(f"  {'A vs B':<20s} {'nA':>5s} {'nB':>5s} {'meanA':>8s} {'meanB':>8s} "
              f"{'AUC':>6s} {'p(MWU)':>9s} {'KS':>6s} {'p(KS)':>9s}  interpretation")
        for A, B in comparisons:
            r = _auc_mwu(pops[A], pops[B])
            tag = f'{A} vs {B}'
            higher = A if (np.isfinite(r['auc']) and r['auc'] >= 0.5) else B
            sep = ('none' if not np.isfinite(r['auc']) else
                   'strong' if abs(r['auc'] - 0.5) > 0.20 else
                   'moderate' if abs(r['auc'] - 0.5) > 0.10 else
                   'weak' if abs(r['auc'] - 0.5) > 0.05 else 'negligible')
            note = f'{higher} higher xi, {sep}'
            print(f"  {tag:<20s} {r['n_a']:>5d} {r['n_b']:>5d} {r['mean_a']:>8.4f} {r['mean_b']:>8.4f} "
                  f"{r['auc']:>6.3f} {r['p_mwu']:>9.2e} {r['ks']:>6.3f} {r['p_ks']:>9.2e}  {note}")
            rows.append(dict(window=wstr, comparison=tag, **r, higher=higher, separation=sep))

    # --- CSV -----------------------------------------------------------------------
    out = _rel(args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import csv
    if rows:
        cols = ['window', 'comparison', 'n_a', 'n_b', 'mean_a', 'mean_b', 'med_a', 'med_b',
                'auc', 'p_mwu', 'ks', 'p_ks', 'higher', 'separation']
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in cols})
        print(f"\n  wrote {out}  ({len(rows)} comparisons)")
    print("\n  AUC reading: 0.50 = no difference; 0.70/0.30 = one population clearly higher/lower xi.")


def _avg_window_desc(win):
    a, b = win
    if np.isinf(b):
        return f'{a:g} km -> Moho'
    if np.isinf(a):
        return f'Moho -> {b:g} km'
    return f'{a:g}-{b:g} km'


if __name__ == '__main__':
    main()
