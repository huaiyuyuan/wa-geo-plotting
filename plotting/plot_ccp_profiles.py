#!/usr/bin/env python3
"""
plot_ccp_profiles.py - CCP receiver-function sections along the picked prof transects.

Python port of the sampling in ccp_profile_extract.m + the 3-page landscape layout.
Reads the CCP stack directly (scipy; the volume is plain arrays, no MATLAB objects):

  P.mat      : P.xmod, P.ymod (node coords, km), P.ax/P.ay (azimuthal-equidistant
               projection centre lon/lat), P.ofile1 (path to the CMP file),
               optional P.slat/P.slon/P.stnm (stations).
  CMP .mat   : cmp [nx,ny,nz] amplitude volume, dzi (depth step km), sdev2 (optional).

Sampling matches ccp_profile_extract: densify each profile along great circles, project
(lat,lon)->(x,y) km azimuthal-equidistant about (ay,ax), then interpolate the cube with a
nearest pass (defines coverage) and a linear/cubic pass (what we look at), averaged.

Output: one landscape page per orientation (WE / NS / SENW), rows = profiles, each an
amplitude section distance (km) x depth (km). Vertical exaggeration is set the same way as
the Vsv sections (plot_xsection._section_scale), so the aspect ratio is correct and
consistent across profiles - no daspect letterboxing.

  source env_stage6.sh
  python3 plotting/plot_ccp_profiles.py \
      --pmat /mnt/data64/workspace/CCP/CCP.WA.Array.2027.phase.3/mat/WA.Array.phase.4.*.P.mat \
      --profiles $FIG/profiles/profiles.latlon.phases.1.2.3.polylines.npz \
      --d-max 80 --clim 0.08 --out-dir $FIG/profiles/ccp
# -> ccp.{we,ns,senw}.png (+ .pdf)
"""
import argparse, os, sys, glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))
import plot_xsection as px

GROUP_TITLE = {'we': 'West-East', 'ns': 'North-South', 'senw': 'SE-NW'}
GROUP_ENDS  = {'we': ('W', 'E'), 'ns': ('N', 'S'), 'senw': ('NW', 'SE')}
R_EARTH = 6371.0


def _ccp_cmap(neutral):
    """Continuous RdBu (blue +, red -) with a soft white centre: values within
    |x-0.5| < neutral/2 of the midpoint fade smoothly to white, so near-zero amplitude
    reads white instead of pale tint. neutral=0 returns plain RdBu."""
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    base = matplotlib.colormaps['RdBu']
    if not neutral or neutral <= 0:
        return base
    w = min(0.49, neutral / 2.0)
    xs = np.linspace(0, 1, 256)
    cols = base(xs)
    d = np.abs(xs - 0.5)
    f = np.clip(1.0 - d / w, 0, 1)                   # 1 at centre -> white, 0 at band edge
    cols[:, :3] = cols[:, :3] * (1 - f)[:, None] + f[:, None]
    return LinearSegmentedColormap.from_list('ccp_rdbu_soft', cols)


# ── geodesy matching MATLAB distance/azimuth/deg2km (earthRadius 6371 km) ──────
def _deg2km(d):
    return np.radians(d) * R_EARTH


def _gc_dist_deg(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def _azimuth(lat1, lon1, lat2, lon2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dlo = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dlo) * np.cos(la2)
    y = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
    return np.degrees(np.arctan2(x, y)) % 360.0


def _proj_xy(lat, lon, ax, ay):
    """Azimuthal-equidistant (x,y) km about (ay,ax), as proj_xy in ccp_profile_extract."""
    a = _azimuth(ay, ax, lat, lon)
    r = _deg2km(_gc_dist_deg(ay, ax, lat, lon))
    th = np.radians(90.0 - a)
    return r * np.cos(th), r * np.sin(th)


# ── load the CCP stack (P.mat + its CMP file) ─────────────────────────────────
def _mat_get(obj, name):
    return getattr(obj, name)


def load_ccp(pmat, ofile=None, amp_var='cmp'):
    from scipy.io import loadmat
    m = loadmat(pmat, squeeze_me=True, struct_as_record=False)
    if 'P' not in m:
        sys.exit(f"{pmat}: no variable 'P'")
    P = m['P']
    for f in ('xmod', 'ymod', 'ax', 'ay'):
        if not hasattr(P, f):
            sys.exit(f"{pmat}: P.{f} missing")
    xmod = np.atleast_1d(np.asarray(P.xmod, float)).ravel()
    ymod = np.atleast_1d(np.asarray(P.ymod, float)).ravel()
    ax, ay = float(P.ax), float(P.ay)

    # locate the CMP file: opt.ofile -> P.ofile1 -> basename beside the P.mat
    if ofile is None:
        ofile = getattr(P, 'ofile1', None)
    if ofile is None or not os.path.isfile(str(ofile)):
        cand = None
        if ofile is not None:
            cand = os.path.join(os.path.dirname(os.path.abspath(pmat)),
                                os.path.basename(str(ofile)))
        if cand and os.path.isfile(cand):
            print(f"  P.ofile1 not found; using {cand}")
            ofile = cand
        else:
            # last resort: a *.CMP.mat next to the P.mat
            alt = glob.glob(os.path.join(os.path.dirname(os.path.abspath(pmat)), '*CMP*.mat'))
            if alt:
                ofile = alt[0]
                print(f"  P.ofile1 unavailable; using {ofile}")
            else:
                sys.exit(f"cannot find the CMP stack file (P.ofile1={getattr(P,'ofile1','?')})")
    D = loadmat(str(ofile), squeeze_me=True, struct_as_record=False)
    if amp_var not in D:
        sys.exit(f"{ofile}: no '{amp_var}' variable (have {[k for k in D if not k.startswith('__')]})")
    cmp = np.asarray(D[amp_var], float).squeeze()
    if 'dzi' in D and np.size(D['dzi']):
        dzi = float(np.atleast_1d(D['dzi']).ravel()[0])
    elif hasattr(P, 'dzi'):
        dzi = float(np.atleast_1d(P.dzi).ravel()[0])
    else:
        sys.exit("depth increment dzi unavailable in CMP file or P")
    nx, ny = len(xmod), len(ymod)
    if cmp.shape[:2] != (nx, ny):
        sys.exit(f"cmp is {cmp.shape} but P.xmod/P.ymod imply ({nx},{ny},nz) - geometry disagrees")
    nz = cmp.shape[2]
    zg = np.arange(nz) * dzi
    sdev = None
    if 'sdev2' in D and np.size(D['sdev2']):
        s = np.asarray(D['sdev2'], float).squeeze()
        if s.shape == cmp.shape:
            sdev = s

    sta = None
    if hasattr(P, 'slat') and hasattr(P, 'slon') and np.size(P.slat):
        slat = np.atleast_1d(np.asarray(P.slat, float)).ravel()
        slon = np.atleast_1d(np.asarray(P.slon, float)).ravel()
        nm = getattr(P, 'stnm', None)
        names = ([str(x) for x in np.atleast_1d(nm).ravel()]
                 if nm is not None and np.size(nm) == len(slat) else [''] * len(slat))
        uniq = {}
        for la, lo, nmn in zip(slat, slon, names):
            uniq.setdefault((round(la, 4), round(lo, 4)), nmn)
        slat = np.array([k[0] for k in uniq]); slon = np.array([k[1] for k in uniq])
        sta = (slon, slat)

    print(f"  CCP volume {cmp.shape} from {os.path.basename(str(ofile))}; "
          f"depth 0-{zg[-1]:.0f} km (dzi {dzi:g}); centre ({ay:.2f},{ax:.2f})")
    return dict(xmod=xmod, ymod=ymod, ax=ax, ay=ay, zg=zg, cmp=cmp, sdev=sdev, sta=sta)


def sample_ccp(ccp, plon, plat, method='linear'):
    """Amplitude [nq, nz] along a profile, matching ccp_profile_extract: nearest pass for
    coverage, method pass for the picture, averaged; coverage mask reapplied."""
    from scipy.interpolate import RegularGridInterpolator
    xmod, ymod, zg, cmp = ccp['xmod'], ccp['ymod'], ccp['zg'], ccp['cmp']
    xq, yq = _proj_xy(np.asarray(plat), np.asarray(plon), ccp['ax'], ccp['ay'])
    nq, nz = len(xq), len(zg)
    XQ = np.repeat(xq[:, None], nz, axis=1).ravel()
    YQ = np.repeat(yq[:, None], nz, axis=1).ravel()
    ZQ = np.repeat(zg[None, :], nq, axis=0).ravel()
    pts = np.column_stack([XQ, YQ, ZQ])
    gn = RegularGridInterpolator((xmod, ymod, zg), cmp, method='nearest',
                                 bounds_error=False, fill_value=np.nan)
    gs = RegularGridInterpolator((xmod, ymod, zg), cmp, method=method,
                                 bounds_error=False, fill_value=np.nan)
    amp_n = gn(pts).reshape(nq, nz)
    amp_s = gs(pts).reshape(nq, nz)
    nanmask = ~np.isfinite(amp_n)
    with np.errstate(invalid='ignore'):          # all-NaN depth columns outside coverage
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            amp = np.nanmean(np.dstack([amp_n, amp_s]), axis=2)
    amp[nanmask] = np.nan
    return amp


# ── profiles (polylines npz) ──────────────────────────────────────────────────
def load_profiles(path, straight=False):
    if path.endswith('.npz'):
        d = np.load(path, allow_pickle=True)
        out = []
        for i in range(len(d['short'])):
            lo = np.asarray(d['lon'][i], float); la = np.asarray(d['lat'][i], float)
            if straight:
                lo = np.array([lo[0], lo[-1]]); la = np.array([la[0], la[-1]])
            out.append(dict(short=str(d['short'][i]), group=str(d['group'][i]),
                            colour=np.asarray(d['colour'][i], float), lon=lo, lat=la,
                            order=int(d['order'][i])))
        return out
    secs = px._load_sections(path)
    out = []
    for la1, lo1, la2, lo2, lab in secs:
        u = lab.upper()
        g = ('senw' if u.startswith('SENW') or u == 'WEST'
             else 'we' if u.startswith('WE') else 'ns')
        out.append(dict(short=lab, group=g, colour=np.array([.2, .2, .2]),
                        lon=np.array([lo1, lo2]), lat=np.array([la1, la2]), order=len(out)))
    return out


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


def plot_group(group, profs, ccp, out_dir, args, moho_interp=None):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    profs = sorted(profs, key=lambda p: p['order'])
    n = len(profs)
    zbot = args.d_max
    # +amp blue, -amp red (CCP convention). Smooth red-white-blue with a soft white centre
    # (|amp| < ~neutral*clim fades to white) so weak near-zero speckle reads clean, like the
    # Vsv maps; --neutral 0 = fully saturated, --ncolors N discretises into N steps.
    cmap = _ccp_cmap(args.neutral)
    if args.ncolors and args.ncolors > 0:
        cmap = cmap.resampled(args.ncolors)
    vmn, vmx = -args.clim, args.clim

    # sample all profiles
    for p in profs:
        plat, plon, dist = polyline_path(p['lon'], p['lat'], args.ds)
        p['_plat'], p['_plon'], p['_dist'] = plat, plon, dist
        amp = sample_ccp(ccp, plon, plat, method=args.method)
        zkeep = ccp['zg'] <= args.d_max
        p['_amp'] = amp[:, zkeep]; p['_z'] = ccp['zg'][zkeep]

    maxd = max(p['_dist'][-1] for p in profs)
    col_w = args.col_w
    km_per_in, row_h_in = px._section_scale(maxd, zbot, args.ve, col_w)

    strip_h = px._strip_height_in(px._GEO)
    gap_in = args.gap + strip_h
    top_in = 1.0 + strip_h
    left_in = 1.0
    fig_h = n * row_h_in + (n - 1) * gap_in + top_in + 0.9
    fig_w = left_in + col_w + 1.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    endL, endR = GROUP_ENDS[group]
    fig.suptitle(f'{GROUP_TITLE[group]} profiles — CCP RF amplitude   '
                 f'[{n} lines, depth 0-{zbot:.0f} km, col width {maxd:.0f} km, VE {args.ve:.0f}x, '
                 f'clim ±{args.clim:g}]', fontsize=11, fontweight='bold')
    rh = row_h_in / fig_h
    geo_res = []
    im0 = None
    for i, p in enumerate(profs):
        dist, z, amp = p['_dist'], p['_z'], p['_amp']
        w_in = dist[-1] / km_per_in
        y0 = 1 - (top_in / fig_h) - (i + 1) * rh - i * (gap_in / fig_h)
        ax = fig.add_axes([left_in / fig_w, y0, w_in / fig_w, rh])
        D, Z = np.meshgrid(dist, z, indexing='ij')
        im0 = ax.pcolormesh(D, Z, amp, cmap=cmap, vmin=vmn, vmax=vmx, shading='auto')
        ax.set_ylim(zbot, 0); ax.set_xlim(0, dist[-1])
        ax.set_ylabel('Depth', fontsize=7); ax.tick_params(labelsize=6)
        ax.set_xticks(np.arange(0, dist[-1] + 1, 200))
        ax.text(0.004, 0.04, f"{p['short']} [{dist[-1]:.0f} km]", transform=ax.transAxes,
                fontsize=8, fontweight='bold', ha='left', va='bottom', zorder=20, color='white',
                bbox=dict(fc=tuple(np.clip(p['colour'], 0, 1)), ec='none', alpha=0.95, pad=1.8))
        if np.isfinite(amp).sum() == 0:
            ax.text(0.5, 0.5, 'no coverage', transform=ax.transAxes, ha='center', va='center',
                    fontsize=7, color='0.4', style='italic')
        if moho_interp is not None:          # AusMoho 2023: thick solid black line
            mp = np.array([float(np.ravel(moho_interp(a, o))[0])
                           for a, o in zip(p['_plat'], p['_plon'])])
            ax.plot(dist, mp, '-', color='k', lw=2.2, zorder=10, solid_capstyle='round')
        if i == n - 1:
            ax.set_xlabel('Distance (km)', fontsize=8)
            ax.text(-0.005, -0.02, endL, transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, fontweight='bold', color='0.3')
            ax.text(1.005, -0.02, endR, transform=ax.transAxes, ha='left', va='top',
                    fontsize=8, fontweight='bold', color='0.3')
        if px._GEO is not None:
            geo_res.append(px._add_geology(ax, [ax], p['_plon'], p['_plat'], dist, names=None))

    # one shared colourbar on the right
    cax = fig.add_axes([(left_in + col_w + 0.25) / fig_w, 0.25, 0.14 / fig_w, 0.5])
    cb = fig.colorbar(im0, cax=cax, extend='both'); cb.ax.tick_params(labelsize=6)
    cb.set_label('RF amplitude', fontsize=8)
    if geo_res and any(g for g in geo_res):
        try:
            px._geology_legend(fig, geo_res, ax, below_in=0.55)
        except Exception:
            pass
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'ccp.{args.tag + "." if args.tag else ""}{group}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  {group}: {out} (+ .pdf)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pmat', required=True, help='CCP P.mat (holds the grid + path to the CMP file)')
    ap.add_argument('--cmp', default=None, help='override the CMP stack file (else from P.ofile1)')
    ap.add_argument('--amp-var', default='cmp', help="volume variable in the CMP file (default cmp)")
    ap.add_argument('--profiles', required=True, help='polylines npz (or sections.txt)')
    ap.add_argument('--straight', action='store_true')
    ap.add_argument('--groups', nargs='+', default=['we', 'ns', 'senw'],
                    choices=['we', 'ns', 'senw'])
    ap.add_argument('--d-max', type=float, default=80.0, dest='d_max')
    ap.add_argument('--clim', type=float, default=0.08, help='symmetric colour limit ±clim')
    ap.add_argument('--ds', type=float, default=0.03, help='profile sampling step (deg)')
    ap.add_argument('--ve', type=float, default=px.DEFAULT_VE, help='vertical exaggeration')
    ap.add_argument('--col-w', type=float, default=7.0, dest='col_w')
    ap.add_argument('--gap', type=float, default=0.55)
    ap.add_argument('--ncolors', type=int, default=16,
                    help='N discrete colour steps (granulated, like the Vsv maps; default 16); '
                         '0 = smooth continuous')
    ap.add_argument('--neutral', type=float, default=0.25,
                    help='soft white centre: fraction of the ±clim range near zero that fades '
                         'to white (0 = fully saturated RdBu). Default 0.25.')
    ap.add_argument('--method', default='linear', choices=['linear', 'nearest', 'cubic'],
                    help='smooth-pass interpolant (cubic needs a uniform grid)')
    ap.add_argument('--moho', default='auto',
                    help='AusMoho 2023 Moho file (lat lon depth; drawn as a thick black line). '
                         '"auto" (default) uses config.MOHO; "none" disables it.')
    ap.add_argument('--strips', nargs='?', const='domain,boundaries,names',
                    default='domain,boundaries,names')
    ap.add_argument('--no-strips', action='store_true')
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--bnd-npz', default=None)
    ap.add_argument('--litho-npz', default=None)
    ap.add_argument('--out-dir', default='.', dest='out_dir')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    px._GEO = None
    if not args.no_strips and args.strips:
        try:
            px._GEO = px._load_geology(args.strips, args.litho_npz, args.tect_npz, args.bnd_npz)
        except Exception as e:
            print(f"  strips unavailable ({e}); drawing without them")

    moho_interp = None
    moho_path = args.moho
    if moho_path == 'auto':
        try:
            import config
            moho_path = getattr(config, 'MOHO', None)
        except Exception:
            moho_path = None
    elif moho_path == 'none':
        moho_path = None
    if moho_path:
        try:
            import config
            skip = getattr(config, 'MOHO_SKIP_HEADER', 11)
        except Exception:
            skip = 11
        try:
            from scipy.interpolate import RectBivariateSpline
            md = np.loadtxt(moho_path, skiprows=skip)
            mla = np.unique(md[:, 0]); mlo = np.unique(md[:, 1])
            moho_interp = RectBivariateSpline(mla, mlo, md[:, 2].reshape(len(mla), len(mlo)),
                                              kx=1, ky=1)
            print(f"  Moho overlay: {moho_path}")
        except Exception as e:
            print(f"  Moho load failed: {e}")

    print(f"Loading CCP stack from {args.pmat} ...")
    ccp = load_ccp(args.pmat, ofile=args.cmp, amp_var=args.amp_var)
    profs = load_profiles(args.profiles, straight=args.straight)
    print(f"Profiles: {len(profs)} in groups {sorted({p['group'] for p in profs})}")

    made = []
    for g in args.groups:
        gp = [p for p in profs if p['group'] == g]
        if not gp:
            print(f"  {g}: no profiles - skipped"); continue
        made.append(plot_group(g, gp, ccp, args.out_dir, args, moho_interp=moho_interp))
    if not made:
        sys.exit("No pages made.")
    print(f"Done: {len(made)} page(s).")


if __name__ == '__main__':
    main()
