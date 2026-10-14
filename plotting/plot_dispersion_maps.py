#!/usr/bin/env python3
"""
plot_dispersion_maps.py - single-period publication maps of the v3b (band120) iter-2
surface-wave group / phase velocity maps (TransD.region.{group,phase}.npz), with the WA
Array coverage mask, tectonic domains (GSWA 1:10M units) and stations.

  source env_stage6.sh
  python3 plotting/plot_dispersion_maps.py --root $V3B --periods 5 10 20 30 40 \\
      --out $FIG/v3b/maps                       # all four sets, dV vs WAC, per-panel limits

Field (--field):
  dv   (default) dV (%) = 100 (v - vref) / vref, vref = mean of the reference domain
       (--ref-domain, default wac) over the masked cells at that period. RdBu, white centre.
  abs  velocity (km/s), Spectral (warm = slow), limits 2-98th percentile in the mask.
  std  posterior std (std2) inside the coverage mask, with the std = 0.80 contour.
Colour scale for dv (--clim), the same rule for every period:
  ref    (default) per panel, symmetric +-L, L = the --clim-pct (98th) percentile of |dV|
         inside the reference domain, rounded up to 0.5 %; arrowed colourbar. Same rule as
         noise_asdf common/plot_masked_periods.py --ref-shp ... --ref-name "West Australian Craton".
  fixed  +-P % for every map (--pct, default 4): amplitudes comparable across periods,
         short periods (basins) saturate.
Mask: coverage (footprint 'density' >= 10, holes filled, largest region) AND posterior
std <= 0.80 (finite maski; --std-max to recompute from std2) AND finite velocity.

Outputs <out>/<tag>.<Z|T>.<phase|group>.<P>s.<field>[-<ref>].<fp|wa>.png (+ .pdf with
--pdf), --sheet adds one page per set with all chosen periods, and <tag>.captions.txt
holds a caption line per map with the caveats of that period.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', 'basemap'))
import tomo2d as t2                                           # noqa: E402

DEFAULT_DOMAINS = ['wac', 'yilgarn', 'pilbara', 'capricorn', 'albany_fraser',
                   'perth', 'canning', 'officer', 'eucla']
DEFAULT_PERIODS = [3, 5, 8, 12, 20, 30, 40, 60]


def _cmap_rel(n):
    try:
        import plot_depth_slice as pds                       # the shared RdBu-with-white rule
        return pds._cmap('RdBu', n)
    except Exception:
        return matplotlib.colormaps['RdBu'].resampled(n)


def nice_up(x, step=0.5, floor=0.5):
    return max(floor, step * np.ceil(x / step - 1e-9))


class Overlay:
    """Domain outlines + labels + stations, prepared once."""

    def __init__(self, a, T):
        self.D, self.keys, self.dmask = None, [], {}
        if a.domains and a.domains != ['none']:
            from domains import Domains
            sys.path.insert(0, os.path.join(_HERE, '..', 'extract'))
            from export_domains import all_definitions
            self.D = Domains(a.tect_npz)
            self.D.defs = all_definitions(True)            # commented domains usable too
            for k in a.domains:
                if k not in self.D.defs:
                    print(f"  (domain '{k}' not in domains.csv - skipped)")
                    continue
                self.keys.append(k)
                self.dmask[k] = self.D.contains(k, T['LON'].ravel(), T['LAT'].ravel()).reshape(T['LON'].shape)
        if a.ref_domain not in ('mask', 'none') and a.ref_domain not in self.dmask:
            from domains import Domains
            D = self.D or Domains(a.tect_npz)
            if self.D is None:
                self.D = D
            self.dmask[a.ref_domain] = D.contains(a.ref_domain, T['LON'].ravel(),
                                                  T['LAT'].ravel()).reshape(T['LON'].shape)
        self.sta = t2.load_stations(a.stations) if a.show_stations != 'none' else None
        self.labels = a.labels
        self.show_all = a.show_stations == 'all'
        self.ref = a.ref_domain

    def draw(self, F, T, cov, fontscale=1.0):
        ax = F.ax
        for k in self.keys:
            rings = t2.domain_outline_rings(self.D, k)
            wac = k == 'wac'
            F.lines(rings, colors='k' if wac else '0.15', linewidths=1.25 if wac else 0.6,
                    zorder=5.5 if wac else 5.4)
            if self.labels == 'none' or wac:
                continue
            p = t2.domain_label_xy(self.D, k, (T['LON'], T['LAT'], self.dmask[k] & cov))
            if p is None or not F.inside(np.array(p[0]), np.array(p[1])):
                continue
            txt = t2.SHORT.get(k, self.D[k]['label']) if self.labels == 'short' else self.D[k]['label']
            x, y = F.xy(p[0], p[1])
            ax.text(x, y, txt, fontsize=6.5 * fontscale, rotation=p[2], ha='center', va='center',
                    style='italic', color='k', zorder=8, clip_on=True,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground='w')])
        if self.sta is not None and len(self.sta[1]):
            names, lon, lat, arr = self.sta
            ins = F.inside(lon, lat)
            x, y = F.xy(lon, lat)
            if self.show_all:
                o = ins & ~arr
                ax.scatter(x[o], y[o], marker='^', s=6 * fontscale, facecolors='none',
                           edgecolors='0.35', linewidths=0.35, zorder=7)
            o = ins & arr
            ax.scatter(x[o], y[o], marker='^', s=5 * fontscale, c='k', edgecolors='w',
                       linewidths=0.2, zorder=7.1)


def panel(ax, cax, T, ip, cov, fm, ov, a, fontscale=1.0):
    """One map. Returns (caption, info dict)."""
    per = T['per'][ip]
    m = fm[ip]
    v = T['v'][ip]
    F = t2.MapFrame(ax, a._extent, grat=a.grat)
    ref_key = a.ref_domain
    info = dict(period=per, ncell=int(m.sum()))
    if a.field == 'dv':
        rm = m & ov.dmask[ref_key] if ref_key in ov.dmask else m
        if rm.sum() < 5:
            print(f"   {per:g} s: only {rm.sum()} masked cells in '{ref_key}' - reference = whole mask")
            rm, ref_name = m, 'mask'
        else:
            ref_name = ref_key
        vref = float(np.mean(v[rm]))
        z = 100.0 * (v - vref) / vref
        if a.clim == 'fixed':
            L = a.pct
            how = f"fixed ±{L:g} %"
        else:
            L = nice_up(np.percentile(np.abs(z[rm]), a.clim_pct))
            how = f"±{L:g} % ({a.clim_pct:g}th pct of |dV| inside {_rname(ov, ref_name)})"
        cmap, norm = _cmap_rel(a.ncolors), Normalize(-L, L)
        clab = f"dV (%) relative to {_rname(ov, ref_name)} mean ({vref:.3f} km/s)"
        info.update(vref=vref, L=L, ref=ref_name, nref=int(rm.sum()))
        extend = 'both'
    elif a.field == 'abs':
        z = v
        lo, hi = (a.vlim if a.vlim else np.percentile(v[m], [2, 98]))
        cmap, norm = matplotlib.colormaps['Spectral'], Normalize(lo, hi)
        clab = f"{T['typ']} velocity (km/s)"
        how = f"{lo:.2f}-{hi:.2f} km/s"
        extend = 'both'
    else:                                                  # std
        z = T['std'][ip]
        m = cov & np.isfinite(z)
        hi = a.vlim[1] if a.vlim else max(1.2, float(np.percentile(z[m], 98)))
        cmap, norm = matplotlib.colormaps['viridis'], Normalize(0, hi)
        clab = 'posterior std (std2); line: 0.80'
        how = f"0-{hi:.2f}"
        extend = 'max'
    la, lo_, zi, mi = t2.upsample(T['lat'], T['lon'], z, m, a.smooth)
    pm = F.raster(la, lo_, zi, mi, cmap, norm)
    if a.field == 'std':
        LO, LA = np.meshgrid(lo_, la)
        X, Y = F.xy(LO, LA)
        ax.contour(X, Y, np.ma.masked_where(~mi, zi), levels=[0.8], colors='w', linewidths=0.8, zorder=3)
    if a.cov_line:
        F.outline(T['lat'], T['lon'], cov, colors='0.4', linewidths=0.6, linestyles='--')
    ov.draw(F, T, cov, fontscale)
    # titles + notes
    title = f"{t2.WAVE.get(T['comp'], T['comp'])} ({T['comp']}) {T['typ']} velocity, {per:g} s"
    ax.set_title(title, fontsize=10 * fontscale, loc='left', pad=4)
    cav = t2.caveats_for(T['comp'], T['typ'], per)
    if cav:
        ax.text(0.01, 0.01, '\n'.join(cav), transform=ax.transAxes, fontsize=6 * fontscale,
                color='#b03a2e', ha='left', va='bottom', zorder=9,
                bbox=dict(fc='w', ec='none', alpha=0.85, pad=1.5))
    cb = plt.colorbar(pm, cax=cax, orientation='horizontal', extend=extend)
    cb.ax.tick_params(labelsize=7 * fontscale)
    cb.set_label(clab, fontsize=7.5 * fontscale)
    info['how'] = how
    cap = (f"{title} ({a.tag}). {clab.split(' (')[0] if a.field == 'dv' else clab}"
           + (f" relative to the {_rname(ov, info.get('ref'))} mean, {info.get('vref', 0):.3f} km/s, "
              f"over {info.get('nref', 0)} cells" if a.field == 'dv' else '')
           + f"; colour scale {how}. Shown where the WA Array path density (10 s, array-to-array "
             f"paths) is >= 10 per cell (holes filled, largest region) and the posterior std "
             f"<= 0.80 ({info['ncell']} cells). Lines: GSWA 1:10M tectonic units "
             f"(thick: West Australian Craton); triangles: stations."
           + (' Caveat: ' + '; '.join(cav) + '.' if cav else ''))
    return cap, info


def _rname(ov, key):
    if key in (None, 'mask'):
        return 'masked-area'
    try:
        return ov.D[key]['label'] if ov.D is not None else key
    except KeyError:
        return key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.environ.get('V3B'),
                    help='v3b dir holding {Z,T}/{group,phase}_v_tomo/ (default $V3B)')
    ap.add_argument('--npz', nargs='+', default=None,
                    help='explicit TransD.region.*.npz files instead of --root (set read from the path)')
    ap.add_argument('--sets', nargs='+', default=['all'], help='Z.phase T.phase Z.group T.group | all')
    ap.add_argument('--periods', nargs='+', default=[str(p) for p in DEFAULT_PERIODS],
                    help='periods (s), nearest within 5 %%; absent periods skipped; "all"')
    ap.add_argument('--field', choices=['dv', 'abs', 'std'], default='dv')
    ap.add_argument('--ref-domain', default='wac', help="domain key, or 'mask' (whole masked area)")
    ap.add_argument('--clim', choices=['ref', 'wac', 'fixed'], default='ref')
    ap.add_argument('--clim-pct', type=float, default=98.0)
    ap.add_argument('--pct', type=float, default=4.0, help='+-P %% for --clim fixed')
    ap.add_argument('--vlim', nargs=2, type=float, default=None, help='limits for --field abs/std')
    ap.add_argument('--ncolors', type=int, default=16)
    ap.add_argument('--footprint', default=None, help='footprint npz (default config: $WA_FOOTPRINT_DIR)')
    ap.add_argument('--density-min', type=float, default=10.0)
    ap.add_argument('--std-max', type=float, default=None,
                    help='recompute the std mask from std2 (default: finite maski = std <= 0.80)')
    ap.add_argument('--extent', default='fp', help="fp (coverage mask +0.6 deg) | wa | lon0 lon1 lat0 lat1 as 'a,b,c,d'")
    ap.add_argument('--grat', type=float, default=None, help='graticule step (deg; default 2 fp, 5 wa)')
    ap.add_argument('--smooth', type=int, default=4, help='bilinear upsampling factor (1 = cells)')
    ap.add_argument('--domains', nargs='+', default=DEFAULT_DOMAINS, help="keys of domains.csv, or 'none'")
    ap.add_argument('--labels', choices=['short', 'full', 'none'], default='short')
    ap.add_argument('--tect-npz', default=None)
    ap.add_argument('--stations', default=None, help='stations file (default config.STATIONS)')
    ap.add_argument('--show-stations', choices=['array', 'all', 'none'], default='array',
                    help='array = W????, C[WESC]??, F??? (the footprint stations); all adds the rest hollow')
    ap.add_argument('--no-cov-line', dest='cov_line', action='store_false',
                    help='no dashed line at the coverage-mask edge')
    ap.add_argument('--sheet', action='store_true', help='also one page per set with all chosen periods')
    ap.add_argument('--no-single', action='store_true', help='only the --sheet pages')
    ap.add_argument('--tag', default='v3b')
    ap.add_argument('--pdf', action='store_true')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--out', default='figures/v3b/maps')
    a = ap.parse_args()
    if a.clim == 'wac':
        a.clim = 'ref'
    os.makedirs(a.out, exist_ok=True)

    if a.npz:
        sets = [t2.set_from_path(p) + (p,) for p in a.npz]
    elif a.root:
        sets = t2.find_sets(a.root, t2.parse_sets(a.sets))
    else:
        sys.exit('give --root (v3b dir) or --npz')
    if not sets:
        sys.exit('no input npz found')

    caps = []
    ov = None
    for comp, typ, path in sets:
        T = t2.load_tomo(path, comp, typ)
        cov, cdesc = t2.load_coverage(T['lat'], T['lon'], a.footprint, thr=a.density_min)
        fm = t2.figure_masks(T, cov, a.std_max)
        print(f"{comp} {typ}: {os.path.relpath(path)}  {len(T['per'])} periods "
              f"{T['per'].min():g}-{T['per'].max():g} s; coverage {cov.sum()} cells ({cdesc})")
        if ov is None:
            ov = Overlay(a, T)
        if a.extent == 'fp':
            a._extent, etag = t2.mask_extent(T['lat'], T['lon'], cov), 'fp'
        elif a.extent == 'wa':
            a._extent, etag = [112.0, 129.6, -35.8, -13.2], 'wa'
        else:
            a._extent, etag = [float(x) for x in a.extent.split(',')], 'box'
        ftag = a.field + (f"-{a.ref_domain}" + (f"-pm{a.pct:g}" if a.clim == 'fixed' else '')
                          if a.field == 'dv' else '')
        ips = t2.pick_periods(T['per'], a.periods)
        if not ips:
            print('   (none of the requested periods in this set)')
            continue
        # map geometry: width 5.9 in, height from the projected box
        F0 = t2.Albers()
        lo0, lo1, la0, la1 = a._extent
        bx, by = F0(np.r_[lo0, lo1, lo1, lo0, 0.5 * (lo0 + lo1)], np.r_[la0, la0, la1, la1, la0])
        asp = (by.max() - by.min()) / (bx.max() - bx.min())
        infos = []
        if not a.no_single:
            for ip in ips:
                W = 5.9
                H = W * asp + 1.45
                fig = plt.figure(figsize=(W + 0.7, H))
                ax = fig.add_axes([0.45 / (W + 0.7), 1.1 / H, W / (W + 0.7), W * asp / H])
                cax = fig.add_axes([0.25, 0.5 / H, 0.55, 0.13 / H])
                cap, info = panel(ax, cax, T, ip, cov, fm, ov, a)
                infos.append(info)
                stem = f"{a.tag}.{comp}.{typ}.{T['per'][ip]:g}s.{ftag}.{etag}"
                fn = os.path.join(a.out, stem + '.png')
                fig.savefig(fn, dpi=a.dpi)
                if a.pdf:
                    fig.savefig(fn[:-4] + '.pdf', dpi=a.dpi)
                plt.close(fig)
                caps.append(f"{stem}: {cap}")
                print(f"   {T['per'][ip]:5g} s  {info['ncell']:5d} cells  {info.get('how', '')}"
                      + (f"  vref {info['vref']:.3f}" if 'vref' in info else '') + f"  -> {fn}")
        if a.sheet:
            n = len(ips)
            nc = min(4, n)
            nr = int(np.ceil(n / nc))
            w = 3.6
            fig = plt.figure(figsize=(nc * (w + 0.35) + 0.3, nr * (w * asp + 1.0) + 0.5))
            Wf, Hf = fig.get_size_inches()
            for k, ip in enumerate(ips):
                r, c = divmod(k, nc)
                x0 = 0.35 + c * (w + 0.35)
                y0 = Hf - 0.5 - (r + 1) * (w * asp + 1.0) + 0.85
                ax = fig.add_axes([x0 / Wf, y0 / Hf, w / Wf, w * asp / Hf])
                cax = fig.add_axes([(x0 + 0.15 * w) / Wf, (y0 - 0.42) / Hf, 0.7 * w / Wf, 0.09 / Hf])
                panel(ax, cax, T, ip, cov, fm, ov, a, fontscale=0.75)
            fig.suptitle(f"{a.tag}: {t2.WAVE.get(comp, comp)} ({comp}) {typ} velocity"
                         + (f" - dV vs {_rname(ov, a.ref_domain)} mean, "
                            f"{'per-panel limits' if a.clim == 'ref' else f'±{a.pct:g} %'}" if a.field == 'dv' else ''),
                         fontsize=11, y=1 - 0.18 / Hf)
            fn = os.path.join(a.out, f"{a.tag}.{comp}.{typ}.sheet.{ftag}.{etag}.png")
            fig.savefig(fn, dpi=200)
            if a.pdf:
                fig.savefig(fn[:-4] + '.pdf', dpi=200)
            plt.close(fig)
            print(f"   sheet -> {fn}")
    if caps:
        cf = os.path.join(a.out, f"{a.tag}.captions.txt")
        with open(cf, 'w') as f:
            f.write(f"# captions for {a.tag} maps; general: {t2.DROPPED}\n")
            f.write('\n'.join(caps) + '\n')
        print(f"captions -> {cf}")


if __name__ == '__main__':
    main()
