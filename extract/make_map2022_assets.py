#!/usr/bin/env python3
"""
make_map2022_assets.py - vector legend + major-unit labels from the GSWA 2022 map PDF.

Source: Geological Survey of Western Australia 2022, 1:10 000 000 simplified tectonic map
of Western Australia (WA_TectonicsMap_10M_A4_Mar2022.pdf), CC BY 4.0.

Writes (into basemap/ by default):
  tectonic_map_2022_legend.json  rock-type x age chart as vector shapes + text (PDF points)
  tectonic_map_2022_labels.csv   unit labels at GSWA's own positions, as lon/lat + angle

Positions: the map is Albers equal-area, central meridian 121E, standard parallels
17.5S / 31.5S (fitted to the printed 5-degree graticule: 0.1 pt rms). Labels are taken
back to lon/lat through that fit. Colours: CMYK converted as poppler renders them (the
same conversion the colour table was sampled with).

  python3 extract/make_map2022_assets.py --pdf WA_TectonicsMap_10M_A4_Mar2022.pdf
Needs pdfplumber (one-off; the outputs are committed).
"""
import argparse, csv, json, os
import numpy as np

LEGEND_BOX = (40.6, 40.6, 243.0, 216.5)     # x0, top, x1, bottom (PDF pt): chart only
GRAT = (0.05, 0.0, 0.0, 0.45)               # graticule stroke colour (CMYK)
LON0, SP1, SP2 = 121.0, -17.5, -31.5


# --- colour ---------------------------------------------------------------------------
_T = [((0,0,0,0),(1,1,1)), ((0,0,0,1),(.1373,.1216,.1255)), ((0,0,1,0),(1,.9490,0)),
      ((0,0,1,1),(.1098,.1020,0)), ((0,1,0,0),(.9255,0,.5490)), ((0,1,0,1),(.1412,0,0)),
      ((0,1,1,0),(.9294,.1098,.1412)), ((0,1,1,1),(.1333,0,0)), ((1,0,0,0),(0,.6784,.9373)),
      ((1,0,0,1),(0,.0588,.1412)), ((1,0,1,0),(0,.6510,.3137)), ((1,0,1,1),(0,.0745,0)),
      ((1,1,0,0),(.1804,.1922,.5725)), ((1,1,0,1),(0,0,.0078)), ((1,1,1,0),(.2118,.2119,.2235)),
      ((1,1,1,1),(0,0,0))]


def to_hex(col):
    """PDF colour (gray / RGB / CMYK tuple) -> '#rrggbb', CMYK as poppler converts it."""
    if col is None:
        return None
    col = tuple(col) if isinstance(col, (list, tuple)) else (col,)
    if len(col) == 1:
        rgb = col * 3
    elif len(col) == 3:
        rgb = col
    else:
        c, m, y, k = col
        v = [1 - c, 1 - m, 1 - y, 1 - k]
        rgb = [0.0, 0.0, 0.0]
        for bits, w in _T:
            x = np.prod([(1 - v[i]) if b else v[i] for i, b in enumerate(bits)])
            rgb = [r + wi * x for r, wi in zip(rgb, w)]
    return '#' + ''.join(f'{int(round(255 * min(1, max(0, t)))):02x}' for t in rgb)


# --- projection -----------------------------------------------------------------------
def _albers_consts():
    p1, p2 = np.radians(SP1), np.radians(SP2)
    n = (np.sin(p1) + np.sin(p2)) / 2
    C = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    return n, C, np.sqrt(C) / n


def albers(lon, lat):
    n, C, rho0 = _albers_consts()
    rho = np.sqrt(C - 2 * n * np.sin(np.radians(lat))) / n
    th = n * np.radians(np.asarray(lon) - LON0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)


def albers_inv(X, Y):
    n, C, rho0 = _albers_consts()
    s = np.sign(n)
    rho = s * np.hypot(X, rho0 - Y)
    th = np.arctan2(s * X, s * (rho0 - Y))
    lat = np.degrees(np.arcsin((C - (rho * n) ** 2) / (2 * n)))
    return LON0 + np.degrees(th / n), lat


def fit_page(page):
    """Affine page<->Albers from the printed graticule (meridians 115..130, parallels 15..35S)."""
    mer, par = [], []
    for c in page.curves + page.lines:
        if tuple(c.get('stroking_color') or ()) != GRAT or len(c['pts']) < 15:
            continue
        pts = np.array(c['pts'], float)
        (mer if np.ptp(pts[:, 1]) > np.ptp(pts[:, 0]) else par).append(pts)
    mer.sort(key=lambda m: m[0, 0]); par.sort(key=lambda q: q[:, 1].mean())
    P, Q = [], []
    for m, lo in zip(mer, [115, 120, 125, 130]):
        a, b = m[0], m[-1]
        nrm = np.array([-(b - a)[1], (b - a)[0]]) / np.hypot(*(b - a))
        for q, la in zip(par, [-15, -20, -25, -30, -35]):
            s = (q - a) @ nrm
            i = np.where(np.sign(s[:-1]) != np.sign(s[1:]))[0]
            if len(i):
                i = i[0]; t = s[i] / (s[i] - s[i + 1])
                P.append(q[i] + t * (q[i + 1] - q[i])); Q.append((lo, la))
    P, Q = np.array(P), np.array(Q)
    X, Y = albers(Q[:, 0], Q[:, 1])
    A = np.c_[X, Y, np.ones_like(X)]
    M = np.linalg.lstsq(A, P, rcond=None)[0]               # [X Y 1] @ M = page (x, top)
    r = np.hypot(*(A @ M - P).T)
    print(f"graticule fit: {len(P)} nodes, rms {np.sqrt((r ** 2).mean()):.2f} pt")
    Minv = np.linalg.inv(np.r_[M.T, [[0, 0, 1]]])          # page -> X, Y
    def to_lonlat(x, top):
        X, Y, _ = Minv @ np.array([x, top, 1.0])
        return albers_inv(X, Y)
    return to_lonlat


# --- legend ---------------------------------------------------------------------------
def legend(page):
    x0, t0, x1, t1 = LEGEND_BOX
    inside = lambda o: o['x0'] >= x0 - 0.1 and o['x1'] <= x1 + 0.1 and \
        o['top'] >= t0 - 0.1 and o['bottom'] <= t1 + 0.1
    shapes = []
    objs = [o for o in page.rects + page.curves + page.lines if inside(o)]
    objs.sort(key=lambda o: (not o.get('fill'), o['object_type'] == 'line'))   # fills first
    for o in objs:
        if tuple(o.get('stroking_color') or ()) == GRAT and not o.get('fill'):
            continue                                        # map graticule behind the box
        fill = to_hex(o['non_stroking_color']) if o.get('fill') else None
        if fill == '#ffffff' and (o['x1'] - o['x0']) > 0.9 * (x1 - x0):
            continue                                        # the white backing box
        stroke = to_hex(o['stroking_color']) if o.get('stroke') else None
        shapes.append(dict(pts=[[round(x - x0, 3), round(t1 - y, 3)] for x, y in o['pts']],
                           fill=fill, stroke=stroke, lw=round(o.get('linewidth') or 0, 3),
                           closed=bool(o.get('fill')) or o['object_type'] == 'rect'))
    chars = [c for c in page.chars if inside(c)]
    texts = []
    for c in [c for c in chars if c['upright']]:            # horizontal: via words below
        pass
    words = page.crop(LEGEND_BOX).extract_words(extra_attrs=['upright', 'size'],
                                                 keep_blank_chars=True)
    for w in words:
        if w['upright']:
            texts.append(dict(s=w['text'], x0=round(w['x0'] - x0, 2), x1=round(w['x1'] - x0, 2),
                              y=round(t1 - w['bottom'], 2), size=round(w['size'], 2), rot=0))
    cols = {}
    for c in chars:                                         # rotated (reads upward): by column
        if not c['upright']:
            cols.setdefault(round(c['x0'], 1), []).append(c)
    for x, cs in cols.items():
        cs.sort(key=lambda c: -c['bottom'])
        s = ''.join(c['text'] for c in cs)
        texts.append(dict(s=s.strip(), x0=round(cs[0]['x0'] - x0, 2), x1=round(cs[0]['x1'] - x0, 2),
                          y0=round(t1 - cs[0]['bottom'], 2), y1=round(t1 - cs[-1]['top'], 2),
                          size=round(abs(cs[0]['matrix'][2]), 2), rot=90))
    return dict(width=round(x1 - x0, 2), height=round(t1 - t0, 2), shapes=shapes, texts=texts,
                source='GSWA 2022 1:10M simplified tectonic map of WA, legend (CC BY 4.0)')


# --- labels ---------------------------------------------------------------------------
STYLE = {'#ed1c24': 'craton'}    # filled in from colours below


def _classify(hexcol):
    r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
    if r > 180 and g < 80:
        return 'craton'                                     # red: cratons, terranes, inliers
    if r < 40 and g < 40 and b < 40:
        return 'unit'                                       # black: basins, provinces, zones
    return 'orogen'                                         # grey: orogens (curved, spaced)


def labels(page, to_lonlat):
    x0, t0, x1, t1 = LEGEND_BOX
    W, H = page.width, page.height
    chars = [c for c in page.chars
             if not (x0 - 2 < c['x0'] < x1 + 2 and t0 - 2 < c['top'] < 260)
             and 22 < c['x0'] < W - 22 and 22 < c['top'] < H - 22 and c['text'].strip()
             and 'Roboto' not in c['fontname'] and 'ArialMT' not in c['fontname']]
    for c in chars:
        c['hex'] = to_hex(c['non_stroking_color'])
        c['cls'] = _classify(c['hex'])
        c['cx'] = (c['x0'] + c['x1']) / 2; c['cy'] = (c['top'] + c['bottom']) / 2
    out = []
    # straight labels (red / black): words -> lines -> blocks
    for cls in ('craton', 'unit'):
        cs = [c for c in chars if c['cls'] == cls]
        lines = []
        for c in sorted(cs, key=lambda c: (round(c['top'], 0), c['x0'])):
            for L in lines:
                if abs(L['top'] - c['top']) < 0.6 and abs(L['size'] - c['size']) < 0.3 \
                        and -0.5 < c['x0'] - L['x1'] < 2.2 * c['size'] * 0.45:
                    sp = ' ' if c['x0'] - L['x1'] > 0.18 * c['size'] else ''
                    L['s'] += sp + c['text']; L['x1'] = c['x1']; L['bottom'] = max(L['bottom'], c['bottom'])
                    L['bold'] |= 'Bold' in c['fontname']
                    break
            else:
                lines.append(dict(s=c['text'], x0=c['x0'], x1=c['x1'], top=c['top'], bottom=c['bottom'],
                                  size=c['size'], bold='Bold' in c['fontname']))
        blocks = []
        for L in sorted(lines, key=lambda L: L['top']):
            cx = (L['x0'] + L['x1']) / 2
            for B in blocks:
                Lb = B[-1]
                if abs(Lb['size'] - L['size']) < 0.3 and 0 <= L['top'] - Lb['top'] < 1.45 * L['size'] \
                        and abs((Lb['x0'] + Lb['x1']) / 2 - cx) < 0.6 * max(Lb['x1'] - Lb['x0'], L['x1'] - L['x0']):
                    B.append(L); break
            else:
                blocks.append([L])
        for B in blocks:
            bx = (min(L['x0'] for L in B) + max(L['x1'] for L in B)) / 2
            by = (B[0]['top'] + B[-1]['bottom']) / 2
            lon, lat = to_lonlat(bx, by)
            out.append(dict(text='\\n'.join(L['s'] for L in B), lon=lon, lat=lat, angle=0.0,
                            cls=cls, size=B[0]['size'], bold=int(any(L['bold'] for L in B)),
                            spacing=0.0))
    # curved grey labels: chain characters into strings, then strings into stacked lines
    cs = [c for c in chars if c['cls'] == 'orogen']
    used, chains = set(), []
    for i, c in enumerate(cs):
        if i in used:
            continue
        chain, used_now = [i], {i}
        grow = True
        while grow:                               # greedy nearest-neighbour both ends
            grow = False
            for end in (0, -1):
                e = cs[chain[end]]
                best, bd = None, 1e9
                for j, d in enumerate(cs):
                    if j in used or j in used_now or abs(d['size'] - e['size']) > 1.8:
                        continue
                    dd = np.hypot(d['cx'] - e['cx'], d['cy'] - e['cy'])
                    if dd < bd:
                        best, bd = j, dd
                if best is not None and bd < 1.25 * max(e['size'], cs[best]['size']):
                    if end == 0: chain.insert(0, best)
                    else: chain.append(best)
                    used_now.add(best); grow = True
        used |= used_now
        chains.append([cs[k] for k in chain])
    for ch in chains:
        pts = np.array([[c['cx'], c['cy']] for c in ch])
        if len(ch) > 1:                           # reading direction: left to right
            d = pts[-1] - pts[0]
            if d[0] < -0.3 * abs(d[1]) or (abs(d[0]) < 0.3 * abs(d[1]) and d[1] < 0):
                ch, pts = ch[::-1], pts[::-1]
        d = pts[-1] - pts[0] if len(ch) > 1 else np.array([1.0, 0.0])
        ang = float(np.degrees(np.arctan2(-d[1], d[0])))
        gaps = np.hypot(*np.diff(pts, axis=0).T) if len(ch) > 1 else np.array([0.0])
        med = np.median(gaps) if len(gaps) else 0
        s = ch[0]['text']
        for g, c in zip(gaps, ch[1:]):
            s += (' ' if g > 1.9 * med else '') + c['text']
        cx, cy = pts.mean(0)
        lon, lat = to_lonlat(cx, cy)
        out.append(dict(text=s, lon=lon, lat=lat, angle=round(ang, 1), cls='orogen',
                        size=float(np.median([c['size'] for c in ch])), bold=0,
                        spacing=round(float(med) / float(np.median([c['size'] for c in ch])), 2)))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pdf', required=True)
    ap.add_argument('--out-dir', default=os.path.join(here, '..', 'basemap'))
    a = ap.parse_args()
    import pdfplumber
    page = pdfplumber.open(a.pdf).pages[0]
    to_lonlat = fit_page(page)
    leg = legend(page)
    p = os.path.join(a.out_dir, 'tectonic_map_2022_legend.json')
    with open(p, 'w') as f:
        json.dump(leg, f, indent=0)
    print(f"{p}: {len(leg['shapes'])} shapes, {len(leg['texts'])} texts")
    labs = labels(page, to_lonlat)
    p = os.path.join(a.out_dir, 'tectonic_map_2022_labels.raw.csv')
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['text', 'lon', 'lat', 'angle', 'class', 'size_pt', 'bold', 'spacing'])
        for L in sorted(labs, key=lambda L: (L['cls'], -L['lat'])):
            w.writerow([L['text'], f"{L['lon']:.3f}", f"{L['lat']:.3f}", L['angle'], L['cls'],
                        f"{L['size']:.1f}", L['bold'], L['spacing']])
    print(f"{p}: {len(labs)} labels (raw; curate into tectonic_map_2022_labels.csv)")


if __name__ == '__main__':
    main()
