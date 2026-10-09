#!/usr/bin/env python3
"""
domains.py - tectonic domains (Yilgarn, West Australian Craton, Perth Basin, ...) as sets
of GSWA 1:10M tectonic units, for regional 1-D averages and as dln references.

Domains are defined in domains.csv (beside this file): key, label, colour, units, where
units are 'name:<TECTNAME>' / 'parent:<parent>' entries. A point is inside a unit when it
lies inside an odd number of that unit's rings (holes are holes).

  from domains import Domains
  D = Domains()                                   # wa_tectonics.npz via config / $WA_TECTONICS_NPZ
  m = D.node_mask(d, 'yilgarn')                   # Fvs nodes in the Yilgarn with Ndat >= 35
  ref = D.mean_profile(d, d['Vsv'], 'yilgarn')    # mean at each depth over those nodes
"""
import csv, os
import numpy as np
from matplotlib.path import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS_CSV = os.path.join(_HERE, 'domains.csv')


def _norm(s):
    return str(s).replace('–', '-').replace('—', '-').strip().lower()


def load_domains(path=DOMAINS_CSV):
    """{key: dict(key, label, colour, units=[(field, value), ...])} in file order."""
    out = {}
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(l for l in f if l.strip() and not l.startswith('#')):
            units = []
            for u in r['units'].split(';'):
                if ':' in u:
                    fld, val = u.split(':', 1)
                    units.append((fld.strip().lower(), _norm(val)))
            out[r['key'].strip()] = dict(key=r['key'].strip(), label=r['label'].strip(),
                                         colour=r['colour'].strip(), units=units)
    return out


def _tect_npz(path=None):
    cands = [path, os.environ.get('WA_TECTONICS_NPZ')]
    try:
        import sys
        sys.path.insert(0, os.path.join(_HERE, '..'))
        import config
        cands.append(config.TECTONICS_NPZ)
    except Exception:
        pass
    cands.append(os.path.join(_HERE, '..', 'data', 'wa_tectonics.npz'))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError('wa_tectonics.npz not found (config.TECTONICS_NPZ / $WA_TECTONICS_NPZ)')


class Domains:
    def __init__(self, tect_npz=None, csv_path=DOMAINS_CSV):
        self.npz = _tect_npz(tect_npz)
        t = np.load(self.npz, allow_pickle=True)
        self.rings = [np.asarray(r, float)[:, :2] for r in t['rings']]
        self.names = np.array([_norm(x) for x in (t['names'] if 'names' in t.files else t['parents'])])
        self.parents = np.array([_norm(x) for x in t['parents']])
        self.defs = load_domains(csv_path)

    def keys(self):
        return list(self.defs)

    def __getitem__(self, key):
        if key not in self.defs:
            raise KeyError(f"domain '{key}' not in {DOMAINS_CSV} (have: {', '.join(self.defs)})")
        return self.defs[key]

    def unit_rings(self, key):
        """{unit name: [ring indices]} of the 10M units making up a domain."""
        dom = self[key]
        units = {}
        for i, (n, p) in enumerate(zip(self.names, self.parents)):
            for fld, val in dom['units']:
                if (fld == 'name' and n == val) or (fld == 'parent' and p == val):
                    units.setdefault(n, []).append(i)
                    break
        return units

    def contains(self, key, lon, lat):
        """Boolean: points inside the domain (inside any of its units)."""
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        pts = np.column_stack([lon, lat])
        inside = np.zeros(len(lon), bool)
        for unit, idx in self.unit_rings(key).items():
            count = np.zeros(len(lon), int)
            for i in idx:
                r = self.rings[i]
                if len(r) < 3:
                    continue
                bb = ((lon >= r[:, 0].min()) & (lon <= r[:, 0].max())
                      & (lat >= r[:, 1].min()) & (lat <= r[:, 1].max()))
                if bb.any():
                    count[bb] += Path(r).contains_points(pts[bb])
            inside |= (count % 2) == 1
        return inside

    def node_mask(self, d, key, min_ndat=35, footprint=None):
        """Fvs nodes inside the domain (and Ndat >= min_ndat, inside footprint if given)."""
        lon, lat = np.asarray(d['Lon'], float), np.asarray(d['Lat'], float)
        m = self.contains(key, lon, lat)
        if min_ndat and 'Ndat' in getattr(d, 'files', d):
            m &= np.asarray(d['Ndat'], float) >= min_ndat
        if footprint is not None:
            m &= footprint.contains(lon, lat)
        return m

    def mean_profile(self, d, arr, key, min_ndat=35, footprint=None):
        """Mean of arr[nodes, depth] over the domain's nodes at each depth."""
        m = self.node_mask(d, key, min_ndat, footprint)
        if not m.any():
            raise ValueError(f"no model nodes in domain '{key}'")
        return np.nanmean(np.asarray(arr, float)[m], axis=0)


def model_fields(d):
    """The physical fields a model carries: {'vsv': true Vsv, 'viso', 'xi', 'vpvs'} as
    [nodes, depth] arrays - xi models give Viso in 'Vsv' and true Vsv from xi; Vp/Vs and
    xi only when they vary."""
    varies = lambda k: k in d and np.nanstd(d[k]) > 1e-4
    F = {}
    if varies('Xi'):
        F['vsv'] = d['Vsv'] / np.sqrt((2.0 + d['Xi'] ** 2) / 3.0)
        F['viso'] = d['Vsv']
        F['xi'] = d['Xi']
    else:
        F['vsv'] = d['Vsv']
    if varies('Vpvs'):
        F['vpvs'] = d['Vpvs']
    return F
