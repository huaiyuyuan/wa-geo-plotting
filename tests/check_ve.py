#!/usr/bin/env python3
"""Measure the DRAWN vertical exaggeration of every depth panel plot_xsection.py
saves, and compare it with the VE in the figure title.

Usage: same arguments as plot_xsection.py, e.g.
  python tests/check_ve.py --fvs Fvs.iter.2.Z.npz --load-sections sections.txt --strips
"""
import os, re, sys
import numpy as np
import matplotlib.figure as mf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plotting'))
_orig = mf.Figure.savefig


def _spy(self, fname, *a, **k):
    if str(fname).endswith('.png'):
        title = self._suptitle.get_text() if self._suptitle else ''
        lab = re.search(r'VE ([\d.]+)x', title)
        W, H = self.get_figwidth(), self.get_figheight()
        for ax in self.axes:
            if ax.get_ylim()[0] <= ax.get_ylim()[1]:        # only depth-down panels
                continue
            p = ax.get_position()
            w, h = p.width * W, p.height * H
            xr, zr = abs(np.diff(ax.get_xlim())[0]), abs(np.diff(ax.get_ylim())[0])
            print(f"  {os.path.basename(fname):30s} panel {w:5.2f} x {h:4.2f} in  "
                  f"drawn VE {(xr / w) / (zr / h):5.2f}  title {lab.group(1) if lab else '-'}")
    return _orig(self, fname, *a, **k)


mf.Figure.savefig = _spy
import plot_xsection  # noqa: E402
plot_xsection.main()
