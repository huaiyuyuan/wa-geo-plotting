"""Synthetic demo of lithology + crustal-boundary strips on a cross-section.

Fake geology: granite background, three NNW-trending greenstone belts,
two terrane boundaries. Fake Vs: fast upper-crustal bands under the belts.
With real data, swap make_fake_* for PolygonIndex.from_npz / BoundarySet.from_npz.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
import xsection_strips as xs  # noqa: E402

GRANITE, GNEISS, GREEN, SED = "#f4b6c2", "#d9a5c9", "#4c9a52", "#e8d9a8"


def belt(lon_c, width, lat0=-32.5, lat1=-27.5, skew=-0.25, wiggle=0.12, seed=0):
    rng = np.random.default_rng(seed)
    lat = np.linspace(lat0, lat1, 25)
    c = lon_c + skew * (lat - lat.mean()) + wiggle * np.sin(3 * lat) + rng.normal(0, .03, 25)
    w = width * (0.7 + 0.3 * np.cos(2 * (lat - lat.mean())))
    return np.vstack([np.column_stack([c - w / 2, lat]),
                      np.column_stack([c + w / 2, lat])[::-1],
                      [[c[0] - w[0] / 2, lat[0]]]])


def make_fake_litho():
    rings = [np.array([[115.5, -33], [124.5, -33], [124.5, -27], [115.5, -27], [115.5, -33]]),
             np.array([[115.5, -33], [117.2, -33], [117.0, -27], [115.5, -27], [115.5, -33]]),
             belt(119.0, 0.45, seed=1), belt(120.6, 0.7, seed=2), belt(122.4, 0.35, seed=3),
             np.array([[123.3, -31], [124.5, -31], [124.5, -29], [123.3, -29], [123.3, -31]])]
    colors = [GRANITE, GNEISS, GREEN, GREEN, GREEN, SED]
    parents = ["Granitic rock", "Gneiss", "Greenstone (mafic-ultramafic)",
               "Greenstone (mafic-ultramafic)", "Greenstone (mafic-ultramafic)",
               "Sedimentary cover"]
    return xs.PolygonIndex(np.array(rings, dtype=object), colors, parents)


def make_fake_tectonic():
    rings = [np.array([[115.5, -33], [117.6, -33], [117.4, -27], [115.5, -27], [115.5, -33]]),
             np.array([[117.6, -33], [121.3, -33], [121.0, -27], [117.4, -27], [117.6, -33]]),
             np.array([[121.3, -33], [124.5, -33], [124.5, -27], [121.0, -27], [121.3, -33]])]
    return xs.PolygonIndex(np.array(rings, dtype=object), ["#c7d7ef", "#e9c9a0", "#bfe3c0"],
                           ["Narryer-like", "Youanmi-like", "Eastern Goldfields-like"])


def make_fake_boundaries():
    lines = [np.array([[117.6, -33], [117.4, -27]]),
             np.array([[121.3, -33], [121.15, -30], [121.0, -27]]),
             np.array([[121.3, -33], [121.15, -30], [121.0, -27]]) + [0.004, 0]]  # dup
    return xs.BoundarySet(np.array(lines, dtype=object),
                          names=["Boundary 1", "Boundary 2", "Boundary 2"])


def fake_vs(d, z, litho_runs):
    vs = 3.45 + 0.012 * z
    vs = np.where(z[:, None] > 38, 4.55 + 0.002 * (z[:, None] - 38), vs[:, None] * np.ones_like(d))
    for r in litho_runs:
        if r.name.startswith("Greenstone"):
            mid, half = 0.5 * (r.d0 + r.d1), 0.5 * (r.d1 - r.d0)
            lat = np.exp(-((d - mid) / max(half, 10)) ** 4)
            vs += 0.18 * lat[None, :] * np.exp(-(z[:, None] / 10) ** 2)
    return vs


def main(out="demo_xsection_strips.png"):
    litho, tect, bnd = make_fake_litho(), make_fake_tectonic(), make_fake_boundaries()
    lon, lat, d = xs.great_circle_points(116.0, -30.3, 124.0, -29.7, step_km=5)
    z = np.linspace(0, 60, 121)

    lon_f, lat_f, d_f = xs.resample_profile(lon, lat, d, 0.5)
    runs = xs.runs_from_samples(d_f, litho.sample(lon_f, lat_f), litho)
    vs = fake_vs(d, z, runs)
    dln = 100 * (vs / vs.mean(axis=1, keepdims=True) - 1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True,
                            gridspec_kw=dict(hspace=0.12, top=0.80, bottom=0.17,
                                             left=0.08, right=0.92))
    axs[0].set_title("Synthetic section A–A′", pad=4)
    for ax, f, cmap, lim, lab in ((axs[0], vs, "jet_r", (3.4, 4.0), "Vsv (km/s)"),
                                  (axs[1], dln, "RdBu", (-4, 4), "dlnVsv (%)")):
        m = ax.pcolormesh(d, z, f, cmap=cmap, vmin=lim[0], vmax=lim[1], shading="auto")
        ax.plot(d, np.full_like(d, 38), "k-", lw=1.2)
        ax.set_ylim(60, 0)
        ax.set_ylabel("Depth (km)")
        fig.colorbar(m, ax=ax, pad=0.01, fraction=0.025, label=lab)
    axs[1].set_xlabel("Distance (km)")
    axs[1].xaxis.set_major_locator(plt.MultipleLocator(100))

    res = xs.add_profile_strips(axs[0], axs, lon, lat, d, litho=litho, tectonic=tect,
                                boundaries=bnd, label_boundaries=False)
    handles = xs.legend_handles([res.runs["litho"]]) + xs.legend_handles([res.runs["tectonic"]])
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.text(0.92, 0.005, "Geology: synthetic (real data © GSWA, CC-BY-4.0)",
             ha="right", fontsize=6, color="0.4")
    fig.savefig(out, dpi=150)
    print("crossings (km):", [round(c.dist, 1) for c in res.crossings])
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
