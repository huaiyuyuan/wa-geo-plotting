"""
config.py — local paths for wa-geo-plotting. EDIT THESE for your machine.
Everything WA-specific and machine-specific lives here; the plotting code is generic.

Data now lives under the repo data/ tree by default (data/geology, data/stations,
data/deposits), so a fresh clone plots with no /workspace or ~/Downloads dependency.
Every path still honours its $WA_* env override if you want to point elsewhere.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))              # repo root
DATA  = os.environ.get('WA_DATA', os.path.join(_HERE, 'data'))  # bundled data tree

# Root of the GSWA / other source shapefiles (only needed to REGENERATE the npz)
SHAPEFILE_ROOT = os.environ.get('WA_SHAPEFILE_ROOT', '/workspace/shape.files')

# Derived basemap data (npz caches produced by extract/ scripts) — bundled in data/geology/
DERIVED = os.environ.get('WA_DERIVED', os.path.join(DATA, 'geology'))
TECTONICS_NPZ  = os.path.join(DERIVED, 'wa_tectonics.npz')    # 10M tectonic units
LITHOLOGY_NPZ  = os.path.join(DERIVED, 'wa_lithology.npz')    # 500k lithology
BOUNDARIES_NPZ = os.path.join(DERIVED, 'wa_crustal_boundaries.npz')  # major crustal boundaries (lines)

# Source shapefiles (for regenerating the npz caches) — external, under SHAPEFILE_ROOT
SHP_TECTONICS  = os.path.join(SHAPEFILE_ROOT, 'WA_Tectonic_Units_10M_2021/WA_Tectonic_Units_10M_2021.shp')
SHP_LITHOLOGY  = os.path.join(SHAPEFILE_ROOT, 'GEOLOGY_500k_Tectonics_GDA2020_SHP/ESRI/SHAPEFILES/500k_tectonicp.shp')
SHP_BOUNDARIES = os.path.join(SHAPEFILE_ROOT, 'Major.Crustal.Boundaries.WA.2025/MajorCrustalBoundaries.shp')

# Moho model (for overlays on sections / depth slices)
MOHO = os.environ.get('WA_MOHO', '/workspace/Others.data/AusMoho2023/AR23-moho-hmp.txt')
MOHO_SKIP_HEADER = 11

# Stations — bundled under data/stations/ (code lat lon, whitespace or comma; or GSWA
# 'New.WA.Seismic' CSVs with Latitude/Longitude columns, auto-detected by the loaders)
STATIONS = os.environ.get('WA_STATIONS', os.path.join(DATA, 'stations', 'stations_v2.txt'))
STATION_COLS = (2, 1)   # (lon_col, lat_col) 0-indexed, for the whitespace stations file
# GSWA deployment exports (green-square Phase 4 layer, Permanent layer)
STATIONS_PHASE4 = os.environ.get('WA_STATIONS_PHASE4',
    os.path.join(DATA, 'stations', 'New.WA.Seismic.2026.Phase.4.csv'))
STATIONS_PERM = os.environ.get('WA_STATIONS_PERM',
    os.path.join(DATA, 'stations', 'New.WA.Seismic.2026.Perm.phase.0.csv'))

# Mineral-deposit / greenstone overlay datasets — bundled under data/deposits/
DEPOSIT_DIR = os.environ.get('WA_DEPOSIT_DIR', os.path.join(DATA, 'deposits'))

# WA Array coverage footprint (ray-path density at 10 s; see the README in that dir)
#   npz 'mask' (70x60 inversion grid) selects cells; outline .txt is for drawing only
FOOTPRINT_DIR = os.environ.get('WA_FOOTPRINT_DIR',
    '/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/footprint')
FOOTPRINT_NPZ     = os.path.join(FOOTPRINT_DIR, 'wa_array_footprint.npz')
FOOTPRINT_OUTLINE = os.path.join(FOOTPRINT_DIR, 'wa_array_outline.txt')

# Default WA map extent (lon_min, lon_max, lat_min, lat_max)
WA_EXTENT = [112, 130, -36, -13]
