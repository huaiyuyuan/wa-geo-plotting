"""
config.py — local paths for wa-geo-plotting. EDIT THESE for your machine.
Everything WA-specific and machine-specific lives here; the plotting code is generic.
"""
import os

# Root of the GSWA / other shapefiles
SHAPEFILE_ROOT = os.environ.get('WA_SHAPEFILE_ROOT', '/workspace/shape.files')

# Derived basemap data (npz caches produced by extract/ scripts)
DERIVED = os.environ.get('WA_DERIVED', os.path.join(SHAPEFILE_ROOT, 'derived'))
TECTONICS_NPZ  = os.path.join(DERIVED, 'wa_tectonics.npz')    # 10M tectonic units
LITHOLOGY_NPZ  = os.path.join(DERIVED, 'wa_lithology.npz')    # 500k lithology
BOUNDARIES_NPZ = os.path.join(DERIVED, 'wa_crustal_boundaries.npz')  # major crustal boundaries (lines)

# Source shapefiles (for regenerating the npz caches)
SHP_TECTONICS  = os.path.join(SHAPEFILE_ROOT, 'WA_Tectonic_Units_10M_2021/WA_Tectonic_Units_10M_2021.shp')
SHP_LITHOLOGY  = os.path.join(SHAPEFILE_ROOT, 'GEOLOGY_500k_Tectonics_GDA2020_SHP/ESRI/SHAPEFILES/500k_tectonicp.shp')
SHP_BOUNDARIES = os.path.join(SHAPEFILE_ROOT, 'Major.Crustal.Boundaries.WA.2025/MajorCrustalBoundaries.shp')

# Moho model (for overlays on sections / depth slices)
MOHO = os.environ.get('WA_MOHO', '/workspace/Others.data/AusMoho2023/AR23-moho-hmp.txt')
MOHO_SKIP_HEADER = 11

# Stations (code lat lon, whitespace or comma)
STATIONS = os.environ.get('WA_STATIONS',
    '/workspace/WA.Array/Year.3/noise.processing/ASDF.processing/pathqc.v2/stations_v2.txt')
STATION_COLS = (2, 1)   # (lon_col, lat_col) 0-indexed

# Default WA map extent (lon_min, lon_max, lat_min, lat_max)
WA_EXTENT = [112, 130, -36, -13]
