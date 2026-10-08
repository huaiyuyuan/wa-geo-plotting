function [x, y] = wa_albers(lon, lat)
%WA_ALBERS  Albers equal-area projection of the GSWA 2022 simplified tectonic map of WA.
%   [x, y] = wa_albers(lon, lat)   lon/lat in degrees (any shape, NaN passes through);
%   x, y in km (sphere, R = 6371 km), x east and y north of the central point.
%   Central meridian 121E, standard parallels 17.5S and 31.5S - fitted to the printed
%   graticule of the map PDF (0.1 pt rms), and used by plot_xsection.py --index-proj albers.
R    = 6371.0;
lon0 = 121.0;
p1   = -17.5 * pi / 180;
p2   = -31.5 * pi / 180;
n    = (sin(p1) + sin(p2)) / 2;
C    = cos(p1)^2 + 2 * n * sin(p1);
rho0 = R * sqrt(C) / n;
rho  = R * sqrt(C - 2 * n * sin(lat * pi / 180)) / n;
th   = n * (lon - lon0) * pi / 180;
x    = rho .* sin(th);
y    = rho0 - rho .* cos(th);
end
