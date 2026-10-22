%% plot_moho_map_WA.m
%  WA Array refined CCP Moho on the GSWA 2022 tectonic base, in GSWA Albers:
%  tectonic units UNFILLED (outlines only), major crustal boundaries, stations,
%  Moho depth as the colour field. One panel.
%
%  On the MATLAB path:  wa_basemap_axes.m, wa_albers.m, add_wa_station_layers.m
%     addpath /data/25/yuan/github/wa-geo-plotting/matlab
%  Base data file, built once from the repo root:
%     python3 plotting/export_matlab.py --moho $MOHO --load-sections $XS --out wa_plotdata.mat

% ---- paths (edit to taste) ---------------------------------------------
BASE = 'wa_plotdata.mat';      % the exported tectonic base (lon/lat; projected here)
MOHO = '/workspace/CCP/CCP.WA.Array.2027.phase.3/Moho.picked/moho_ccp_refined_AR23grid.txt';
DL   = '/mnt/c/Users/MIHYUAN/Downloads';
PERM = fullfile(DL, 'New.WA.Seismic.2026- Perm-phase-0.csv');
P4   = fullfile(DL, 'New.WA.Seismic.2026- Phase 4.csv');
CLIM = [30 55];                % km, as the 4-panel figure
OUT  = 'fig.CCP.moho.map';

S = load(BASE);

% ---- refined Moho grid: lon lat moho_km, 11 header lines (MOHO_SKIP_HEADER) --
M    = readmatrix(MOHO, 'NumHeaderLines', 11);
lon  = M(:,1);  lat = M(:,2);  moho = M(:,3);
good = isfinite(lon) & isfinite(lat) & isfinite(moho);
lon  = lon(good); lat = lat(good); moho = moho(good);
[mx, my] = wa_albers(lon, lat);              % into the base's Albers km

% ---- draw ---------------------------------------------------------------
figure('Color','w', 'Units','centimeters', 'Position',[1 1 18 20], ...
       'Name','WA Array refined CCP Moho');
ax = axes; hold(ax, 'on');

% Moho field FIRST (bottom layer); the base line-work is drawn on top of it so
% boundaries / coast / unit outlines stay visible over the colour field.
scatter(ax, mx, my, 22, moho, 's', 'filled', 'MarkerEdgeColor','none');

% tectonic base with units UNFILLED: UnitAlpha 0 keeps the thin unit outlines
% but shows the Moho field through them. Set 'Units',false to drop unit outlines
% entirely and keep only the major crustal boundaries + coast.
wa_basemap_axes(ax, S, 'UnitAlpha', 0, 'Boundaries', true, 'Coast', true, ...
                'Graticule', true, 'Labels', false);

% stations on top, projected to Albers; Permanent labelled with station codes
% ('LabelPermanent',false to drop the codes if the whole-state map gets busy).
add_wa_station_layers(ax, 'Permanent', PERM, 'Phase4', P4, 'Project','albers');
% To also show the deployed WA Array from the base file as black triangles, add
%   wa_basemap_axes(..., 'Stations', true)  above, or pass an Available CSV here.

colormap(ax, flipud(jet(256)));              % shallow = red, deep = blue (as panel 1)
set(ax, 'CLim', CLIM);
cb = colorbar(ax, 'eastoutside');  cb.Label.String = 'Moho depth (km)';  cb.FontSize = 8;
title(ax, 'WA Array phase-3 refined CCP Moho  |  AR23 grid', 'FontWeight','normal');

exportgraphics(gcf, [OUT '.png'], 'Resolution', 250);
exportgraphics(gcf, [OUT '.pdf'], 'ContentType','vector');
fprintf('Saved %s.{png,pdf}\n', OUT);


% ======================================================================
% ALTERNATIVE: a smooth filled Moho surface from the interpolant instead of the
% node scatter. Replace the scatter(...) line above with this block:
%
%   F  = load('/workspace/CCP/CCP.WA.Array.2027.phase.3/Moho.picked/Fmoho_refined.mat');
%   Fm = F.Fmoho_refined;                               % query Fm(lat, lon)
%   gl = linspace(S.extent(1), S.extent(2), 500);
%   ga = linspace(S.extent(3), S.extent(4), 500);
%   [GLON, GLAT] = meshgrid(gl, ga);
%   Z = Fm(GLAT, GLON);                                 % NaN outside coverage -> transparent
%   [GX, GY] = wa_albers(GLON, GLAT);
%   surf(ax, GX, GY, zeros(size(Z)), Z, 'EdgeColor','none'); view(ax, 2);
%
% (A scatteredInterpolant returns NaN outside its convex hull so the sea/edges
%  stay blank; if Fmoho_refined is a griddedInterpolant set its
%  ExtrapolationMethod to 'none' first to avoid painting Moho offshore.)
