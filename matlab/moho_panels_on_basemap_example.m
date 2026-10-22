%% moho_panels_on_basemap_example.m
%  Template: the 4-panel CCP Moho figure (as from pick_moho_ccp_auto.m) drawn
%  on the GSWA 2022 simplified tectonic basemap, in the GSWA Albers projection,
%  the same whole-state base the Python locator uses.
%
%  Needs on the MATLAB path:  wa_basemap_axes.m, wa_albers.m
%  Needs once (made on d204403 from the wa-geo-plotting repo):
%     python3 plotting/export_matlab.py --moho $MOHO --load-sections $XS \
%         --out wa_plotdata.mat
%  The .mat carries the whole-state base (units/boundaries/coast/outline/
%  sea+states) in lon/lat; wa_basemap_axes projects it with wa_albers.
%
%  Replace the demo block below with your picker outputs. Each field is a
%  per-node vector aligned with node_lon / node_lat (the 6900 valid nodes).

BASE = 'wa_plotdata.mat';          % path to the exported base
S    = load(BASE);

% ---- picker outputs (SUBSTITUTE THESE) ---------------------------------
% node_lon, node_lat : [N x 1] node coordinates (deg)
% moho_km            : [N x 1] blended CCP Moho depth (km)
% conf               : [N x 1] confidence flag (1 strict, 2 relaxed)
% iband_std          : [N x 1] inter-band STD (km)   (0/NaN when 1 volume)
% ar23_km            : [N x 1] AusMoho AR23 at the node (km)
% --- demo stand-ins so the script runs on its own; delete for real data:
if ~exist('node_lon','var')
    rng(0);
    node_lon = 113 + 12*rand(6900,1);
    node_lat = -35 + 12.5*rand(6900,1);
    moho_km   = 38 + 6*randn(6900,1);
    conf      = 1 + (rand(6900,1) > 0.8);
    iband_std = zeros(6900,1);                 % 1-volume run -> all zero
    ar23_km   = 40 + 3*randn(6900,1);
end
ccp_minus_ar23 = moho_km - ar23_km;

% ---- panel spec: field, title, colormap, colour limits -----------------
panels = { ...
    moho_km,        'CCP Moho (km)',        flipud(jet(256)), [30 55]; ...
    conf,           'Confidence (1/2)',     [0 0.6 0.3; 1 0.6 0], [1 2]; ...
    iband_std,      'Inter-band STD (km)',  hot(256),         [0 5]; ...
    ccp_minus_ar23, 'CCP - AR23 (km)',      local_rwb(256),   [-8 8] };

% ---- project node coords ONCE into the basemap's Albers km -------------
[mx, my] = wa_albers(node_lon, node_lat);

% ---- draw ---------------------------------------------------------------
figure('Color','w', 'Units','centimeters', 'Position',[1 1 32 22], ...
       'Name','CCP Moho on GSWA tectonic base');
for k = 1:size(panels,1)
    ax = subplot(2, 2, k);

    % whole-state tectonic base, dimmed so the data field reads on top.
    % 'UnitAlpha',0.35 keeps the geology legible under the colour field;
    % use 'Units',false to drop the fills and keep only coast/boundaries.
    wa_basemap_axes(ax, S, 'UnitAlpha', 0.35, 'Labels', false);

    % Moho field on top (node scatter). For a gridded field instead, build
    % [GX,GY] = wa_albers(grid_lon, grid_lat) and use surf/pcolor(ax,GX,GY,F).
    scatter(ax, mx, my, 16, panels{k,1}, 'filled', 'MarkerEdgeColor','none');

    colormap(ax, panels{k,3});
    set(ax, 'CLim', panels{k,4});
    cb = colorbar(ax, 'eastoutside'); cb.FontSize = 7;
    title(ax, panels{k,2}, 'FontWeight','normal');

    % OPTIONAL: Available/Permanent/Phase 4 station layers. This base is Albers
    % km, so pass 'Project','albers' and the helper projects the station lon/lat
    % through wa_albers to register on it (default is lon/lat, for the locator).
    % add_wa_station_layers(ax, 'Permanent', PERM_CSV, 'Phase4', P4_CSV, 'Project','albers');
end

sgtitle('Phase 3 CCP Moho on GSWA 2022 tectonic base  |  AR23 reference', ...
        'FontWeight','bold');

% exportgraphics(gcf, 'fig.CCP.moho.base.png', 'Resolution', 250);


% ------------------------------------------------------------------------
function cmap = local_rwb(n)
%LOCAL_RWB  blue-white-red diverging map (stand-in for your cmapx.mat).
    if nargin < 1, n = 256; end
    m = ceil(n/2);
    up = linspace(0, 1, m)';
    lower = [up, up, ones(m,1)];                 % blue -> white
    upper = [ones(m,1), flipud(up), flipud(up)]; % white -> red
    cmap = [lower; upper(2:end,:)];
    if size(cmap,1) > n, cmap = cmap(1:n,:); end
end
