function h = wa_basemap_axes(ax, src, varargin)
%WA_BASEMAP_AXES  Draw the GSWA 2022 simplified tectonic basemap onto one axes.
%   h = wa_basemap_axes(ax, 'wa_plotdata.mat', 'Name',Value, ...)
%   h = wa_basemap_axes(ax, S,                 'Name',Value, ...)
%
%   The whole-state WA basemap from wa_plot_map.m -- 2022 tectonic units,
%   crustal boundaries, sea/other states, graticule, coast + WA outline -- in
%   the GSWA Albers projection (via wa_albers), drawn onto an EXISTING axes AX
%   instead of a whole figure. No sections, no legend chart, no title, no file
%   is written. Made for overlaying a data field (e.g. CCP Moho depth per node)
%   on a consistent basemap across several panels.
%
%   SRC is the wa_plotdata.mat produced by plotting/export_matlab.py, given as a
%   path or an already-loaded struct. It carries everything in lon/lat; this
%   function projects with wa_albers, so AX ends up in Albers km -- project your
%   own overlay the same way before plotting it:
%       [mx, my] = wa_albers(node_lon, node_lat);
%       scatter(ax, mx, my, 14, moho_km, 'filled');      % on top of the base
%
%   Draw order (matches wa_plot_map.m): units -> boundaries -> sea/states ->
%   graticule -> coast/WA outline. Everything is added at the bottom-to-top
%   order a basemap wants, so a data field plotted AFTER this call with a normal
%   zorder sits above the fills; the coast/outline/graticule are thin line work
%   that still reads through a semi-transparent field. For a filled Moho colour
%   field that would otherwise bury the geology, dim the base with 'UnitAlpha'
%   (e.g. 0.35) or turn units off with 'Units',false and keep only the outlines.
%
%   Name/Value options
%     'Extent'         [lon0 lon1 lat0 lat1] frame        (default: S.extent)
%     'Units'          draw filled tectonic units         (true)
%     'UnitAlpha'      face alpha for the units           (1)
%     'Boundaries'     draw major crustal boundaries      (true)
%     'Ocean'          draw sea (white) + other states    (true)
%     'Graticule'      draw the 5-deg graticule           (true)
%     'GraticuleLabels'label the graticule on L/bottom    (true)
%     'Coast'          draw coast + WA outline on top      (true)
%     'Labels'         draw GSWA unit names               (false)
%     'Stations'       draw S.stations as black triangles  (false)
%                      (for the Available/Permanent/Phase 4 layers use
%                       add_wa_station_layers(ax,...) after this call instead)
%     'Tag'            tag put on every object drawn       ('wa_basemap')
%
%   Returns h, a struct of handle arrays by layer.
%
%   Needs wa_albers.m on the path. 'Labels' uses a condensed font if installed.
%
%   Example (2x2 CCP Moho panels, Albers, on the state tectonic base):
%       S  = load('wa_plotdata.mat');
%       ff = {moho, conf, iband_std, moho_minus_ar23};
%       tt = {'CCP Moho (km)','Confidence','Inter-band STD (km)','CCP - AR23 (km)'};
%       [mx,my] = wa_albers(node_lon, node_lat);
%       figure('Color','w');
%       for k = 1:4
%           ax = subplot(2,2,k);
%           wa_basemap_axes(ax, S, 'UnitAlpha',0.35, 'Labels',false);
%           scatter(ax, mx, my, 16, ff{k}, 'filled');     % Moho field on top
%           colorbar(ax); title(ax, tt{k});
%       end

    % ---- source struct ---------------------------------------------------
    if ischar(src) || isstring(src)
        S = load(char(src));
    elseif isstruct(src)
        S = src;
    else
        error('wa_basemap_axes:src', 'SRC must be a .mat path or a loaded struct.');
    end

    p = inputParser;
    p.addRequired('ax');
    p.addParameter('Extent', []);
    p.addParameter('Units', true);
    p.addParameter('UnitAlpha', 1);
    p.addParameter('Boundaries', true);
    p.addParameter('Ocean', true);
    p.addParameter('Graticule', true);
    p.addParameter('GraticuleLabels', true);
    p.addParameter('Coast', true);
    p.addParameter('Labels', false);
    p.addParameter('Stations', false);
    p.addParameter('Tag', 'wa_basemap');
    p.parse(ax, varargin{:});
    o = p.Results;
    TAG = {'Tag', o.Tag};

    if exist('wa_albers', 'file') ~= 2
        error('wa_basemap_axes:noproj', 'wa_albers.m must be on the MATLAB path.');
    end

    warning('off', 'MATLAB:polyshape:repairedBySimplify');
    warning('off', 'MATLAB:polyshape:boundary3Points');

    if isempty(o.Extent), ext = double(S.extent(:)'); else, ext = double(o.Extent(:)'); end

    % ---- axes frame: project the extent box (Albers edges bow) -----------
    t = linspace(0, 1, 120);
    blon = [ext(1)+t*(ext(2)-ext(1)), ext(2)*ones(1,120), ext(2)-t*(ext(2)-ext(1)), ext(1)*ones(1,120)];
    blat = [ext(3)*ones(1,120), ext(3)+t*(ext(4)-ext(3)), ext(4)*ones(1,120), ext(4)-t*(ext(4)-ext(3))];
    [bx, by] = wa_albers(blon, blat);
    xl = [min(bx) max(bx)];
    yl = [min(by) max(by)];

    washold = ishold(ax);
    hold(ax, 'on');
    set(ax, 'XLim', xl, 'YLim', yl, 'DataAspectRatio', [1 1 1], ...
        'XTick', [], 'YTick', [], 'Box', 'on', 'Color', 'w', ...
        'Layer', 'top', 'SortMethod', 'childorder', 'LineWidth', 0.8);
    offx = 0.012 * diff(xl);           % label offsets in km, panel-size robust
    offy = 0.012 * diff(yl);

    h = struct('units',[], 'bnd',[], 'ocean',[], 'states',[], 'grat',[], ...
               'coast',[], 'wa',[], 'labels',[], 'stations',[]);

    % ---- tectonic units (2022 map colours), big first -------------------
    if o.Units && isfield(S, 'units')
        U = S.units;
        hu = gobjects(numel(U.xy), 1);
        for k = 1:numel(U.xy)
            xy = U.xy{k};
            [x, y] = wa_albers(xy(:,1), xy(:,2));
            ps = polyshape(x, y, 'SolidBoundaryOrientation', 'ccw');
            hu(k) = plot(ax, ps, 'FaceColor', U.rgb(k,:), 'FaceAlpha', o.UnitAlpha, ...
                         'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 0.2, TAG{:});
        end
        h.units = hu;
    end

    % ---- major crustal boundaries ---------------------------------------
    if o.Boundaries && isfield(S, 'bnd')
        Bd = S.bnd;
        hb = gobjects(numel(Bd.xy), 1);
        for k = 1:numel(Bd.xy)
            xy = Bd.xy{k};
            [x, y] = wa_albers(xy(:,1), xy(:,2));
            lw = 1.1; if Bd.crustal(k), lw = 0.5; end
            ls = '-';  if Bd.low(k),     ls = '--'; end
            hb(k) = line(ax, x, y, 'Color', [0 0 0], 'LineWidth', lw, 'LineStyle', ls, TAG{:});
        end
        h.bnd = hb;
    end

    % ---- sea (white) and other states (pale grey) -----------------------
    if o.Ocean
        if isfield(S, 'ocean')
            ho = gobjects(numel(S.ocean), 1);
            for k = 1:numel(S.ocean)
                xy = S.ocean{k};
                [x, y] = wa_albers(xy(:,1), xy(:,2));
                ho(k) = plot(ax, polyshape(x, y, 'SolidBoundaryOrientation', 'ccw'), ...
                    'FaceColor', 'w', 'FaceAlpha', 1, 'EdgeColor', 'none', TAG{:});
            end
            h.ocean = ho;
        end
        if isfield(S, 'states')
            hs = gobjects(numel(S.states), 1);
            for k = 1:numel(S.states)
                xy = S.states{k};
                [x, y] = wa_albers(xy(:,1), xy(:,2));
                hs(k) = plot(ax, polyshape(x, y, 'SolidBoundaryOrientation', 'ccw'), ...
                    'FaceColor', [0.91 0.91 0.91], 'FaceAlpha', 1, ...
                    'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 0.4, TAG{:});
            end
            h.states = hs;
        end
    end

    % ---- graticule: 5 deg, thin grey ------------------------------------
    if o.Graticule
        gcol = [133 140 140] / 255;
        latd = -50:0.25:5;  lond = 95:0.25:155;
        glon = 100:5:150;   glat = -45:5:0;
        hg = gobjects(0);
        for lo = glon
            [x, y] = wa_albers(lo*ones(size(latd)), latd);
            hg(end+1) = line(ax, x, y, 'Color', gcol, 'LineWidth', 0.35, TAG{:}); %#ok<AGROW>
            if o.GraticuleLabels
                xb = interp1(y, x, yl(1));
                if ~isnan(xb) && xb > xl(1) && xb < xl(2)
                    text(ax, xb, yl(1) - offy, sprintf('%g%c', lo, char(176)), 'Color', gcol, ...
                        'FontSize', 8, 'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'top', 'Clipping', 'off', TAG{:});
                end
            end
        end
        for la = glat
            [x, y] = wa_albers(lond, la*ones(size(lond)));
            hg(end+1) = line(ax, x, y, 'Color', gcol, 'LineWidth', 0.35, TAG{:}); %#ok<AGROW>
            if o.GraticuleLabels
                yb = interp1(x, y, xl(1));
                if ~isnan(yb) && yb > yl(1) && yb < yl(2)
                    text(ax, xl(1) - offx, yb, sprintf('%g%c', abs(la), char(176)), 'Color', gcol, ...
                        'FontSize', 8, 'HorizontalAlignment', 'right', ...
                        'VerticalAlignment', 'middle', 'Clipping', 'off', TAG{:});
                end
            end
        end
        h.grat = hg;
    end

    % ---- optional GSWA unit names ---------------------------------------
    if o.Labels && isfield(S, 'labels') && isfield(S.labels, 'text') && ~isempty(S.labels.text)
        Lb = S.labels;
        if ischar(Lb.text), Lb.text = {Lb.text}; end
        [lx, ly] = wa_albers(Lb.lon(:), Lb.lat(:));
        off = 0.004 * diff(xl);                      % halo offset in km
        dirs = [1 0; -1 0; 0 1; 0 -1; 0.7 0.7; -0.7 0.7; 0.7 -0.7; -0.7 -0.7];
        fam = local_narrow_font();
        ht = gobjects(0);
        for k = 1:numel(Lb.text)
            str = strsplit(Lb.text{k}, char(10));
            wt = 'normal'; if Lb.bold(k), wt = 'bold'; end
            args = {'FontName', fam, 'FontSize', Lb.size(k), 'FontWeight', wt, ...
                    'Rotation', Lb.angle(k), 'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', 'Clipping', 'on'};
            for q = 1:size(dirs, 1)
                ht(end+1) = text(ax, lx(k)+off*dirs(q,1), ly(k)+off*dirs(q,2), str, ...
                    'Color', [1 1 1], args{:}, TAG{:}); %#ok<AGROW>
            end
            ht(end+1) = text(ax, lx(k), ly(k), str, 'Color', Lb.rgb(k,:), args{:}, TAG{:}); %#ok<AGROW>
        end
        h.labels = ht;
    end

    % ---- optional stations (plain triangles; richer layers via helper) --
    if o.Stations && isfield(S, 'stations') && ~isempty(S.stations)
        [x, y] = wa_albers(S.stations(:,1), S.stations(:,2));
        h.stations = plot(ax, x, y, '^', 'MarkerSize', 4.5, 'MarkerFaceColor', 'k', ...
            'MarkerEdgeColor', 'w', 'LineWidth', 0.35, 'LineStyle', 'none', TAG{:});
    end

    % ---- coast and WA outline on top ------------------------------------
    if o.Coast
        if isfield(S, 'coast')
            hc = gobjects(numel(S.coast), 1);
            for k = 1:numel(S.coast)
                xy = S.coast{k};
                [x, y] = wa_albers(xy(:,1), xy(:,2));
                hc(k) = line(ax, x, y, 'Color', 'k', 'LineWidth', 0.8, TAG{:});
            end
            h.coast = hc;
        end
        if isfield(S, 'wa')
            hw = gobjects(numel(S.wa), 1);
            for k = 1:numel(S.wa)
                xy = S.wa{k};
                [x, y] = wa_albers(xy(:,1), xy(:,2));
                hw(k) = line(ax, x, y, 'Color', 'k', 'LineWidth', 0.8, TAG{:});
            end
            h.wa = hw;
        end
    end

    if ~washold, hold(ax, 'off'); end
end


function fam = local_narrow_font()
% A condensed sans like the GSWA map's Arial Narrow when installed.
    have = listfonts;
    fam = 'Helvetica';
    for c = {'Arial Narrow', 'Liberation Sans Narrow', 'Nimbus Sans Narrow', 'Arial'}
        if any(strcmpi(have, c{1})), fam = c{1}; return; end
    end
end
