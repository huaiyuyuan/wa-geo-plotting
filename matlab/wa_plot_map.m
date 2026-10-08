function fig = wa_plot_map(matfile, outfile)
%WA_PLOT_MAP  Cross-section location map in the style of the GSWA 2022 simplified
%   tectonic map of WA - the same figure as plot_xsection.py --index-map (map2022 style).
%
%   wa_plot_map('wa_plotdata.mat')              -> xsection_index_map.png / .pdf beside it
%   wa_plot_map('wa_plotdata.mat', 'mymap')     -> mymap.png / mymap.pdf
%
%   The .mat comes from plotting/export_matlab.py. Base MATLAB only (R2017b+ for
%   polyshape; exportgraphics used when available, R2020a+). Layers, colours, line
%   widths and order follow the Python figure:
%     units -> crustal boundaries -> footprint veil -> sea / other states -> graticule
%     -> stations -> sections -> unit labels -> coast / WA outline -> legends.

if nargin < 2 || isempty(outfile)
    p = fileparts(matfile);
    outfile = fullfile(p, 'xsection_index_map');
end
S = load(matfile);
warning('off', 'MATLAB:polyshape:repairedBySimplify');
warning('off', 'MATLAB:polyshape:boundary3Points');

% ---------- figure and map box (13 x 13 in; map fills [0.06 0.05 0.88 0.90]) ----------
FIG = 13;
ext = double(S.extent(:)');                          % [lon0 lon1 lat0 lat1]
t = linspace(0, 1, 120);
blon = [ext(1) + t*(ext(2)-ext(1)), ext(2)*ones(1,120), ext(2) - t*(ext(2)-ext(1)), ext(1)*ones(1,120)];
blat = [ext(3)*ones(1,120), ext(3) + t*(ext(4)-ext(3)), ext(4)*ones(1,120), ext(4) - t*(ext(4)-ext(3))];
[bx, by] = wa_albers(blon, blat);
xl = [min(bx) max(bx)];
yl = [min(by) max(by)];
r = diff(yl) / diff(xl);
w = min(0.88*FIG, 0.90*FIG/r);
h = w * r;
pos = [0.06*FIG + (0.88*FIG - w)/2, 0.05*FIG + (0.90*FIG - h)/2, w, h];

fig = figure('Units', 'inches', 'Position', [0.5 0.5 FIG FIG], 'Color', 'w', ...
             'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
ax = axes('Parent', fig, 'Units', 'inches', 'Position', pos);
hold(ax, 'on');
set(ax, 'XLim', xl, 'YLim', yl, 'DataAspectRatio', [1 1 1], 'XTick', [], 'YTick', [], ...
    'Box', 'on', 'Color', 'w', 'Layer', 'top', 'LineWidth', 0.8, 'SortMethod', 'childorder');
kmpt = diff(xl) / (w * 72);                          % km per typographic point on paper
fam = narrow_font();

% ---------- tectonic units (2022 map colours), big first ----------
U = S.units;
for k = 1:numel(U.xy)
    xy = U.xy{k};
    [x, y] = wa_albers(xy(:,1), xy(:,2));
    ps = polyshape(x, y, 'SolidBoundaryOrientation', 'ccw');
    plot(ax, ps, 'FaceColor', U.rgb(k,:), 'FaceAlpha', 1, ...
         'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 0.2);
end

% ---------- major crustal boundaries ----------
Bd = S.bnd;
for k = 1:numel(Bd.xy)
    xy = Bd.xy{k};
    [x, y] = wa_albers(xy(:,1), xy(:,2));
    lw = 1.1; if Bd.crustal(k), lw = 0.5; end
    ls = '-';  if Bd.low(k), ls = '--'; end
    line(ax, x, y, 'Color', [0 0 0], 'LineWidth', lw, 'LineStyle', ls);
end

% ---------- footprint (optional): veil outside + outline ----------
F = S.footprint;
if ~isempty(F.xy) && size(F.xy, 1) > 2
    [fx, fy] = wa_albers(F.xy(:,1), F.xy(:,2));
    if F.veil > 0
        L1 = [linspace(95,150,200), 150*ones(1,200), linspace(150,95,200), 95*ones(1,200)];
        L2 = [-50*ones(1,200), linspace(-50,0,200), zeros(1,200), linspace(0,-50,200)];
        [vx, vy] = wa_albers(L1, L2);
        veil = subtract(polyshape(vx, vy), polyshape(fx, fy));
        plot(ax, veil, 'FaceColor', 'w', 'FaceAlpha', F.veil, 'EdgeColor', 'none');
    end
end

% ---------- sea (white) and the other states (pale grey) ----------
for k = 1:numel(S.ocean)
    xy = S.ocean{k};
    [x, y] = wa_albers(xy(:,1), xy(:,2));
    plot(ax, polyshape(x, y, 'SolidBoundaryOrientation', 'ccw'), 'FaceColor', 'w', ...
         'FaceAlpha', 1, 'EdgeColor', 'none');
end
for k = 1:numel(S.states)
    xy = S.states{k};
    [x, y] = wa_albers(xy(:,1), xy(:,2));
    plot(ax, polyshape(x, y, 'SolidBoundaryOrientation', 'ccw'), 'FaceColor', [0.91 0.91 0.91], ...
         'FaceAlpha', 1, 'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 0.4);
end

% ---------- graticule: 5 deg, thin grey, over land and sea ----------
gcol = [133 140 140] / 255;
latd = -50:0.25:5;  lond = 95:0.25:155;
glon = 100:5:150;   glat = -45:5:0;
for lo = glon
    [x, y] = wa_albers(lo*ones(size(latd)), latd);
    line(ax, x, y, 'Color', gcol, 'LineWidth', 0.35);
    xb = interp1(y, x, yl(1));                       % label where it meets the bottom edge
    if ~isnan(xb) && xb > xl(1) && xb < xl(2)
        text(ax, xb, yl(1) - 4*kmpt, sprintf('%g%c', lo, char(176)), 'Color', gcol, ...
             'FontSize', 8, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
             'Clipping', 'off');
    end
end
for la = glat
    [x, y] = wa_albers(lond, la*ones(size(lond)));
    line(ax, x, y, 'Color', gcol, 'LineWidth', 0.35);
    yb = interp1(x, y, xl(1));                       % label where it meets the left edge
    if ~isnan(yb) && yb > yl(1) && yb < yl(2)
        text(ax, xl(1) - 4*kmpt, yb, sprintf('%g%c', abs(la), char(176)), 'Color', gcol, ...
             'FontSize', 8, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
             'Clipping', 'off');
    end
end

% ---------- stations ----------
nsta = size(S.stations, 1);
if nsta > 0
    [x, y] = wa_albers(S.stations(:,1), S.stations(:,2));
    plot(ax, x, y, '^', 'MarkerSize', 4.5, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w', ...
         'LineWidth', 0.35, 'LineStyle', 'none');
end

% ---------- sections: white halo, black line, ends, distance ticks ----------
X = S.xsec;
nsec = numel(X);
tick = double(S.tick_km);
ticklab = cell(0, 3);
for k = 1:nsec
    s = X{k};
    [x, y] = wa_albers(s.lon(:), s.lat(:));
    line(ax, x, y, 'Color', 'w', 'LineWidth', 3.4);
    line(ax, x, y, 'Color', 'k', 'LineWidth', 2.0);
    plot(ax, x(1), y(1), 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
    plot(ax, x(end), y(end), 's', 'MarkerSize', 6, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
    dist = s.dist(:);
    for dkm = 0:tick:dist(end) - 1e-6
        [~, j] = min(abs(dist - dkm));
        plot(ax, x(j), y(j), '|', 'MarkerSize', 7, 'Color', 'k', 'LineWidth', 1.2);
        ticklab(end+1, :) = {x(j), y(j) + 3*kmpt, sprintf('%d', round(dkm))}; %#ok<AGROW>
    end
    ticklab(end+1, :) = {x(1) + 5*kmpt, y(1) + 5*kmpt, ['#' s.label]}; %#ok<AGROW>
end

% ---------- unit names at the GSWA map positions ----------
Lb = S.labels;
if isfield(Lb, 'text') && ~isempty(Lb.text)
    if ischar(Lb.text), Lb.text = {Lb.text}; end
    [lx, ly] = wa_albers(Lb.lon(:), Lb.lat(:));
    off = 0.9 * kmpt;                                % halo: white copies around the text
    dirs = [1 0; -1 0; 0 1; 0 -1; 0.7 0.7; -0.7 0.7; 0.7 -0.7; -0.7 -0.7];
    for k = 1:numel(Lb.text)
        str = strsplit(Lb.text{k}, char(10));
        wt = 'normal'; if Lb.bold(k), wt = 'bold'; end
        args = {'FontName', fam, 'FontSize', Lb.size(k), 'FontWeight', wt, ...
                'Rotation', Lb.angle(k), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'Clipping', 'on'};
        for q = 1:size(dirs, 1)
            text(ax, lx(k) + off*dirs(q,1), ly(k) + off*dirs(q,2), str, 'Color', [1 1 1], args{:});
        end
        text(ax, lx(k), ly(k), str, 'Color', Lb.rgb(k,:), args{:});
    end
end

% ---------- coast and WA outline on top ----------
for k = 1:numel(S.coast)
    xy = S.coast{k};
    [x, y] = wa_albers(xy(:,1), xy(:,2));
    line(ax, x, y, 'Color', 'k', 'LineWidth', 0.8);
end
for k = 1:numel(S.wa)
    xy = S.wa{k};
    [x, y] = wa_albers(xy(:,1), xy(:,2));
    line(ax, x, y, 'Color', 'k', 'LineWidth', 0.8);
end
if ~isempty(F.xy) && size(F.xy, 1) > 2
    line(ax, fx, fy, 'Color', 'k', 'LineWidth', 1.3);
end

% ---------- section tick labels and names (above everything on the map) ----------
for k = 1:size(ticklab, 1)
    if ticklab{k,3}(1) == '#'
        text(ax, ticklab{k,1}, ticklab{k,2}, ticklab{k,3}(2:end), 'FontSize', 9, ...
             'FontWeight', 'bold', 'Color', [0.545 0 0], 'HorizontalAlignment', 'left', ...
             'VerticalAlignment', 'bottom', 'Clipping', 'on');
    else
        text(ax, ticklab{k,1}, ticklab{k,2}, ticklab{k,3}, 'FontSize', 5.5, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'BackgroundColor', 'w', 'Margin', 0.5, 'Clipping', 'on');
    end
end

% ---------- crustal-boundary key (lower left) ----------
h1 = plot(ax, NaN, NaN, 'k-',  'LineWidth', 1.1);
h2 = plot(ax, NaN, NaN, 'k-',  'LineWidth', 0.5);
h3 = plot(ax, NaN, NaN, 'k--', 'LineWidth', 0.5);
lg = legend(ax, [h1 h2 h3], {'lithospheric boundary', 'crustal boundary', 'low confidence'}, ...
            'Location', 'southwest', 'FontSize', 7, 'AutoUpdate', 'off');
title(lg, 'Major crustal boundaries (GSWA)');
set(lg.Title, 'FontWeight', 'normal', 'FontSize', 7);

title(ax, sprintf('Cross-section locations (%d lines, ticks every %g km; %d stations)', ...
                  nsec, tick, nsta), 'FontSize', 11, 'FontWeight', 'normal');

% ---------- the map's own legend (rock type x age chart), top left ----------
draw_map_legend(fig, pos, S.legend, fam);

% ---------- write ----------
save_fig(fig, outfile);
end


% =====================================================================================
function draw_map_legend(fig, mappos, G, fam)
% Vector legend chart (PDF points as data units) on a white panel 3.2 in wide,
% anchored at the map's top-left corner. Text is fitted to the original lengths.
W = double(G.width);  H = double(G.height);
pad = 4;  top = 13;  width_in = 3.2;
k = width_in / (W + 2*pad);                          % inches per PDF point
h_in = k * (H + pad + top);
x0 = mappos(1) + 0.01*mappos(3);
y0 = mappos(2) + 0.99*mappos(4) - h_in;
ia = axes('Parent', fig, 'Units', 'inches', 'Position', [x0 y0 width_in h_in]);
hold(ia, 'on');
set(ia, 'XLim', [-pad W+pad], 'YLim', [-pad H+top], 'DataAspectRatio', [1 1 1], ...
    'XTick', [], 'YTick', [], 'Box', 'on', 'Color', 'w', 'XColor', [0.6 0.6 0.6], ...
    'YColor', [0.6 0.6 0.6], 'LineWidth', 0.6, 'SortMethod', 'childorder');
pt = k * 72;                                         % 1 PDF point -> points on the figure
for q = 1:numel(G.pts)
    xy = G.pts{q};
    fc = G.fill{q};  ec = G.stroke{q};
    if isempty(ec), ecol = 'none'; lw = 0.5; else, ecol = ec; lw = max(G.lw(q)*pt, 0.3); end
    if G.closed(q) || ~isempty(fc)
        if isempty(fc), fc = 'none'; end
        patch(ia, xy(:,1), xy(:,2), [0 0 0], 'FaceColor', fc, 'EdgeColor', ecol, 'LineWidth', lw);
    else
        if isempty(ec), ec = [0 0 0]; end
        line(ia, xy(:,1), xy(:,2), 'Color', ec, 'LineWidth', lw);
    end
end
txt = G.text;  if ischar(txt), txt = {txt}; end
for q = 1:numel(txt)
    h = text(ia, G.tx(q,1), G.tx(q,2), txt{q}, 'FontName', fam, 'FontSize', G.tsize(q)*pt, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    e = get(h, 'Extent');                            % measured unrotated, in PDF points
    if e(3) > 0 && abs(G.tlen(q)/e(3) - 1) > 0.03
        set(h, 'FontSize', G.tsize(q)*pt * G.tlen(q)/e(3));
    end
    if G.trot(q), set(h, 'Rotation', 90); end
end
text(ia, -pad + 3, H + top - 2.5, G.title, 'FontSize', 7, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
end


function fam = narrow_font()
% A condensed sans like the GSWA map's Arial Narrow when installed.
have = listfonts;
fam = 'Helvetica';
for c = {'Arial Narrow', 'Liberation Sans Narrow', 'Nimbus Sans Narrow', 'Arial'}
    if any(strcmpi(have, c{1})), fam = c{1}; return; end
end
end


function save_fig(fig, outfile)
if ~isempty(which('exportgraphics'))
    exportgraphics(fig, [outfile '.png'], 'Resolution', 150);
    exportgraphics(fig, [outfile '.pdf'], 'ContentType', 'vector');
else
    print(fig, [outfile '.png'], '-dpng', '-r150');
    print(fig, [outfile '.pdf'], '-dpdf', '-painters');
end
fprintf('Saved: %s.png (+ .pdf)\n', outfile);
end
