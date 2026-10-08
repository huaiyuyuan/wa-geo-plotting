function wa_plot_stack(matfile, field, groups, outdir)
%WA_PLOT_STACK  All cross-sections of one field on one page per orientation - the same
%   figures as plot_xsection.py --stack FIELD (split into north-south and west-east pages).
%
%   wa_plot_stack('wa_plotdata.mat', 'dvsv')            -> xsection_stack.dvsv.ns / .ew
%   wa_plot_stack('wa_plotdata.mat', 'vsv', 'ns')       -> north-south page only
%   wa_plot_stack('wa_plotdata.mat', 'xi', 'all', 'figs')  one page, all sections, in figs/
%
%   field  : 'vsv' (absolute), 'dvsv' (% vs model mean at each depth), and for xi models
%            'xi', 'dxi'
%   groups : 'both' (default: ns + ew pages) | 'ns' | 'ew' | 'all'
%   Both pages share one km-per-inch (the longest section spans 9.5 in), VE 3.
%   Each row: domain strip in the 2022 tectonic-map colours with crustal-boundary
%   down-arrows and names, the section with the AR23 Moho (dashed) and grey veil below,
%   and its own colour bar.

if nargin < 2 || isempty(field),  field = 'dvsv'; end
if nargin < 3 || isempty(groups), groups = 'both'; end
if nargin < 4 || isempty(outdir), outdir = fileparts(matfile); end
if isempty(outdir), outdir = '.'; end
S = load(matfile);
if ~isfield(S.fields, field)
    error('wa_plot_stack: field "%s" not in %s (have: %s)', field, matfile, ...
          strjoin(fieldnames(S.fields), ', '));
end
if ~isfield(S.xsec{1}, field)
    error('wa_plot_stack: "%s" not exported for this model (xi fields need an xi model)', field);
end
switch groups
    case 'both', todo = {'ns', 'ew'};
    case 'all',  todo = {'all'};
    otherwise,   todo = {groups};
end
for g = 1:numel(todo)
    key = todo{g};
    if strcmp(key, 'all')
        idx = 1:numel(S.xsec);  gname = '';  tag = '';
    else
        idx = S.groups.(key);  idx = idx(:)';
        gname = struct('ns', 'north-south ', 'ew', 'west-east ');  gname = gname.(key);
        tag = ['.' key];
    end
    if isempty(idx), continue; end
    one_page(S, field, idx, gname, fullfile(outdir, sprintf('xsection_stack.%s%s', field, tag)));
end
end


% =====================================================================================
function one_page(S, field, idx, gname, outfile)
F = S.fields.(field);
cmap = S.cmaps.(F.cmap);
Ly = S.layout;
z = double(S.z(:));
zbot = z(end);
n = numel(idx);
dists = cellfun(@(s) s.dist(end), S.xsec(idx));
maxd = max(double(S.groups.longest_km), max(dists));     % shared scale across pages
page_w = double(Ly.page_w_in);  ve = double(Ly.ve);
km_per_in = maxd / page_w;
row_h = zbot * ve / km_per_in;
strip_h = 0.17 + 0.28 + 0.55;                             % domain strip + arrows + names
gap = 0.55 + strip_h;
top = 0.9 + strip_h;
fig_h = n*row_h + (n-1)*gap + 1.2 + strip_h;
fig_w = page_w + 1.8;
fig = figure('Units', 'inches', 'Position', [0.5 0.5 fig_w fig_h], 'Color', 'w', ...
             'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
annotation(fig, 'textbox', [0 1 - 0.55/fig_h 1 0.45/fig_h], 'String', ...
    sprintf('%s (%s) %s %s%d sections [depth 0-%g km, page width %.0f km, VE %gx]', ...
            F.title, S.vname, char(8212), gname, n, zbot, maxd, ve), ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontSize', 11, 'FontWeight', 'bold');
left = 0.08 * fig_w;
for i = 1:n
    s = S.xsec{idx(i)};
    dist = double(s.dist(:));
    D = double(s.(field));                                % ndist x nz
    w_in = dist(end) / km_per_in;
    y0 = fig_h - top - i*row_h - (i-1)*gap;
    ax = axes('Parent', fig, 'Units', 'inches', 'Position', [left y0 w_in row_h]);
    hold(ax, 'on');
    im = imagesc(ax, dist, z, D.');
    set(im, 'AlphaData', ~isnan(D.'));
    colormap(ax, cmap);
    clim = double(F.clim(:)');
    if any(isnan(clim))                                   % absolute: per-panel median +- sigma*std
        v = D(isfinite(D));
        clim = median(v) + [-1 1] * F.sigma * std(v, 1);
    end
    caxis(ax, clim);
    set(ax, 'YDir', 'reverse', 'XLim', [0 dist(end)], 'YLim', [0 zbot], 'FontSize', 6, ...
        'XTick', 0:200:dist(end), 'Box', 'on', 'Layer', 'top', 'TickDir', 'out');
    ylabel(ax, 'Depth', 'FontSize', 7);
    if i == n, xlabel(ax, 'Distance (km)', 'FontSize', 8); end
    text(ax, 0.01, 0.90, sprintf('%s [%.0f km]', s.label, dist(end)), 'Units', 'normalized', ...
         'FontSize', 8, 'FontWeight', 'bold', 'VerticalAlignment', 'top', ...
         'BackgroundColor', 'w', 'Margin', 1);
    draw_moho(ax, dist, double(s.moho(:)), zbot);
    cb = colorbar(ax);
    set(cb, 'Units', 'inches', 'Position', [left + w_in + 0.008*fig_w, y0, 0.012*fig_w, row_h], ...
        'FontSize', 6);
    set(ax, 'Position', [left y0 w_in row_h]);            % colorbar must not shrink the panel
    if i == 1, cb.Label.String = F.label; cb.Label.FontSize = 8; end
    draw_strip(fig, s, [left y0 w_in row_h], dist, km_per_in, Ly);
end
save_fig(fig, outfile);
end


function draw_moho(ax, dist, moho, zbot)
% Grey veil (alpha 0.5) from the Moho to the bottom, then the dashed Moho.
ok = isfinite(moho);
if ~any(ok), return; end
m = min(moho, zbot);
d = diff([0; ok; 0]);
st = find(d == 1);  en = find(d == -1) - 1;
for k = 1:numel(st)
    j = st(k):en(k);
    patch(ax, [dist(j); flipud(dist(j))], [m(j); zbot*ones(numel(j), 1)], [0.5 0.5 0.5], ...
          'FaceAlpha', 0.5, 'EdgeColor', 'none');
end
plot(ax, dist, moho, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.9);
end


function draw_strip(fig, s, apos, dist, km_per_in, Ly)
% Domain strip above the panel, boundary down-arrows standing on it, names at 35 deg.
sh = double(Ly.strip_h_in);  sg = double(Ly.strip_gap_in);
spos = [apos(1), apos(2) + apos(4) + sg, apos(3), sh];
sax = axes('Parent', fig, 'Units', 'inches', 'Position', spos);
hold(sax, 'on');
set(sax, 'XLim', [0 dist(end)], 'YLim', [0 1], 'XTick', [], 'YTick', [], 'Box', 'on', ...
    'LineWidth', 0.6, 'Layer', 'top');
text(sax, -0.005, 0.5, 'Domain', 'Units', 'normalized', 'FontSize', 7, ...
     'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
R = s.runs;
if ~isempty(R.d0)
    names = R.name;  if ischar(names), names = {names}; end
    span = R.d1(end) - R.d0(1);
    for k = 1:numel(R.d0)
        patch(sax, [R.d0(k) R.d1(k) R.d1(k) R.d0(k)], [0 0 1 1], R.rgb(k,:), 'EdgeColor', 'none');
    end
    for k = 1:numel(R.d0)                                 % names where they fit
        run_in = (R.d1(k) - R.d0(k)) / span * spos(3);
        for cand = {names{k}, shorten(names{k})}
            if numel(cand{1}) * 6 * 0.55 / 72 < run_in * 0.92
                text(sax, 0.5*(R.d0(k) + R.d1(k)), 0.5, cand{1}, 'FontSize', 6, ...
                     'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                     'BackgroundColor', R.rgb(k,:), 'Margin', 0.6, 'Clipping', 'on');
                break
            end
        end
    end
end
C = s.cross;
if isempty(C.dist), return; end
cn = C.names;  if ischar(cn), cn = {cn}; end
tallest = 0;
for k = 1:numel(C.dist)
    line(sax, [C.dist(k) C.dist(k)], [0 1], 'Color', 'k', 'LineWidth', 0.6);
    if C.crustal(k), lw = 0.9; hd = 6; len = 12; else, lw = 1.6; hd = 9; len = 17; end
    tallest = max(tallest, len);
    figW = fig.Position(3);  figH = fig.Position(4);
    xf = (spos(1) + C.dist(k) / dist(end) * spos(3)) / figW;
    yt = (spos(2) + spos(4)) / figH;
    annotation(fig, 'arrow', [xf xf], [yt + len/72/figH, yt], 'LineWidth', lw, ...
               'HeadLength', hd, 'HeadWidth', hd, 'Color', 'k');
end
% one label per group of crossings closer than 0.35 in, each name once
gd = [];  gn = {};
for k = 1:numel(C.dist)
    if isempty(cn{k}), continue; end
    nm = strtrim(strsplit(cn{k}, ' / '));
    if ~isempty(gd) && (C.dist(k) - gd(end)) / km_per_in < 0.35
        gn{end} = unique([gn{end}, nm], 'stable');
    else
        gd(end+1) = C.dist(k);  gn{end+1} = nm; %#ok<AGROW>
    end
end
for k = 1:numel(gd)
    text(sax, gd(k), 1 + ((tallest + 2) / 72) / sh, strjoin(gn{k}, ' / '), 'Rotation', 35, ...
         'FontSize', 6, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
         'Clipping', 'off');
end
end


function out = shorten(name)
out = name;
for suf = {' Superterrane', ' Terrane', ' Province', ' Domain', ' Zone'}
    out = strrep(out, suf{1}, '');
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
