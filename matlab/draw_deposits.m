function h = draw_deposits(ax, spec, szrange, gamma, projfn)
%DRAW_DEPOSITS  Shared mineral-deposit overlay for ALL WA map plots (MATLAB side).
%   Markers are sized by CONTAINED METAL (kg): rank log10(size), normalise 0..1,
%   raise to ^GAMMA, map to the marker-AREA range. GAMMA>1 keeps small/medium
%   deposits small so the giants stand out. Use the SAME datasets/sizing/colours
%   on every map (Moho depth / amplitude / error / xi-avg / Viso / avg-Vsv) so
%   they stay consistent -- this is the one copy of the recipe.
%
%   draw_deposits(ax, spec)
%   draw_deposits(ax, spec, szrange, gamma, projfn)
%     spec    = {matfile, var, marker, colour, filled}   (see the config block below)
%     szrange = marker area range (pts^2), default [6 520]
%     gamma   = >1 lets the giants stand out, default 2.0
%     projfn  = handle lon,lat -> (x,y) for the map's axes coords. Default identity
%               (plot in lon/lat). For a GSWA-Albers map pass @wa_albers.
%
%   Config block to paste into a map script (toggle each layer independently):
%     MINDIR  = '/data/25/yuan/github/wa-geo-plotting/matlab';
%     GOLD    = {fullfile(MINDIR,'gold_deposits.mat'),    'gold',    'o', [1.00 0.90 0.00], true };
%     COPPER  = {fullfile(MINDIR,'copper_deposits.mat'),  'copper',  '^', [0.85 0.00 0.00], true };
%     LITHIUM = {fullfile(MINDIR,'lithium_deposits.mat'), 'lithium', 'h', [0.60 0.20 0.80], true };
%     NICKEL  = {fullfile(MINDIR,'nickel_deposits.mat'),  'nickel',  'h', [0.20 0.70 0.20], true };
%     % ... after the Moho/Vs field + basemap are drawn into axes ax:
%     if SHOW_GOLD,   draw_deposits(ax, GOLD,   [6 520], 2.0, @wa_albers); end
%
    if nargin < 3 || isempty(szrange), szrange = [6 520]; end
    if nargin < 4 || isempty(gamma),   gamma   = 2.0;     end
    if nargin < 5 || isempty(projfn),  projfn  = @(lo,la) deal(lo,la); end  % default lon/lat
    matfile = spec{1}; var = spec{2}; marker = spec{3}; col = spec{4}; filled = spec{5};
    h = [];
    if isempty(matfile) || ~isfile(matfile)
        fprintf('  (%s: file not found, skipped: %s)\n', var, matfile); return
    end
    Gd = load(matfile); g = Gd.(var);
    v = double(g.size(:)); v(v <= 0 | ~isfinite(v)) = NaN;
    [x, y] = projfn(double(g.lon(:)), double(g.lat(:)));   % lon/lat -> axes coords
    lv = log10(v); lo = min(lv); hi = max(lv);
    frac = (lv - lo) ./ max(hi - lo, eps);
    sz = szrange(1) + (szrange(2) - szrange(1)) .* frac.^gamma;
    ok = isfinite(x) & isfinite(y) & isfinite(sz);
    if filled
        h = scatter(ax, x(ok), y(ok), sz(ok), marker, 'MarkerFaceColor', col, ...
            'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.7, 'LineWidth', 0.3, 'Tag', var);
    else
        h = scatter(ax, x(ok), y(ok), sz(ok), marker, 'MarkerEdgeColor', col, ...
            'MarkerFaceColor', 'none', 'LineWidth', 0.8, 'Tag', var);
    end
    fprintf('%s: %d deposits (size %.3g-%.3g)\n', var, nnz(ok), min(v), max(v));
end
