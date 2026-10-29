function draw_greenstones(ax, matfile, col, alpha, projfn)
%DRAW_GREENSTONES  Fill greenstone-belt rings under the deposit markers.
%   Companion to draw_deposits. Loads a struct 'greenstones' holding [lon lat]
%   with NaN rows separating rings (SW Yilgarn for now), and fills each ring.
%
%   draw_greenstones(ax, matfile)
%   draw_greenstones(ax, matfile, col, alpha, projfn)
%     col    = fill colour, default deep green [0 0.35 0]
%     alpha  = face alpha, default 0.5
%     projfn = handle lon,lat -> (x,y), default identity (lon/lat); @wa_albers for Albers
%
    if nargin < 3 || isempty(col),    col    = [0 0.35 0]; end
    if nargin < 4 || isempty(alpha),  alpha  = 0.5;        end
    if nargin < 5 || isempty(projfn), projfn = @(lo,la) deal(lo,la); end
    if isempty(matfile) || ~isfile(matfile)
        fprintf('  (greenstones: file not found, skipped: %s)\n', matfile); return
    end
    Gd = load(matfile); g = Gd.greenstones;
    if isstruct(g) && isfield(g,'greenstones'), g = g.greenstones; end
    ll = double(g);
    if size(ll,2) ~= 2 && size(ll,1) == 2, ll = ll.'; end
    brk = [0; find(~isfinite(ll(:,1))); size(ll,1)+1];   % NaN rows split rings
    n = 0;
    for k = 1:numel(brk)-1
        ring = ll(brk(k)+1 : brk(k+1)-1, :);
        if size(ring,1) < 3, continue; end
        [x, y] = projfn(ring(:,1), ring(:,2));
        patch(ax, x, y, col, 'FaceAlpha', alpha, 'EdgeColor', 'none', 'Tag', 'greenstones');
        n = n + 1;
    end
    fprintf('greenstones: %d rings\n', n);
end
