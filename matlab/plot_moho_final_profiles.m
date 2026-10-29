% plot_moho_final_profiles.m
%
% Final merged Moho on the GSWA Albers basemap, with the CCP profile transects
% overlaid and 100 km tick marks along each (0 km = the profile's defined start),
% so the map ties to the cross-sections' 100 km x-axis.
%
% Straight great-circle lines come from their endpoints. WEST is a picked
% polyline -> set WEST_MAT to latlon.west.basins.mat (vars lat/lon) to include it.

addpath(genpath('/data/25/yuan/mfiles'))
addpath('/data/25/yuan/github/wa-geo-plotting/matlab')

CCP_DIR  = '/workspace/CCP/CCP.WA.Array.2027.phase.3';
OUT_DIR  = fullfile(CCP_DIR,'Moho.picked');
PLOTDATA = '/data/25/yuan/github/wa-geo-plotting/wa_plotdata.mat';
FINAL    = fullfile(OUT_DIR,'moho_ccp_refined.final.mat');
WEST_MAT = '/workspace/CCP/CCP.WA.Array.2027.phase.3/latlon.west.basins.mat';  % WEST (Darling Fault / Perth Basin) polyline, vars lat/lon; '' skips
TICK_KM  = 100;  % tick spacing along each profile

%% ── Final Moho ──────────────────────────────────────────────────────────────
R = load(FINAL);  m = R.moho_refined;
LAT = m.lat2;  LON = m.lon2;  z = m.z_moho;  z(~m.stn_mask) = NaN;
S = load(PLOTDATA);
[GX,GY] = wa_albers(LON, LAT);

figure('Color','w','Position',[60 60 820 1000]);
ax = gca;
wa_basemap_axes(ax, S, 'Units', false);  hold(ax,'on');
h = pcolor(ax, GX, GY, z);  set(h,'EdgeColor','none','FaceAlpha',0.9);
colormap(ax, flipud(parula(10)));  clim(ax, [30 55]);
cb = colorbar(ax);  cb.Label.String = 'Moho depth (km)';  cb.TickDirection='out';
uistack(h,'bottom');
title(ax, 'Final merged Moho with CCP profiles (100 km ticks)');

%% ── Straight profiles (0 km = first point) ─────────────────────────────────
PROF = {
 'SENW1' -33.7464 121.7458 -25.8040 113.0105
 'SENW2' -33.6332 123.7514 -24.2201 113.4785
 'SENW3' -31.5741 124.0411 -23.3602 114.4812
 '116.5E' -35.0000 116.5000 -22.5000 116.5000
 '117.5E' -35.0000 117.5000 -23.0000 118.0000
 'NS120' -33.8142 120.9881 -22.6814 118.6706
 'WE1'  -24.8785 113.4116 -22.9529 120.1859
 'WE2'  -27.0259 113.9018 -24.4464 119.8740
 'WE3'  -29.1303 114.8823 -26.2339 119.9185
 'WE4'  -31.2120 115.6177 -28.1799 123.6845
 'WE5'  -33.9500 115.0606 -32.4339 124.1302
};
for k = 1:size(PROF,1)
    draw_profile(ax, PROF{k,1}, [PROF{k,2} PROF{k,3}], [PROF{k,4} PROF{k,5}], [], TICK_KM);
end

%% ── WEST polyline (optional) ───────────────────────────────────────────────
if ~isempty(WEST_MAT) && isfile(WEST_MAT)
    draw_profile(ax, 'WEST', [], [], read_west(WEST_MAT), TICK_KM, [0.10 0.20 0.85 0.40], 5.6);
else
    fprintf('WEST skipped (set WEST_MAT to latlon.west.basins.mat to include it)\n');
end

exportgraphics(gcf, fullfile(OUT_DIR,'fig.moho.final.profiles.png'),'Resolution',150);
exportgraphics(gcf, fullfile(OUT_DIR,'fig.moho.final.profiles.pdf'),'ContentType','vector');
fprintf('Saved -> %s\n', fullfile(OUT_DIR,'fig.moho.final.profiles.png'));

%% ── helpers ─────────────────────────────────────────────────────────────────
function draw_profile(ax, lab, p1, p2, poly, tick_km, col, lw)
    if nargin<7||isempty(col), col=[0 0 0]; end     % default solid black
    if nargin<8||isempty(lw),  lw=1.6;     end
    if isempty(poly)                               % straight great circle
        n = 2000;
        la = linspace(p1(1),p2(1),n)';  lo = linspace(p1(2),p2(2),n)';
    else                                           % densify a polyline
        la = []; lo = [];
        for s = 1:size(poly,1)-1
            la = [la; linspace(poly(s,1),poly(s+1,1),400)'];
            lo = [lo; linspace(poly(s,2),poly(s+1,2),400)'];
        end
    end
    d = [0; cumsum(hav(la(1:end-1),lo(1:end-1),la(2:end),lo(2:end)))];
    [X,Y] = wa_albers(lo, la);
    hh = plot(ax, X, Y, '-', 'Color',col(1:3), 'LineWidth',lw);
    if numel(col)>=4, try, hh.Color=col(1:4); catch, hh.Color=col(1:3)+(1-col(1:3))*(1-col(4)); end, end
    % 100 km ticks
    for dd = 0:tick_km:d(end)
        [~,ix] = min(abs(d-dd));
        plot(ax, X(ix), Y(ix), 'o', 'MarkerSize',3.2, ...
            'MarkerFaceColor','w','MarkerEdgeColor','k','LineWidth',0.6);
    end
    % label at the 0 km end
    text(ax, X(1), Y(1), [' ' lab], 'FontSize',8,'FontWeight','bold', ...
        'Color','k','BackgroundColor',[1 1 1 0.6],'Margin',0.5, ...
        'HorizontalAlignment','left','VerticalAlignment','middle');
end
function km = hav(la1,lo1,la2,lo2)
    R=6371; dla=deg2rad(la2-la1); dlo=deg2rad(lo2-lo1);
    a=sin(dla/2).^2 + cosd(la1).*cosd(la2).*sin(dlo/2).^2;
    km=R*2.*atan2(sqrt(a),sqrt(1-a));
end
function ll = read_west(f)
% [lat lon] Nx2 from a picked-polyline .mat, whatever the fields are called.
    W=load(f); fn=fieldnames(W); la=[]; lo=[];
    for c={'lat','latitude','la','y'}, if isfield(W,c{1}), la=W.(c{1})(:); break; end, end
    for c={'lon','long','longitude','lo','x'}, if isfield(W,c{1}), lo=W.(c{1})(:); break; end, end
    if isempty(la)||isempty(lo)
        for k=1:numel(fn), v=W.(fn{k});
            if isnumeric(v)&&isvector(v)&&numel(v)>1
                m=median(double(v(:)),'omitnan');
                if m<0&&m>-60&&isempty(la), la=v(:); end
                if m>90&&m<180&&isempty(lo), lo=v(:); end
            end
        end
    end
    if isempty(la)||isempty(lo)
        for k=1:numel(fn), v=W.(fn{k});
            if isnumeric(v)&&ndims(v)==2&&min(size(v))==2&&max(size(v))>2
                if size(v,2)~=2, v=v.'; end
                if median(v(:,1),'omitnan')<0, la=v(:,1); lo=v(:,2); else, la=v(:,2); lo=v(:,1); end
                break
            end
        end
    end
    if isempty(la)||isempty(lo)
        error('read_west: no lat/lon found in %s (fields: %s)', f, strjoin(fn,', '));
    end
    ll=[la(:) lo(:)];
    n0=size(ll,1);
    if n0>60                      % decimate: few joints -> a thick alpha stroke stays see-through
        step=ceil(n0/40); keep=unique([1:step:n0, n0]); ll=ll(keep,:);
    end
    fprintf('WEST profile: %d points (from %d) in %s\n', size(ll,1), n0, f);
end
