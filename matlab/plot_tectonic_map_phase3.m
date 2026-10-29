% plot_tectonic_map_phase3.m
%
% GSWA 2022 simplified tectonic map, cropped to the WA Array Phase 1-3 extent
% (same Albers projection/extent as the Moho maps). Filled units, bold deep-blue
% coast, crustal boundaries, 2-deg graticule, CCP profiles with 100 km ticks,
% and seismic stations as open black triangles.
%
% Legends:
%   - unit colour blocks split into two age groups (Archean; Proterozoic ->
%     Phanerozoic), placed over the empty ocean (SW/W and S/SE), long names
%     wrapped to two lines;
%   - a map key at the NE for profiles / crustal boundaries / stations.
%
% Positions are normalized within the map axes (lower-left of each block) so you
% can nudge them freely.

addpath(genpath('/data/25/yuan/mfiles'))
addpath('/data/25/yuan/github/wa-geo-plotting/matlab')

PLOTDATA = '/data/25/yuan/github/wa-geo-plotting/wa_plotdata.mat';
OUT_DIR  = '/workspace/CCP/CCP.WA.Array.2027.phase.3/Moho.picked';
OUTFILE  = fullfile(OUT_DIR, 'fig.tectonic.phase3');
STA_PMAT = '/workspace/CCP/CCP.WA.Array.2027.phase.3/mat/WA.Array.phase.4.BHR.0.0.360.fl.0.1.fh.0.5.dmy.10.s.0.04.3.dmx.10.rot.0.imove.0.dzi.0.5.vmodel.WA.Array.rf.0.Mantle.vpvs.1.84.vcorr.0.Z.80.n-th.0.lowpass.0.f0.3.0.halfweight.3.pvh.6.P.mat';
GRAT_DEG = 2;        % graticule spacing (deg)
TICK_KM  = 100;      % profile tick spacing (km)
% Frame extent [lon0 lon1 lat0 lat1]. [] = the phases 1-3 crop stored in the base;
% set EXTENT = [112 129.6 -35.8 -13.2] for the WHOLE-WA map (same base data, AF
% profiles and stations - they just sit in the SW of the state). On the whole-WA
% frame GRAT_DEG=4 reads better, and the legends below may need a nudge.
if ~exist('EXTENT','var'), EXTENT = []; end
WEST_MAT = '/workspace/CCP/CCP.WA.Array.2027.phase.3/latlon.west.basins.mat';  % WEST (Darling Fault / Perth Basin) polyline, vars lat/lon; '' skips

% ── Optional overlays: flip each on/off to plot the map with or without them ──
SHOW_PROFILES   = true;    % seismic CCP profiles + 100 km ticks
SHOW_GREENSTONE = false;   % greenstone belts (SW Yilgarn only for now - whole-Yilgarn file pending)
SHOW_GOLD       = true;    % gold    deposits - open gold circles
SHOW_COPPER     = false;   % copper  deposits - red triangles
SHOW_LITHIUM    = false;   % lithium deposits - purple hexagons
SHOW_NICKEL     = false;   % nickel  deposits - green hexagons

MINDIR = '/data/25/yuan/github/wa-geo-plotting/matlab';   % where the deposit .mat files live
GREENSTONE_MAT = fullfile(MINDIR,'greenstones_sw_yilgarn.mat');   % belts (lon/lat, NaN-sep)
GREENSTONE_COL = [0 0.35 0];             % deep green fill
% deposit layers: {matfile, struct-var, marker, colour, filled?} - all sized (log) by contained metal
DEP_SZ    = [6 520];                     % marker area range (points^2): min .. max
DEP_GAMMA = 2.0;                         % >1 shrinks the small/medium, lets the giants stand out
GOLD    = {fullfile(MINDIR,'gold_deposits.mat'),    'gold',    'o', [1.00 0.90 0.00], true };
COPPER  = {fullfile(MINDIR,'copper_deposits.mat'),  'copper',  '^', [0.85 0.00 0.00], true };
LITHIUM = {fullfile(MINDIR,'lithium_deposits.mat'), 'lithium', 'h', [0.60 0.20 0.80], true };
NICKEL  = {fullfile(MINDIR,'nickel_deposits.mat'),  'nickel',  'h', [0.20 0.70 0.20], true };
PROF_COLOR    = [0.10 0.20 0.85 0.40];   % seismic profile: blue + alpha (RGBA); faint
PROF_LW       = 5.6;                      % profile line width (thick but transparent)

% Unit legend: a single BOXLESS list down the left, Archean (top) -> Phanerozoic
% with era sub-headers, top at LEG_TOP_FRAC of the map height, flowing down.
LEG_TOP_FRAC = 0.74;  % top of the list as a fraction of map height (fills the SW white)
LEG_PAD      = 0.10;  % inches in from the left map edge
LEG_W        = 2.7;   % legend column width (inches)
LEG_FS       = 9;     % legend text size (unit names; era headers are +1)
ANCH_KEY     = 'ne';  % map key corner
LEG_ALPHA    = 1.0;   % map-key background opacity (1 = fully opaque white)

% ── Age lookup (Ma, approximate midpoints) — sort oldest -> youngest ─────────
AGE = {
 'Narryer Terrane'                                3680
 'Pilbara Craton'                                 3450
 'Youanmi Terrane'                                2950
 'South West Terrane'                             2750
 'Fortescue Basin'                                2720
 'Eastern Goldfields Superterrane'                2700
 'Hamersley Basin'                                2500
 'Turee Creek Basin'                              2350
 'Bryah Basin'                                    2030
 'Yerrida Basin'                                  2000
 'Earaheedy Basin'                                1960
 'Yarlarweelor Gneiss Complex, Narryer Terrane'   1950
 'Gascoyne Province'                              1900
 'Capricorn Orogen'                               1850
 'Ashburton Basin'                                1800
 'Biranup Zone'                                   1700
 'Bresnahan Basin'                                1620
 'Albany-Fraser Orogen, Northern Foreland'        1600
 'Edmund Basin'                                   1550
 'Albany-Fraser Orogen'                           1330
 'Fraser Zone'                                    1300
 'Collier Basin'                                  1200
 'Nornalup Zone'                                  1180
 'Pinjarra Orogen'                                1050
 'Officer Basin (Phase 1)'                         780
 'Officer Basin (Phase 2)'                         700
 'Southern Carnarvon Basin'                        420
 'Northern Carnarvon Basin'                        350
 'Perth Basin'                                     280
 'Eucla Basin'                                      65
};

% ── Major units labelled ON the map (and removed from the crowded legend) ────
% The big, easy-to-point-at units get a text label at the centroid of their
% largest in-view polygon; only the small units stay in the era legend.
%   col 1: label text (char, or cellstr {'line1','line2'} for two lines)
%   col 2: GSWA unit name(s) whose polygon centroid anchors the label ({} = none)
%   col 3: manual [lon lat] override, or [] to use the computed centroid
%   col 4: font size override, or [] for the default LBL_FS
%   col 5: rotation in degrees (CCW), or [] / 0 for horizontal
SHOW_UNITLABELS = true;
LBL_FS = 9;          % default on-map label font size
MAJORLBL = {
  'YILGARN'           {}                                                        [118 -28]   16   0
  'Pilbara Craton'    {'Pilbara Craton'}                                        []          []   0
  'Narryer'           {'Narryer Terrane'}                                       []          []   0
  'Youanmi'           {'Youanmi Terrane'}                                       []          []   0
  'E. Goldfields'     {'Eastern Goldfields Superterrane'}                       []          []   0
  'SW Terrane'        {'South West Terrane'}                                    []          []   0
  'Fortescue Basin'   {'Fortescue Basin'}                                       []          8    0
  'Hamersley Basin'   {'Hamersley Basin'}                                       []          8    0
  'Gascoyne Province' {'Gascoyne Province'}                                     [116.2 -25.4]  8   -30
  'Perth Basin'       {'Perth Basin'}                                           [115.9 -31.4]  8   -80
  'Carnarvon Basin'   {'Southern Carnarvon Basin','Northern Carnarvon Basin'}   [113.9 -24.1]  8   -78
  'Officer Basin'     {'Officer Basin (Phase 1)','Officer Basin (Phase 2)'}     [124.3 -26.2]  []   0
};

% ── Province outlines: ONE contour dissolved from its member units ───────────
% GSWA has no single "West Australian Craton" / "Capricorn Orogen" polygon, so
% we union the member units with polyshape at run time and clip to the frame.
% Edit the member lists to match unique(string(S.units.name)) — any unit name
% not present is just skipped (may leave a small notch in the outline).
SHOW_PROVINCES = false;   % dropped: dissolved outline was messy — re-enable + fix member lists later
PROV = {
% label                      member unit names                                        colour           lw    style
 'West Australian Craton', {'Pilbara Craton','Fortescue Basin','Hamersley Basin', ...
     'Turee Creek Basin','Ashburton Basin','Bryah Basin','Yerrida Basin','Earaheedy Basin', ...
     'Bresnahan Basin','Edmund Basin','Collier Basin','Gascoyne Province','Capricorn Orogen', ...
     'Yarlarweelor Gneiss Complex, Narryer Terrane','Narryer Terrane','Youanmi Terrane', ...
     'Eastern Goldfields Superterrane','South West Terrane'},               [0 0 0],        2.6, '-'
 'Capricorn Orogen',       {'Gascoyne Province','Capricorn Orogen','Bryah Basin', ...
     'Yerrida Basin','Earaheedy Basin','Ashburton Basin','Turee Creek Basin','Bresnahan Basin', ...
     'Edmund Basin','Collier Basin','Yarlarweelor Gneiss Complex, Narryer Terrane'}, [0.45 0.05 0.55], 1.8, '--'
};

S   = load(PLOTDATA);
fam = narrow_font();
if isempty(EXTENT), ext = double(S.extent(:)'); else, ext = double(EXTENT(:)'); end
if ~isequal(ext, double(S.extent(:)')), OUTFILE = [OUTFILE '.WA']; end   % distinct file

%% ── Figure + full map axes ─────────────────────────────────────────────────
FIG = 13;
t = linspace(0,1,120);
blon = [ext(1)+t*(ext(2)-ext(1)), ext(2)*ones(1,120), ext(2)-t*(ext(2)-ext(1)), ext(1)*ones(1,120)];
blat = [ext(3)*ones(1,120), ext(3)+t*(ext(4)-ext(3)), ext(4)*ones(1,120), ext(4)-t*(ext(4)-ext(3))];
[bx,by] = wa_albers(blon, blat);
r = (max(by)-min(by))/(max(bx)-min(bx));
w = min(0.88*FIG, 0.90*FIG/r);  h = w*r;
mapx0 = 0.06*FIG + (0.88*FIG-w)/2;
mapy0 = 0.05*FIG + (0.90*FIG-h)/2;
pos = [mapx0 mapy0 w h];

fig = figure('Units','inches','Position',[0.5 0.5 FIG FIG],'Color','w', ...
             'PaperPositionMode','auto','InvertHardcopy','off');
ax = axes('Parent',fig,'Units','inches','Position',pos);

%% ── Tectonic basemap (units + boundaries only; own coast/graticule/stations) ─
wa_basemap_axes(ax, S, 'Units',true, 'UnitAlpha',1, 'Boundaries',true, ...
    'Coast',false, 'Graticule',false, 'GraticuleLabels',false, 'Labels',false, 'Stations',false);
hold(ax,'on');
xl = get(ax,'XLim');  yl = get(ax,'YLim');

%% ── Greenstone belts (SW Yilgarn), filled deep green over the units ─────────
% From the GSWA 'Greenstones' category, reprojected to lon/lat. Drawn on top of
% the tectonic units but under the graticule / coast / profiles / stations, with
% holes (granite inliers) cut out via polyshape.
if SHOW_GREENSTONE && ~isempty(GREENSTONE_MAT) && isfile(GREENSTONE_MAT)
    Gs = load(GREENSTONE_MAT);
    [gx, gy] = wa_albers(Gs.greenstones(:,1), Gs.greenstones(:,2));
    if exist('polyshape','file')
        warning('off','MATLAB:polyshape:repairedBySimplify');
        warning('off','MATLAB:polyshape:boundary3Points');
        pg = polyshape(gx, gy, 'Simplify', true);
        plot(ax, pg, 'FaceColor', GREENSTONE_COL, 'FaceAlpha', 1, ...
             'EdgeColor', 'none', 'Tag', 'greenstone');
        fprintf('greenstones: %d regions <- %s\n', pg.NumRegions, GREENSTONE_MAT);
    else                                       % no polyshape: fill rings (holes not cut)
        patch(ax, gx, gy, GREENSTONE_COL, 'EdgeColor', 'none', 'Tag', 'greenstone');
        fprintf('greenstones (patch, holes not cut) <- %s\n', GREENSTONE_MAT);
    end
end

%% ── 2-degree graticule + labels ────────────────────────────────────────────
gcol = [133 140 140]/255;
glon = ceil(ext(1)/GRAT_DEG)*GRAT_DEG : GRAT_DEG : floor(ext(2)/GRAT_DEG)*GRAT_DEG;
glat = ceil(ext(3)/GRAT_DEG)*GRAT_DEG : GRAT_DEG : floor(ext(4)/GRAT_DEG)*GRAT_DEG;
% extend beyond the extent so the lines overshoot; the axes clip them to the frame
latd = linspace(ext(3)-2,ext(4)+2,300);  lond = linspace(ext(1)-2,ext(2)+2,300);
offx = 0.010*diff(xl);  offy = 0.010*diff(yl);
for lo = glon                       % meridians; label where they cross the SOUTH frame edge
    [x,y]=wa_albers(lo*ones(size(latd)),latd); line(ax,x,y,'Color',gcol,'LineWidth',0.3);
    xb=interp1(y,x,yl(1),'linear',NaN);
    if ~isnan(xb)&&xb>xl(1)&&xb<xl(2)
        text(ax,xb,yl(1)+offy,sprintf('%g%cE',lo,char(176)),'Color',[.2 .2 .2],'FontSize',8,'FontWeight','bold', ...
            'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
end
for la = glat                       % parallels; label where they cross the EAST frame edge
    [x,y]=wa_albers(lond,la*ones(size(lond))); line(ax,x,y,'Color',gcol,'LineWidth',0.3);
    yr=interp1(x,y,xl(2),'linear',NaN);
    if ~isnan(yr)&&yr>yl(1)&&yr<yl(2)
        text(ax,xl(2)-offx,yr,sprintf('%g%cS',abs(la),char(176)),'Color',[.2 .2 .2],'FontSize',8,'FontWeight','bold', ...
            'HorizontalAlignment','right','VerticalAlignment','middle');
    end
end

%% ── Coast + WA outline in bold deep blue ───────────────────────────────────
navy = [0 0 0.6];
for fld = {'coast','wa'}
    if isfield(S,fld{1})
        C = S.(fld{1});
        for k=1:numel(C), [x,y]=wa_albers(C{k}(:,1),C{k}(:,2)); line(ax,x,y,'Color',navy,'LineWidth',1.8); end
    end
end

%% ── Province outlines (dissolved unit unions, clipped to the map frame) ─────
if SHOW_PROVINCES && exist('polyshape','file')
    for p=1:size(PROV,1)
        draw_province_contour(ax, S, PROV{p,2}, PROV{p,3}, PROV{p,4}, PROV{p,5}, bx, by, PROV{p,1});
    end
end

%% ── CCP profiles + 100 km ticks (single source: ccp_profiles.txt) ──────────
% Read the SAME profile definition the Python side (tools/mat_to_profiles.py)
% and the CCP landscape (plot_ccp_landscape_WA.m) use, so AF / WEST / any line
% added or regrouped there shows up here automatically and never drifts.
BASE = '/workspace/CCP/CCP.WA.Array.2027.phase.3';
if ~exist('PROFDEF','var') || isempty(PROFDEF) || ~isfile(PROFDEF)
    PROFDEF = fullfile(BASE, 'ccp_profiles.txt');
end
if SHOW_PROFILES
    profs = read_all_profiles(PROFDEF, BASE);
    fprintf('Profiles: %d from %s\n', numel(profs), PROFDEF);
    for k = 1:numel(profs)
        pk = profs(k);
        fprintf('  %-8s %s\n', pk.short, pk.kind);
        if strcmp(pk.kind, 'gc')          % straight 2-pt draw -> transparent stroke
            draw_profile(ax, pk.short, pk.p1, pk.p2, [], TICK_KM, PROF_COLOR, PROF_LW);
        else                               % curved pick (AF/WEST): decimated polyline
            draw_profile(ax, pk.short, [], [], pk.poly, TICK_KM, PROF_COLOR, PROF_LW);
        end
    end
end

%% ── Stations as open black triangles ───────────────────────────────────────
slon=[]; slat=[];
if isfield(S,'stations') && ~isempty(S.stations)
    slon=S.stations(:,1); slat=S.stations(:,2);
elseif isfile(STA_PMAT)
    P=load(STA_PMAT); P=P.P;
    if isfield(P,'slon'), slon=P.slon(:); slat=P.slat(:); end
end
if ~isempty(slon)
    [sx,sy]=wa_albers(slon,slat);
    plot(ax,sx,sy,'^','MarkerSize',3.0,'MarkerEdgeColor','k','MarkerFaceColor','k', ...
        'LineWidth',0.4,'LineStyle','none');
    fprintf('Stations plotted: %d\n', numel(slon));
else
    fprintf('No stations found (S.stations absent and P.mat not readable).\n');
end

%% ── Mineral deposits: open/filled markers sized (log) by contained metal ────
if SHOW_GOLD,    draw_deposits(ax, GOLD,    DEP_SZ, DEP_GAMMA); end
if SHOW_COPPER,  draw_deposits(ax, COPPER,  DEP_SZ, DEP_GAMMA); end
if SHOW_LITHIUM, draw_deposits(ax, LITHIUM, DEP_SZ, DEP_GAMMA); end
if SHOW_NICKEL,  draw_deposits(ax, NICKEL,  DEP_SZ, DEP_GAMMA); end

%% ── Major-unit text labels on the map (toggle 'unitlabel') ──────────────────
if SHOW_UNITLABELS, draw_unit_labels(ax, S, MAJORLBL, xl, yl, LBL_FS); end

%% ── Units in extent -> age-sorted legend entries ───────────────────────────
U=S.units; names=U.name; if ischar(names), names={names}; end
vis=false(numel(U.xy),1);
for k=1:numel(U.xy)
    xy=U.xy{k};
    vis(k)=any(xy(:,1)>=ext(1)&xy(:,1)<=ext(2)&xy(:,2)>=ext(3)&xy(:,2)<=ext(4));
end
[unames,~,ic]=unique(string(names(:)),'stable');
urgb=zeros(numel(unames),3); uvis=false(numel(unames),1);
for u=1:numel(unames), sel=find(ic==u); urgb(u,:)=U.rgb(sel(1),:); uvis(u)=any(vis(sel)); end
vis_names=unames(uvis); vis_rgb=urgb(uvis,:);

% Units shown as text ON the map are dropped from the legend (keeps it short).
labeled = string([]);
if SHOW_UNITLABELS
    for i=1:size(MAJORLBL,1), labeled = [labeled, string(MAJORLBL{i,2})]; end %#ok<AGROW>
end
agekeys=string(AGE(:,1)); ageval=cell2mat(AGE(:,2));
leg_names=strings(0); leg_rgb=zeros(0,3); leg_age=[];
for v=1:numel(vis_names)
    if any(strcmpi(vis_names(v), labeled)), continue; end   % labelled on the map
    mm=find(strcmpi(agekeys,vis_names(v)),1);
    if ~isempty(mm), leg_names(end+1)=vis_names(v); leg_rgb(end+1,:)=vis_rgb(v,:); leg_age(end+1)=ageval(mm); end %#ok<AGROW>
end
[~,si]=sort(leg_age,'descend'); leg_names=leg_names(si); leg_rgb=leg_rgb(si,:); leg_age=leg_age(si);
arch = leg_age>=2500;             % Archean
phan = leg_age<541;               % Phanerozoic
prot = ~arch & ~phan;             % Proterozoic

%% ── Boxless era legend down the left margin (Archean -> Phanerozoic) ────────
% The Yarlarweelor Gneiss Complex sits inside the Narryer Terrane but is a
% distinct Proterozoic unit — give it its own name so it is not a second
% "Narryer Terrane" row clashing with the Archean craton label.
leg_disp = leg_names;
leg_disp(leg_disp=="Yarlarweelor Gneiss Complex, Narryer Terrane") = "Yarlarweelor Gneiss";
% build only the non-empty era sections (an empty section would draw a bare header)
eras = {'Archean', arch; 'Proterozoic', prot; 'Phanerozoic basins', phan};
sections = struct('title',{}, 'names',{}, 'rgb',{});
for e=1:size(eras,1)
    sel = eras{e,2};
    if ~any(sel), continue; end
    sections(end+1) = struct('title',eras{e,1}, ...
        'names',{cellstr(leg_disp(sel))}, 'rgb',leg_rgb(sel,:)); %#ok<AGROW>
end
draw_era_legend(fig, pos, sections, LEG_TOP_FRAC, LEG_PAD, LEG_W, LEG_FS);

%% ── NE map key: profiles / boundaries / stations ───────────────────────────
draw_key_ne(fig, pos, ANCH_KEY, LEG_ALPHA, SHOW_PROFILES, PROF_COLOR, PROF_LW);

%% ── Save ────────────────────────────────────────────────────────────────────
if ~SHOW_PROFILES, OUTFILE=[OUTFILE '.noprofiles']; end   % separate file for the clean version
if ~isempty(which('exportgraphics'))
    exportgraphics(fig,[OUTFILE '.png'],'Resolution',150);
    exportgraphics(fig,[OUTFILE '.pdf'],'ContentType','vector');   % editable vector PDF
else
    print(fig,[OUTFILE '.png'],'-dpng','-r150'); print(fig,[OUTFILE '.pdf'],'-dpdf','-painters');
end
fprintf('Saved: %s.png (+ .pdf)\n', OUTFILE);

%% ── helpers ─────────────────────────────────────────────────────────────────
function draw_swatch_legend(fig, mappos, anchor, titlestr, names, rgb, ncol, alpha)
    N=numel(names); if N==0, return; end
    if nargin<7||isempty(ncol), ncol=1; end
    if nargin<8||isempty(alpha), alpha=0.7; end
    rowh=0.235; col_w=2.5; nrow=ceil(N/ncol);
    width_in=ncol*col_w; h_in=(nrow+1.4)*rowh;
    [x0,y0]=anchor_pos(mappos, anchor, width_in, h_in, 0.06);
    XM=ncol*10; YM=nrow+1.4;
    ia=axes('Parent',fig,'Units','inches','Position',[x0 y0 width_in h_in]);
    hold(ia,'on');
    set(ia,'XLim',[0 XM],'YLim',[0 YM],'XTick',[],'YTick',[],'Box','off', ...
        'Color','none','SortMethod','childorder');
    patch(ia,[0 XM XM 0],[0 0 YM YM],'w','FaceAlpha',alpha,'EdgeColor',[0.3 0.3 0.3],'LineWidth',0.6);
    text(ia,0.3,nrow+0.8,titlestr,'FontSize',8,'FontWeight','bold', ...
        'HorizontalAlignment','left','VerticalAlignment','middle');
    for i=1:N
        c=floor((i-1)/nrow); rr=mod(i-1,nrow); xo=c*10; y=nrow-rr-0.5;
        patch(ia,[xo+0.3 xo+1.3 xo+1.3 xo+0.3],[y-0.3 y-0.3 y+0.3 y+0.3],rgb(i,:), ...
            'EdgeColor',[0.3 0.3 0.3],'LineWidth',0.3);
        text(ia,xo+1.55,y,wrap_name(char(names{i})),'FontSize',7,'HorizontalAlignment','left', ...
            'VerticalAlignment','middle','Interpreter','none');
    end
end
function draw_unit_labels(ax, S, spec, xl, yl, fs)
    % Place a text label at the centroid of each major unit's largest in-view
    % polygon (or at a manual [lon lat]); labels outside the frame are skipped.
    if nargin<6||isempty(fs), fs=9; end
    U=S.units; names=string(U.name);
    for i=1:size(spec,1)
        disp_txt = spec{i,1};            % char or cellstr (multi-line)
        want     = string(spec{i,2});    % unit name(s)
        manual   = spec{i,3};            % [] or [lon lat]
        lfs      = fs;                   % per-label font size (col 4 override)
        if size(spec,2)>=4 && ~isempty(spec{i,4}), lfs = spec{i,4}; end
        rot      = 0;                    % per-label rotation (col 5)
        if size(spec,2)>=5 && ~isempty(spec{i,5}), rot = spec{i,5}; end
        if ~isempty(manual)
            [lx,ly]=wa_albers(manual(1),manual(2));
        else
            bestA=-1; lx=NaN; ly=NaN;
            for k=1:numel(U.xy)
                if ~any(names(k)==want), continue; end
                xy=U.xy{k};
                [px,py]=wa_albers(xy(:,1),xy(:,2));
                [a,gx,gy]=poly_centroid(px,py);
                if a>bestA, bestA=a; lx=gx; ly=gy; end
            end
        end
        if ~isfinite(lx) || lx<xl(1) || lx>xl(2) || ly<yl(1) || ly>yl(2)
            if ischar(disp_txt), nm=disp_txt; else, nm=strjoin(disp_txt,' '); end
            fprintf('  unit label skipped (off-frame or not found): %s\n', nm); continue;
        end
        text(ax,lx,ly,disp_txt,'FontSize',lfs,'FontWeight','bold','Color',[0 0 0], ...
            'HorizontalAlignment','center','VerticalAlignment','middle','Interpreter','none', ...
            'Rotation',rot,'BackgroundColor',[1 1 1 0.55],'Margin',0.5,'Tag','unitlabel');
    end
end
function draw_province_contour(ax, S, members, col, lw, sty, framebx, frameby, label)
    % Union the member units into one polyshape, clip to the frame, draw outline.
    if nargin<9, label=''; end
    U=S.units; names=string(U.name); want=string(members);
    warning('off','MATLAB:polyshape:repairedBySimplify');
    warning('off','MATLAB:polyshape:boundary3Points');
    pg=polyshape(); nfound=0;
    for k=1:numel(U.xy)
        if ~any(names(k)==want), continue; end
        xy=U.xy{k}; [px,py]=wa_albers(xy(:,1),xy(:,2));
        good=isfinite(px)&isfinite(py); if nnz(good)<3, continue; end
        try, pg=union(pg, polyshape(px(good),py(good),'Simplify',true)); nfound=nfound+1; catch, end
    end
    if nfound==0 || pg.NumRegions==0
        fprintf('  province "%s": no member units matched - outline skipped\n', label); return;
    end
    try, pg=intersect(pg, polyshape(framebx(:),frameby(:),'Simplify',true)); catch, end   % clip to frame
    plot(ax, pg, 'FaceColor','none', 'EdgeColor',col, 'LineWidth',lw, 'LineStyle',sty, 'Tag','province');
    fprintf('  province "%s": %d member unit(s), %d region(s)\n', label, nfound, pg.NumRegions);
end
function [A,cx,cy] = poly_centroid(x,y)
    % Shoelace area + centroid of a single ring (NaNs stripped).
    x=x(:); y=y(:); good=isfinite(x)&isfinite(y); x=x(good); y=y(good);
    if numel(x)<3, A=0; cx=mean(x); cy=mean(y); return; end
    x2=[x(2:end);x(1)]; y2=[y(2:end);y(1)];
    cr=x.*y2 - x2.*y; A=sum(cr)/2;
    if abs(A)<eps, cx=mean(x); cy=mean(y); A=abs(A); return; end
    cx=sum((x+x2).*cr)/(6*A); cy=sum((y+y2).*cr)/(6*A); A=abs(A);
end
function draw_era_legend(fig, mappos, sections, topfrac, pad, width_in, fs)
    % Boxless era list in its own inches-axes, anchored to the map box: top at
    % topfrac of the map height, flowing down to just above the bottom frame.
    if nargin<7||isempty(fs), fs=7; end
    mx=mappos(1); my=mappos(2); mh=mappos(4);
    nrows=0; for s=1:numel(sections), nrows=nrows+1+numel(sections(s).names); end
    bgap=0.3;                         % lift off the bottom frame so the last row clears
    x0=mx+pad; y0=my+bgap; h_in=topfrac*mh-bgap;
    ia=axes('Parent',fig,'Units','inches','Position',[x0 y0 width_in h_in]);
    hold(ia,'on');
    set(ia,'XLim',[0 10],'YLim',[0 nrows],'XTick',[],'YTick',[],'Box','off', ...
        'Color','none','Clipping','off','SortMethod','childorder','Tag','legend');
    yrow=nrows-0.5;
    for s=1:numel(sections)
        text(ia,0.1,yrow,sections(s).title,'FontSize',fs+1,'FontWeight','bold','VerticalAlignment','middle');
        yrow=yrow-1;
        for i=1:numel(sections(s).names)
            patch(ia,[0.2 1.1 1.1 0.2],[yrow-0.32 yrow-0.32 yrow+0.32 yrow+0.32], ...
                sections(s).rgb(i,:),'EdgeColor',[.3 .3 .3],'LineWidth',0.3);
            text(ia,1.35,yrow,wrap_name(sections(s).names{i}),'FontSize',fs,'FontWeight','bold', ...
                'VerticalAlignment','middle','Interpreter','none');
            yrow=yrow-1;
        end
    end
    fprintf('Legend: %d rows in %.1f in (%.3f in/row)\n', nrows, h_in, h_in/nrows);
end
function [x0,y0] = anchor_pos(mp, anchor, wb, hb, pad)
    mx=mp(1); my=mp(2); mw=mp(3); mh=mp(4);
    switch lower(anchor)
        case 'sw', x0=mx+pad;            y0=my+pad;
        case 'se', x0=mx+mw-wb-pad;      y0=my+pad;
        case 'nw', x0=mx+pad;            y0=my+mh-hb-pad;
        case 'ne', x0=mx+mw-wb-pad;      y0=my+mh-hb-pad;
        case 's',  x0=mx+(mw-wb)/2;      y0=my+pad;
        case 'n',  x0=mx+(mw-wb)/2;      y0=my+mh-hb-pad;
        case 'w',  x0=mx+pad;            y0=my+(mh-hb)/2;
        case 'e',  x0=mx+mw-wb-pad;      y0=my+(mh-hb)/2;
        otherwise, x0=mx+pad;            y0=my+pad;
    end
end
function s = wrap_name(name)
    if contains(name,', ')
        p=strsplit(name,', '); s={p{1}; strjoin(p(2:end),', ')};
    elseif length(name) > 24
        sp=strfind(name,' ');
        if isempty(sp), s={name};
        else
            [~,k]=min(abs(sp - length(name)/2));   % space nearest the middle
            s={strtrim(name(1:sp(k))); strtrim(name(sp(k)+1:end))};
        end
    else
        s={name};
    end
end
function draw_key_ne(fig, mappos, anchor, alpha, showprof, profcol, proflw)
    if nargin<4||isempty(alpha), alpha=1.0; end
    if nargin<5||isempty(showprof), showprof=true; end
    if nargin<6||isempty(profcol), profcol=[0.10 0.20 0.85 0.40]; end
    if nargin<7||isempty(proflw),  proflw=5.6; end
    items = {};
    if showprof, items{end+1}='profile'; end
    items{end+1}='bnd'; items{end+1}='bndlow'; items{end+1}='station';
    n = numel(items);
    ww=2.5; YM=n+1.3; hh=0.33*YM;
    [x0,y0]=anchor_pos(mappos, anchor, ww, hh, 0.06);
    ia=axes('Parent',fig,'Units','inches','Position',[x0 y0 ww hh]);
    hold(ia,'on');
    set(ia,'XLim',[0 10],'YLim',[0 YM],'XTick',[],'YTick',[],'Box','off','Color','none');
    patch(ia,[0 10 10 0],[0 0 YM YM],'w','FaceAlpha',alpha,'EdgeColor',[0.3 0.3 0.3],'LineWidth',0.6);
    text(ia,0.3,YM-0.5,'Map key','FontSize',8,'FontWeight','bold','VerticalAlignment','middle');
    for k=1:n
        y=n-k+0.5;
        switch items{k}
            case 'profile'
                hl=line(ia,[0.4 1.8],[y y],'LineWidth',min(proflw,4),'Color',profcol(1:3));
                if numel(profcol)>=4
                    try, hl.Color=profcol(1:4); catch, hl.Color=profcol(1:3)+(1-profcol(1:3))*(1-profcol(4)); end
                end
                plot(ia,1.1,y,'o','MarkerSize',3,'MarkerFaceColor','w','MarkerEdgeColor','k','LineWidth',0.5);
                txt='Seismic Profile (100 km ticks)';
            case 'bnd'
                line(ia,[0.4 1.8],[y y],'Color','k','LineWidth',1.1); txt='major crustal boundary';
            case 'bndlow'
                line(ia,[0.4 1.8],[y y],'Color','k','LineWidth',1.1,'LineStyle','--'); txt='boundary (low confidence)';
            case 'station'
                plot(ia,1.1,y,'^','MarkerSize',5,'MarkerEdgeColor','k','MarkerFaceColor','k','LineWidth',0.6);
                txt='seismic station';
        end
        text(ia,2.1,y,txt,'FontSize',7.5,'FontWeight','bold','VerticalAlignment','middle');
    end
end
function draw_profile(ax, lab, p1, p2, poly, tick_km, col, lw)
    if nargin<7||isempty(col), col=[0.10 0.20 0.85 0.40]; end
    if nargin<8||isempty(lw),  lw=5.6; end
    if isempty(poly)
        n=2000; la=linspace(p1(1),p2(1),n)'; lo=linspace(p1(2),p2(2),n)';
    else
        la=[]; lo=[];
        for s=1:size(poly,1)-1
            la=[la; linspace(poly(s,1),poly(s+1,1),400)']; lo=[lo; linspace(poly(s,2),poly(s+1,2),400)'];
        end
    end
    d=[0; cumsum(hav(la(1:end-1),lo(1:end-1),la(2:end),lo(2:end)))];
    [X,Y]=wa_albers(lo,la);
    % Tags let you toggle these on a finished figure without re-plotting:
    %   profile line + name label -> Tag 'profile'
    %   100 km ticks + distance label -> Tag 'profilemark'
    % e.g.  set(findobj(gcf,'Tag','profilemark'),'Visible','off')     % drop distance marks
    %       set(findobj(gcf,'-regexp','Tag','^profile'),'Visible','off')  % drop all profile stuff
    hh=plot(ax,X,Y,'-','LineWidth',lw,'Color',col(1:3),'Tag','profile');
    if numel(col)>=4      % apply line transparency (fallback: lighten toward white)
        try, hh.Color=col(1:4); catch, hh.Color=col(1:3)+(1-col(1:3))*(1-col(4)); end
    end
    for dd=0:tick_km:d(end)
        [~,ix]=min(abs(d-dd));
        plot(ax,X(ix),Y(ix),'o','MarkerSize',3.0,'MarkerFaceColor','w','MarkerEdgeColor','k','LineWidth',0.5,'Tag','profilemark');
    end
    text(ax,X(1),Y(1),[' ' lab],'FontSize',8,'FontWeight','bold','Color','k', ...
        'BackgroundColor',[1 1 1 0.6],'Margin',0.5,'HorizontalAlignment','left','Tag','profile');
    text(ax,X(end),Y(end),sprintf(' %g km',round(d(end))),'FontSize',7,'FontWeight','bold','Color','k', ...
        'BackgroundColor',[1 1 1 0.6],'Margin',0.5,'HorizontalAlignment','left','VerticalAlignment','middle','Tag','profilemark');
end
function draw_deposits(ax, spec, szrange, gamma)
% spec = {matfile, var, marker, colour, filled?}. Open or filled markers at the
% deposits in <matfile>.<var> (fields lon/lat/size), sized by the contained-metal
% 'size'. The size spans ~6 decades, so we rank each deposit by log10(size),
% normalise to 0..1, raise to GAMMA (>1 keeps the small/medium small so the
% giants stand out), then map to the marker-area range.
    if nargin<4||isempty(gamma), gamma=2.0; end
    matfile = spec{1}; var = spec{2}; marker = spec{3}; col = spec{4}; filled = spec{5};
    if isempty(matfile) || ~isfile(matfile)
        fprintf('  (%s: file not found, skipped: %s)\n', var, matfile); return
    end
    Gd = load(matfile); g = Gd.(var);
    v = double(g.size(:)); v(v <= 0 | ~isfinite(v)) = NaN;
    [x, y] = wa_albers(double(g.lon(:)), double(g.lat(:)));
    lv = log10(v); lo = min(lv); hi = max(lv);
    frac = (lv - lo) ./ max(hi - lo, eps);            % 0..1 by log-rank
    sz = szrange(1) + (szrange(2)-szrange(1)) .* frac.^gamma;
    ok = isfinite(x) & isfinite(y) & isfinite(sz);
    if filled
        scatter(ax, x(ok), y(ok), sz(ok), marker, 'MarkerFaceColor', col, ...
                'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.7, 'LineWidth', 0.3, 'Tag', var);
    else
        scatter(ax, x(ok), y(ok), sz(ok), marker, 'MarkerEdgeColor', col, ...
                'MarkerFaceColor', 'none', 'LineWidth', 0.8, 'Tag', var);
    end
    fprintf('%s: %d deposits (size %.3g-%.3g)\n', var, nnz(ok), min(v), max(v));
end

function km = hav(la1,lo1,la2,lo2)
    R=6371; dla=deg2rad(la2-la1); dlo=deg2rad(lo2-lo1);
    a=sin(dla/2).^2 + cosd(la1).*cosd(la2).*sin(dlo/2).^2; km=R*2.*atan2(sqrt(a),sqrt(1-a));
end
function P = read_all_profiles(deffile, base)
% Read the shared ccp_profiles.txt -> struct array, ALL groups. Same columns as
% read_ccp_profiles in plot_ccp_landscape_WA.m:
%   short  group(senw|ns|we)  type(gc|file)  args
%     gc   -> args = lat1 lon1 lat2 lon2   (straight line between endpoints)
%     file -> args = latlon .mat           (curved pick; resolved vs BASE)
% gc profiles return endpoints (p1,p2) and draw as a straight 2-pt stroke -> the
% transparent blue survives. file profiles return a decimated polyline (poly) via
% read_west. '#' lines and blanks are skipped. Fields: short, kind, p1, p2, poly.
    P = struct('short',{},'kind',{},'p1',{},'p2',{},'poly',{});
    fid = fopen(deffile,'r');
    assert(fid > 0, '[profiles] cannot open %s', deffile);
    closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
    while true
        ln = fgetl(fid);
        if ~ischar(ln), break; end
        ln = strtrim(ln);
        if isempty(ln) || ln(1) == '#', continue; end
        tok = strsplit(ln);
        if numel(tok) < 4, continue; end
        short = tok{1}; grp = lower(tok{2}); typ = lower(tok{3});
        switch typ
            case 'gc'
                v = str2double(tok(4:7));
                assert(all(isfinite(v)), '[profiles] bad gc coords for %s', short);
                p1 = [v(1) v(2)]; p2 = [v(3) v(4)];
                [p1, p2] = orient_gc(grp, p1, p2);        % W->E, N->S, NW->SE
                P(end+1) = struct('short',short,'kind','gc','p1',p1,'p2',p2,'poly',[]); %#ok<AGROW>
            case 'file'
                f = tok{4};
                if ~isfile(f), f = fullfile(base, f); end
                assert(isfile(f), '[profiles] latlon file not found for %s: %s', short, tok{4});
                ll = read_west(f);                        % [lat lon], flexible + decimated
                ll = orient_latlon(grp, ll);
                P(end+1) = struct('short',short,'kind','file','p1',[],'p2',[],'poly',ll); %#ok<AGROW>
            otherwise
                error('[profiles] unknown type "%s" for %s', typ, short);
        end
    end
    assert(~isempty(P), '[profiles] no profiles parsed from %s', deffile);
end
function [p1, p2] = orient_gc(grp, p1, p2)
% Put the canonical start first (p1): we -> W (min lon); ns/senw -> N/NW (max lat).
    if strcmp(grp,'we')
        if p1(2) > p2(2), t=p1; p1=p2; p2=t; end
    else
        if p1(1) < p2(1), t=p1; p1=p2; p2=t; end
    end
end
function ll = orient_latlon(grp, ll)
% Canonical direction so the label/0-km end is consistent (matches the landscape):
%   we -> start at W (min lon);  ns / senw -> start at N / NW (max lat).
    if strcmp(grp,'we')
        if ll(1,2) > ll(end,2), ll = flipud(ll); end
    else
        if ll(1,1) < ll(end,1), ll = flipud(ll); end
    end
end
function ll = read_west(f)
% [lat lon] Nx2 from a picked-polyline .mat, whatever the fields are called.
    W=load(f); fn=fieldnames(W); la=[]; lo=[];
    for c={'lat','latitude','la','y'}, if isfield(W,c{1}), la=W.(c{1})(:); break; end, end
    for c={'lon','long','longitude','lo','x'}, if isfield(W,c{1}), lo=W.(c{1})(:); break; end, end
    if isempty(la)||isempty(lo)            % detect vector fields by value range
        for k=1:numel(fn), v=W.(fn{k});
            if isnumeric(v)&&isvector(v)&&numel(v)>1
                m=median(double(v(:)),'omitnan');
                if m<0&&m>-60&&isempty(la), la=v(:); end
                if m>90&&m<180&&isempty(lo), lo=v(:); end
            end
        end
    end
    if isempty(la)||isempty(lo)            % or a single Nx2 / 2xN matrix
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
function fam = narrow_font()
    have=listfonts; fam='Helvetica';
    for c={'Arial Narrow','Liberation Sans Narrow','Nimbus Sans Narrow','Arial'}
        if any(strcmpi(have,c{1})), fam=c{1}; return; end
    end
end
