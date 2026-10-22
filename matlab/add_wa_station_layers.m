function h = add_wa_station_layers(ax, varargin)
%ADD_WA_STATION_LAYERS  WA Array station layers on a WA map axes.
%   h = add_wa_station_layers(ax, 'Name',Value, ...)
%
%   Draws the WA Array station layers used by the Python locator map
%   (plotting/plot_wa_map.py in huaiyuyuan/wa-geo-plotting), reading the same
%   GSWA "New.WA.Seismic" CSV exports so the Python and MATLAB maps stay fed
%   from one source.
%
%   The axes may be in one of two coordinate systems, selected by 'Project':
%     'none'/'lonlat' (default)  AX data units are LONGITUDE (x), LATITUDE (y)
%                                in degrees -- e.g. the CCP locator base map
%                                fig.base.map.fig, where everything is
%                                plot(ax, lon, lat). No projection applied.
%     'albers'                   AX is in GSWA Albers km (e.g. wa_basemap_axes
%                                / wa_plot_map). Station lon/lat are projected
%                                through wa_albers before plotting, so the
%                                layers register on the tectonic base.
%
%   Conventions (identical to the Python map):
%     Available  red square,  white edge   (WA Array deployed)
%     Permanent  red square,  black edge   (labelled with station codes)
%     Phase 4    lime square, white edge
%
%   The deployed "Available" array is normally already baked into
%   fig.base.map.fig, so 'ShowAvailable' defaults to false. Permanent and
%   Phase 4 are the new layers brought onto the map.
%
%   Every object drawn is given a Tag (default 'ccp_overlay') so the caller's
%   existing  delete(findobj(fig,'Tag','ccp_overlay'))  clears and redraws them
%   on each run, exactly like the profile lines in plot_ccp_profiles_WA.m.
%
%   Name/Value options
%     'Permanent'       CSV path for the permanent/phase-0 stations     ('')
%     'Phase4'          CSV path for the Phase 4 stations               ('')
%     'Available'       CSV path for the deployed WA Array              ('')
%     'ShowPermanent'   draw the permanent layer            (true)
%     'ShowPhase4'      draw the Phase 4 layer              (true)
%     'ShowAvailable'   draw the available layer            (false)
%     'LabelPermanent'  annotate permanent stations with their code   (true)
%     'LabelPhase4'     annotate Phase 4 stations with their code      (false)
%     'LabelAvailable'  annotate available stations with their code    (false)
%     'Project'         'none'|'lonlat'|'albers' coordinate system  ('none')
%     'MarkerSize'      marker size (points)               (6)
%     'CodeFontSize'    station-code label font size        (6)
%     'CodeOffset'      code label offset, in DEGREES of longitude east of the
%                       marker (applied before projection)  (0.12)
%     'ShowLegend'      add a "Stations" legend             (true)
%     'LegendLocation'  legend location                    ('southeast')
%     'Tag'             tag for every object drawn          ('ccp_overlay')
%
%   Returns h, a struct of graphics handles (one field per drawn layer, plus
%   .legend when a legend is made), handy for further tweaking.
%
%   With 'Project','albers', wa_albers.m must be on the MATLAB path.
%
%   Example (lon/lat CCP locator, inside plot_ccp_profiles_WA.m):
%     DL = fullfile(getenv('HOME'),'Downloads');
%     add_wa_station_layers(axmap, ...
%        'Permanent', fullfile(DL,'New.WA.Seismic.2026- Perm-phase-0.csv'), ...
%        'Phase4',    fullfile(DL,'New.WA.Seismic.2026- Phase 4.csv'));
%
%   Example (Albers tectonic base, e.g. the CCP Moho panels):
%     add_wa_station_layers(ax, 'Permanent', PERM_CSV, 'Phase4', P4_CSV, ...
%        'Project','albers');

    p = inputParser;
    p.addRequired('ax');
    p.addParameter('Permanent', '');
    p.addParameter('Phase4', '');
    p.addParameter('Available', '');
    p.addParameter('ShowPermanent', true);
    p.addParameter('ShowPhase4', true);
    p.addParameter('ShowAvailable', false);
    p.addParameter('LabelPermanent', true);
    p.addParameter('LabelPhase4', false);
    p.addParameter('LabelAvailable', false);
    p.addParameter('Project', 'none');
    p.addParameter('MarkerSize', 6);
    p.addParameter('CodeFontSize', 6);
    p.addParameter('CodeOffset', 0.12);
    p.addParameter('ShowLegend', true);
    p.addParameter('LegendLocation', 'southeast');
    p.addParameter('Tag', 'ccp_overlay');
    p.parse(ax, varargin{:});
    o = p.Results;

    o.Project = lower(char(o.Project));
    if ~ismember(o.Project, {'none','lonlat','albers'})
        error('add_wa_station_layers:proj', ...
            'Project must be ''none'', ''lonlat'' or ''albers'' (got ''%s'').', o.Project);
    end
    if strcmp(o.Project, 'albers') && exist('wa_albers','file') ~= 2
        error('add_wa_station_layers:noproj', ...
            'Project=''albers'' needs wa_albers.m on the MATLAB path.');
    end

    LIME = [0 1 0];          % matplotlib 'lime' = #00FF00, as the Python map
    RED  = [1 0 0];

    % layer table: {field, csv, show, label, faceColour, edgeColour, legendText}
    layers = {
        'available', o.Available, o.ShowAvailable, o.LabelAvailable, RED,  'w', 'Available'
        'permanent', o.Permanent, o.ShowPermanent, o.LabelPermanent, RED,  'k', 'Permanent'
        'phase4',    o.Phase4,    o.ShowPhase4,    o.LabelPhase4,    LIME, 'w', 'Phase 4'  };

    washold = ishold(ax);
    hold(ax, 'on');

    h = struct();
    leg_h = gobjects(0);
    leg_t = {};
    for k = 1:size(layers,1)
        fld   = layers{k,1};
        csv   = layers{k,2};
        show  = layers{k,3};
        dolab = layers{k,4};
        face  = layers{k,5};
        edge  = layers{k,6};
        ltext = layers{k,7};
        if ~show,            continue; end
        if isempty(csv)
            warning('add_wa_station_layers:noCSV', ...
                'Layer "%s" is on but no CSV was given -- skipped.', ltext);
            continue;
        end
        if exist(csv,'file') ~= 2
            warning('add_wa_station_layers:missing', ...
                'Layer "%s": file not found (%s) -- skipped.', ltext, csv);
            continue;
        end
        [lon, lat, code] = local_read_station_csv(csv);
        if isempty(lon)
            warning('add_wa_station_layers:empty', ...
                'Layer "%s": no usable rows in %s -- skipped.', ltext, csv);
            continue;
        end
        [X, Y] = local_project(o.Project, lon, lat);
        hp = plot(ax, X, Y, 's', 'LineStyle','none', ...
            'MarkerFaceColor', face, 'MarkerEdgeColor', edge, ...
            'MarkerSize', o.MarkerSize, 'LineWidth', 0.4, ...
            'Tag', o.Tag, 'DisplayName', ltext);
        h.(fld) = hp;
        leg_h(end+1) = hp;          %#ok<AGROW>
        leg_t{end+1} = ltext;       %#ok<AGROW>
        fprintf('  station layer "%s": %d stations <- %s\n', ltext, numel(lon), csv);

        if dolab
            if isempty(code)
                warning('add_wa_station_layers:noCode', ...
                    'Layer "%s": no code column (Code/Station/Sta/Site/Name) in %s -- labels skipped.', ...
                    ltext, csv);
            else
                th = gobjects(numel(lon),1);
                for i = 1:numel(lon)
                    c = strtrim(code{i});
                    if isempty(c), continue; end
                    % offset east in degrees, then project, so the label sits
                    % just right of the marker in either coordinate system
                    [xl, yl] = local_project(o.Project, lon(i)+o.CodeOffset, lat(i));
                    th(i) = text(ax, xl, yl, c, ...
                        'FontSize', o.CodeFontSize, ...
                        'HorizontalAlignment','left', 'VerticalAlignment','middle', ...
                        'Color','k', 'BackgroundColor',[1 1 1 0.6], 'Margin',0.5, ...
                        'Clipping','on', 'Tag', o.Tag);
                end
                h.([fld '_labels']) = th(isgraphics(th));
            end
        end
    end

    if o.ShowLegend && ~isempty(leg_h)
        lg = legend(ax, leg_h, leg_t, 'Location', o.LegendLocation, ...
            'FontSize', 7, 'AutoUpdate', 'off');
        try, lg.Title.String = 'Stations'; catch, end   % older MATLAB: no title
        set(lg, 'Tag', o.Tag);
        h.legend = lg;
    end

    if ~washold, hold(ax, 'off'); end
end


% ------------------------------------------------------------------------
function [X, Y] = local_project(mode, lon, lat)
%LOCAL_PROJECT  identity for lon/lat axes, wa_albers for an Albers base.
    if strcmp(mode, 'albers')
        [X, Y] = wa_albers(lon, lat);
    else
        X = lon; Y = lat;
    end
end


% ------------------------------------------------------------------------
function [lon, lat, code] = local_read_station_csv(path)
%LOCAL_READ_STATION_CSV  lon, lat and (if present) code from a GSWA station CSV.
%   Reads a CSV with named Latitude/Longitude columns (any order; header names
%   may contain spaces, e.g. the GSWA "New.WA.Seismic" exports). CODE is a
%   cellstr from the first of Code/Station Code/Station/Sta/Site/Name found, or
%   {} if no such column exists. Matches _load_station_csv in plot_wa_map.py.
    lon = []; lat = []; code = {};
    T  = readtable(path, 'VariableNamingRule','preserve');
    vn = T.Properties.VariableNames;
    vl = lower(strtrim(vn));
    ila = local_find(vl, {'latitude','lat'});
    ilo = local_find(vl, {'longitude','lon','long'});
    if isempty(ila) || isempty(ilo)
        error('add_wa_station_layers:cols', ...
            '%s: need Latitude/Longitude columns; header was: %s', ...
            path, strjoin(vn, ', '));
    end
    lat = local_tonum(T.(vn{ila}));
    lon = local_tonum(T.(vn{ilo}));
    ic  = local_find(vl, {'code','station code','station','sta','site','name'});
    if ~isempty(ic)
        col = T.(vn{ic});
        if isnumeric(col)
            code = strtrim(cellstr(num2str(col(:))));
        else
            code = cellstr(string(col));
        end
    end
    good = isfinite(lon) & isfinite(lat);
    lon = lon(good); lat = lat(good);
    if ~isempty(code), code = code(good); end
end


function idx = local_find(vl, names)
%LOCAL_FIND  index of the first of NAMES present in the lowercased header VL.
    idx = [];
    for n = 1:numel(names)
        j = find(strcmp(vl, names{n}), 1);
        if ~isempty(j), idx = j; return; end
    end
end


function v = local_tonum(x)
%LOCAL_TONUM  coerce a table column (numeric, char or cellstr) to double.
    if isnumeric(x)
        v = double(x(:));
    else
        v = str2double(string(x(:)));
    end
end
