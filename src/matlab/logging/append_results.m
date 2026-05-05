function append_results(log_file, results, use_backbone)
%APPEND_RESULTS Append an array of result structs to CSV.
%
% The first column is sequential across the whole file:
%   - backbone mode -> ExpID
%   - non-backbone  -> Iteration
%
% This version is header-driven and avoids fragile cell concatenation.

    if nargin < 3
        use_backbone = false;
    end

    assert(isfile(log_file), ...
        'append_results:MissingLogFile', ...
        'Log file does not exist: %s. Call init_log_file() first.', log_file);

    % ------------------------------------------------------------
    % Read header and find last used sequential ID
    % ------------------------------------------------------------
    fid_read = fopen(log_file, 'r');
    assert(fid_read ~= -1, 'Cannot open %s for reading.', log_file);
    c_read = onCleanup(@() fclose(fid_read));

    header_line = fgetl(fid_read);
    assert(ischar(header_line), 'Log file %s is empty or unreadable.', log_file);

    header = strsplit(strtrim(header_line), ',');
    ncols  = numel(header);

    lastID = 0;
    line = fgetl(fid_read);
    while ischar(line)
        if ~isempty(strtrim(line))
            commaPos = find(line == ',', 1, 'first');
            if ~isempty(commaPos)
                token = line(1:commaPos-1);
            else
                token = line;
            end
            val = str2double(strtrim(token));
            if ~isnan(val)
                lastID = val;
            end
        end
        line = fgetl(fid_read);
    end

    % ------------------------------------------------------------
    % Append rows
    % ------------------------------------------------------------
    fid = fopen(log_file, 'a');
    assert(fid ~= -1, 'Cannot open %s for appending.', log_file);
    c = onCleanup(@() fclose(fid));

    for i = 1:numel(results)
        expID = lastID + i;
        row = result_to_row(results(i), expID, header, ncols, use_backbone);
        write_csv_row(fid, row);
    end
end

% ============================================================
% Build one 1 x N cell row exactly matching the header
% ============================================================
function row = result_to_row(r, expID, header, ncols, use_backbone)

    row = cell(1, ncols);

    for j = 1:ncols
        col = header{j};

        switch col
            case {'ExpID', 'Iteration'}
                row{j} = expID;

            case 'TimeElapsed'
                row{j} = getfield_or(r, 'TimeElapsed', '');

            case 'ElapsedTime'
                row{j} = getfield_or(r, 'TimeElapsed', '');

            case 'TrainRatio'
                row{j} = 100 * getfield_or(r, 'TrainRatio', NaN);

            case 'BackboneRatio'
                row{j} = 100 * getfield_or(r, 'BackboneRatio', NaN);

            case 'NonBackboneRatio'
                row{j} = 100 * getfield_or(r, 'BackboneRatio', NaN);

            case 'BestThreshold'
                row{j} = getfield_or(r, 'Threshold', NaN);

            case 'Threshold'
                row{j} = getfield_or(r, 'Threshold', NaN);

            otherwise
                row{j} = getfield_or(r, col, default_for_column(col, use_backbone));
        end
    end
end

% ============================================================
% Default values for missing fields
% ============================================================
function v = default_for_column(col, use_backbone)

    switch col
        case {'CvK', 'FoldID', 'NumFolds', 'ExperimentID', 'Seed'}
            if use_backbone
                v = 0;
            else
                v = 0;
            end

        case {'TimeElapsed', 'ElapsedTime', 'Version', 'ThresholdMode'}
            v = '';

        otherwise
            v = NaN;
    end
end

% ============================================================
% Write one CSV row
% ============================================================
function write_csv_row(fid, row)
    txt = cell(1, numel(row));
    for j = 1:numel(row)
        txt{j} = csv_value_to_string(row{j});
    end
    fprintf(fid, '%s\n', strjoin(txt, ','));
end

% ============================================================
% Convert scalar value to CSV-safe string
% ============================================================
function s = csv_value_to_string(v)

    if isempty(v)
        s = '';

    elseif isstring(v) || ischar(v)
        s = char(string(v));

    elseif isnumeric(v) || islogical(v)
        if ~isscalar(v)
            error('csv_value_to_string:NonScalarNumeric', ...
                'Only scalar numeric values can be written to the results CSV.');
        end

        if isnan(v)
            s = 'NaN';
        elseif isinf(v)
            if v > 0
                s = 'Inf';
            else
                s = '-Inf';
            end
        elseif abs(v - round(v)) < 1e-12
            s = sprintf('%d', round(v));
        else
            s = sprintf('%.4f', v);
        end

    else
        error('csv_value_to_string:UnsupportedType', ...
            'Unsupported value type in CSV writer: %s', class(v));
    end
end

% ============================================================
% Safe struct field access
% ============================================================
function v = getfield_or(s, fieldName, defaultValue)
    if isfield(s, fieldName) && ~isempty(s.(fieldName))
        v = s.(fieldName);
    else
        v = defaultValue;
    end
end
