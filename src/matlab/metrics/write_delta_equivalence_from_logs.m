function [summary, ecosystem_summary] = write_delta_equivalence_from_logs(log_dir, summary_file, varargin)
%WRITE_DELTA_EQUIVALENCE_FROM_LOGS Build equivalence summaries from CSV logs.
%
% This is intended for already-computed WLNM_dir_neg prediction logs. It reads
% the per-food-web CSV files, attaches Foodweb/EcosystemType metadata, and then
% delegates to write_delta_equivalence_summary.

    p = inputParser;
    addParameter(p, 'alpha', 0.05);
    addParameter(p, 'version', 'WLNM_dir_neg');
    addParameter(p, 'byEcosystemFile', '');
    addParameter(p, 'marginsFile', '');
    addParameter(p, 'metadataFile', '');
    addParameter(p, 'trainRatios', []);
    parse(p, varargin{:});
    opt = p.Results;

    if nargin < 1 || isempty(log_dir)
        error('write_delta_equivalence_from_logs:MissingLogDir', ...
            'log_dir must be provided.');
    end
    if nargin < 2 || isempty(summary_file)
        error('write_delta_equivalence_from_logs:MissingSummaryFile', ...
            'summary_file must be provided.');
    end
    if ~isfolder(log_dir)
        error('write_delta_equivalence_from_logs:LogDirNotFound', ...
            'Log directory does not exist: %s', log_dir);
    end

    ecosystem_by_foodweb = read_ecosystem_metadata(char(string(opt.metadataFile)));
    results = read_prediction_logs(log_dir, ecosystem_by_foodweb, opt.trainRatios);

    write_delta_equivalence_summary(summary_file, results, ...
        'alpha', opt.alpha, ...
        'version', opt.version, ...
        'byEcosystemFile', opt.byEcosystemFile, ...
        'marginsFile', opt.marginsFile);

    if nargout >= 1
        summary = readtable(summary_file, 'TextType', 'string');
    end

    if nargout >= 2
        by_ecosystem_file = char(string(opt.byEcosystemFile));
        if isempty(by_ecosystem_file)
            by_ecosystem_file = derive_by_ecosystem_file(summary_file);
        end

        if isfile(by_ecosystem_file)
            ecosystem_summary = readtable(by_ecosystem_file, 'TextType', 'string');
        else
            ecosystem_summary = table();
        end
    end
end

function results = read_prediction_logs(log_dir, ecosystem_by_foodweb, train_ratios)
    files = dir(fullfile(log_dir, '*.csv'));
    if isempty(files)
        results = struct([]);
        return;
    end

    results = struct([]);

    for i = 1:numel(files)
        path = fullfile(files(i).folder, files(i).name);
        T = readtable(path, 'TextType', 'string');
        if isempty(T)
            continue;
        end

        if ismember('TrainRatio', T.Properties.VariableNames) && ~isempty(train_ratios)
            keep = train_ratio_mask(T.TrainRatio, train_ratios);
            T = T(keep, :);
        end

        if isempty(T)
            continue;
        end

        foodweb = foodweb_from_log_filename(files(i).name);
        ecosystem_type = 'Unknown';
        if isKey(ecosystem_by_foodweb, foodweb)
            ecosystem_type = ecosystem_by_foodweb(foodweb);
        end

        S = table2struct(T);
        for r = 1:numel(S)
            S(r).Foodweb = foodweb;
            S(r).EcosystemType = ecosystem_type;
        end

        if isempty(results)
            results = S(:);
        else
            results = [results(:); S(:)];
        end
    end
end

function keep = train_ratio_mask(values, train_ratios)
    values = double(values);
    requested = double(train_ratios(:));
    keep = false(size(values));

    for i = 1:numel(requested)
        r = requested(i);
        keep = keep | abs(values - r) < 1e-9;

        if r <= 1
            keep = keep | abs(values - 100 * r) < 1e-9;
        else
            keep = keep | abs(values / 100 - r) < 1e-9;
        end
    end
end

function ecosystem_by_foodweb = read_ecosystem_metadata(metadata_file)
    ecosystem_by_foodweb = containers.Map('KeyType', 'char', 'ValueType', 'char');

    if isempty(metadata_file) || ~isfile(metadata_file)
        return;
    end

    T = readtable(metadata_file, 'TextType', 'string');
    required = {'Foodweb', 'EcosystemType'};
    if ~all(ismember(required, T.Properties.VariableNames))
        warning('write_delta_equivalence_from_logs:InvalidMetadata', ...
            'Metadata file lacks Foodweb/EcosystemType columns: %s', metadata_file);
        return;
    end

    for i = 1:height(T)
        foodweb = char(string(T.Foodweb(i)));
        ecosystem_type = char(string(T.EcosystemType(i)));
        if ~isempty(foodweb)
            ecosystem_by_foodweb(foodweb) = ecosystem_type;
        end
    end
end

function foodweb = foodweb_from_log_filename(filename)
    suffix = '_results_random_wlnm_dir_neg.csv';
    if endsWith(filename, suffix)
        foodweb = filename(1:end - length(suffix));
        return;
    end

    [~, foodweb] = fileparts(filename);
end

function by_ecosystem_file = derive_by_ecosystem_file(summary_file)
    [out_dir, base_name, ext] = fileparts(summary_file);
    if isempty(ext)
        ext = '.csv';
    end
    by_ecosystem_file = fullfile(out_dir, [base_name '_by_ecosystem' ext]);
end
