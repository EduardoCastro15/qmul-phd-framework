function summary = write_delta_equivalence_summary(summary_file, results, varargin)
%WRITE_DELTA_EQUIVALENCE_SUMMARY Write TOST equivalence tests for deltas.
%
% This implements Lachlan's suggested question:
%
%   Is the mean paired difference small enough to ignore?
%
% using two one-sided tests (TOST) on food-web-level mean deltas:
%
%   delta = pseudo_metric - empirical_metric
%
% Repeated runs for the same food web are averaged first, matching
% write_delta_ttest_summary and avoiding pseudoreplication.

    p = inputParser;
    addParameter(p, 'alpha', 0.05);
    addParameter(p, 'version', '');
    addParameter(p, 'byEcosystemFile', '');
    addParameter(p, 'marginsFile', '');
    addParameter(p, 'includeUnconfiguredMetrics', false);
    parse(p, varargin{:});
    opt = p.Results;

    if nargin < 1 || isempty(summary_file)
        error('write_delta_equivalence_summary:MissingFile', ...
            'summary_file must be provided.');
    end

    alpha = double(opt.alpha);
    version = char(string(opt.version));
    by_ecosystem_file = char(string(opt.byEcosystemFile));
    margins_file = char(string(opt.marginsFile));
    include_unconfigured = logical(opt.includeUnconfiguredMetrics);

    margins = load_equivalence_margins(margins_file);

    if nargin < 2 || isempty(results)
        summary = empty_summary_table();
        write_summary_table(summary_file, summary);
        if ~isempty(by_ecosystem_file)
            write_summary_table(by_ecosystem_file, empty_ecosystem_summary_table());
        end
        return;
    end

    structural_metric_names = extended_structural_metric_names();

    metric_names = [ ...
    { ...
        'Links', ...
        'Connectance', ...
        'MeanTrophicLevel', ...
        'MeanDegree', ...
        'MeanGenerality', ...
        'MeanVulnerability', ...
        'PropBasal', ...
        'PropIntermediate', ...
        'PropTop' ...
    }, ...
    structural_metric_names];

    metric_fields = strcat('Delta', metric_names);
    empirical_fields = strcat('Empirical', metric_names);
    pseudo_fields = strcat('Pseudo', metric_names);

    summary = build_summary( ...
        results, metric_names, metric_fields, empirical_fields, pseudo_fields, ...
        margins, alpha, version, false, include_unconfigured);
    write_summary_table(summary_file, summary);

    fprintf('[DeltaEquivalence] Wrote %d rows to %s\n', height(summary), summary_file);

    if ~isempty(by_ecosystem_file)
        ecosystem_summary = build_summary( ...
            results, metric_names, metric_fields, empirical_fields, pseudo_fields, ...
            margins, alpha, version, true, include_unconfigured);
        write_summary_table(by_ecosystem_file, ecosystem_summary);

        fprintf('[DeltaEquivalence] Wrote %d ecosystem rows to %s\n', ...
            height(ecosystem_summary), by_ecosystem_file);
    end
end

function summary = build_summary( ...
    results, metric_names, metric_fields, empirical_fields, pseudo_fields, ...
    margins, alpha, version, include_ecosystem, include_unconfigured)

    groups = build_groups(results, include_ecosystem);
    out = empty_output_struct(include_ecosystem);

    if include_ecosystem
        test_type = 'tost_equivalence_on_foodweb_mean_paired_deltas_by_ecosystem';
    else
        test_type = 'tost_equivalence_on_foodweb_mean_paired_deltas';
    end

    for g = 1:numel(groups)
        idx = groups(g).Indices;
        group_rows = results(idx);

        for m = 1:numel(metric_fields)
            metric_name = metric_names{m};
            margin = margin_for_metric(margins, metric_name);
            has_margin = ~isempty(margin);

            if ~has_margin && ~include_unconfigured
                continue;
            end

            delta_means = foodweb_level_means(group_rows, metric_fields{m});
            empirical_means = foodweb_level_means(group_rows, empirical_fields{m});
            pseudo_means = foodweb_level_means(group_rows, pseudo_fields{m});

            mean_empirical = mean_or_nan(empirical_means);
            mean_pseudo = mean_or_nan(pseudo_means);

            if has_margin
                [lower_margin, upper_margin] = resolve_margin(margin, mean_empirical);
                margin_mode = char(string(margin.MarginMode));
                justification = char(string(margin.Justification));
            else
                lower_margin = NaN;
                upper_margin = NaN;
                margin_mode = 'unconfigured';
                justification = 'No default margin configured; metric not tested by default.';
            end

            [stats, n_foodwebs] = one_sample_tost( ...
                delta_means, lower_margin, upper_margin, alpha);

            if n_foodwebs == 0
                continue;
            end

            out.Version{end+1, 1} = version;
            out.TestType{end+1, 1} = test_type;
            out.Metric{end+1, 1} = metric_name;
            if include_ecosystem
                out.EcosystemType{end+1, 1} = groups(g).EcosystemType;
            end
            out.K(end+1, 1) = groups(g).K;
            out.TrainRatio(end+1, 1) = groups(g).TrainRatio;
            out.BackboneRatio(end+1, 1) = groups(g).BackboneRatio;
            out.CvK(end+1, 1) = groups(g).CvK;
            out.ThresholdMode{end+1, 1} = groups(g).ThresholdMode;
            out.Threshold(end+1, 1) = groups(g).Threshold;
            out.EvaluateOnAllUnseen(end+1, 1) = groups(g).EvaluateOnAllUnseen;
            out.NumFoodWebs(end+1, 1) = n_foodwebs;
            out.NumResultRows(end+1, 1) = numel(idx);
            out.MeanEmpirical(end+1, 1) = mean_empirical;
            out.MeanPseudo(end+1, 1) = mean_pseudo;
            out.MeanDelta(end+1, 1) = stats.MeanDelta;
            out.StdDelta(end+1, 1) = stats.StdDelta;
            out.SEDelta(end+1, 1) = stats.SEDelta;
            out.LowerMargin(end+1, 1) = lower_margin;
            out.UpperMargin(end+1, 1) = upper_margin;
            out.MarginMode{end+1, 1} = margin_mode;
            out.MarginJustification{end+1, 1} = justification;
            out.TLower(end+1, 1) = stats.TLower;
            out.PLower(end+1, 1) = stats.PLower;
            out.TUpper(end+1, 1) = stats.TUpper;
            out.PUpper(end+1, 1) = stats.PUpper;
            out.TOSTPValue(end+1, 1) = stats.TOSTPValue;
            out.CILevel(end+1, 1) = stats.CILevel;
            out.CILower(end+1, 1) = stats.CILower;
            out.CIUpper(end+1, 1) = stats.CIUpper;
            out.DF(end+1, 1) = stats.DF;
            out.Alpha(end+1, 1) = alpha;
            out.Equivalent(end+1, 1) = double(stats.Equivalent);
        end
    end

    summary = output_struct_to_table(out, include_ecosystem);
end

function [stats, n] = one_sample_tost(x, lower_margin, upper_margin, alpha)
    x = x(:);
    x = x(isfinite(x));

    n = numel(x);
    df = n - 1;

    stats = struct( ...
        'MeanDelta', NaN, ...
        'StdDelta', NaN, ...
        'SEDelta', NaN, ...
        'TLower', NaN, ...
        'PLower', NaN, ...
        'TUpper', NaN, ...
        'PUpper', NaN, ...
        'TOSTPValue', NaN, ...
        'CILevel', 100 * (1 - 2 * alpha), ...
        'CILower', NaN, ...
        'CIUpper', NaN, ...
        'DF', df, ...
        'Equivalent', false ...
    );

    if n == 0
        return;
    end

    stats.MeanDelta = mean(x);

    if n == 1 || ~isfinite(lower_margin) || ~isfinite(upper_margin) || ...
            lower_margin >= upper_margin
        return;
    end

    stats.StdDelta = std(x, 0);

    if stats.StdDelta == 0
        stats.SEDelta = 0;
        stats.CILower = stats.MeanDelta;
        stats.CIUpper = stats.MeanDelta;

        if stats.MeanDelta > lower_margin
            stats.TLower = Inf;
            stats.PLower = 0;
        else
            stats.TLower = -Inf;
            stats.PLower = 1;
        end

        if stats.MeanDelta < upper_margin
            stats.TUpper = -Inf;
            stats.PUpper = 0;
        else
            stats.TUpper = Inf;
            stats.PUpper = 1;
        end

        stats.TOSTPValue = max(stats.PLower, stats.PUpper);
        stats.Equivalent = stats.PLower < alpha && stats.PUpper < alpha;
        return;
    end

    stats.SEDelta = stats.StdDelta / sqrt(n);

    stats.TLower = (stats.MeanDelta - lower_margin) / stats.SEDelta;
    stats.PLower = 1 - student_t_cdf(stats.TLower, df);

    stats.TUpper = (stats.MeanDelta - upper_margin) / stats.SEDelta;
    stats.PUpper = student_t_cdf(stats.TUpper, df);

    stats.TOSTPValue = max(stats.PLower, stats.PUpper);
    stats.Equivalent = stats.PLower < alpha && stats.PUpper < alpha;

    t_crit = student_t_inv_cdf(1 - alpha, df);
    stats.CILower = stats.MeanDelta - t_crit * stats.SEDelta;
    stats.CIUpper = stats.MeanDelta + t_crit * stats.SEDelta;
end

function margins = load_equivalence_margins(margins_file)
    if isempty(margins_file)
        margins = default_delta_equivalence_margins();
        return;
    end

    if isfile(margins_file)
        margins = readtable(margins_file, 'TextType', 'string');
        margins = normalize_margin_table(margins);
        return;
    end

    margins = default_delta_equivalence_margins();
    write_summary_table(margins_file, margins);
    fprintf('[DeltaEquivalence] Wrote default margins to %s\n', margins_file);
end

function margins = normalize_margin_table(margins)
    required = {'Metric', 'LowerMargin', 'UpperMargin', 'MarginMode', 'Justification'};
    missing = setdiff(required, margins.Properties.VariableNames);
    if ~isempty(missing)
        error('write_delta_equivalence_summary:InvalidMarginsFile', ...
            'Margins table is missing columns: %s', strjoin(missing, ', '));
    end

    margins.Metric = cellstr(string(margins.Metric));
    margins.MarginMode = cellstr(string(margins.MarginMode));
    margins.Justification = cellstr(string(margins.Justification));
    margins.LowerMargin = double(margins.LowerMargin);
    margins.UpperMargin = double(margins.UpperMargin);
end

function margin = margin_for_metric(margins, metric_name)
    margin = [];
    if isempty(margins)
        return;
    end

    idx = find(strcmp(margins.Metric, metric_name), 1, 'first');
    if isempty(idx)
        return;
    end

    margin = table2struct(margins(idx, :));
end

function [lower_margin, upper_margin] = resolve_margin(margin, mean_empirical)
    mode = char(string(margin.MarginMode));
    switch lower(mode)
        case 'absolute'
            lower_margin = double(margin.LowerMargin);
            upper_margin = double(margin.UpperMargin);

        case 'relative_mean_empirical'
            scale = abs(double(mean_empirical));
            lower_margin = double(margin.LowerMargin) * scale;
            upper_margin = double(margin.UpperMargin) * scale;

        otherwise
            error('write_delta_equivalence_summary:UnknownMarginMode', ...
                'Unknown margin mode: %s', mode);
    end
end

function groups = build_groups(results, include_ecosystem)
    groups = struct( ...
        'Key', {}, ...
        'Indices', {}, ...
        'EcosystemType', {}, ...
        'K', {}, ...
        'TrainRatio', {}, ...
        'BackboneRatio', {}, ...
        'CvK', {}, ...
        'ThresholdMode', {}, ...
        'Threshold', {}, ...
        'EvaluateOnAllUnseen', {} ...
    );

    for i = 1:numel(results)
        k = scalar_field(results(i), 'K', NaN);
        train_ratio = scalar_field(results(i), 'TrainRatio', NaN);
        backbone_ratio = scalar_field(results(i), 'BackboneRatio', NaN);
        cv_k = scalar_field(results(i), 'CvK', 0);
        threshold_mode = text_field(results(i), 'ThresholdMode', '');
        threshold = scalar_field(results(i), 'Threshold', NaN);
        evaluate_all = scalar_field(results(i), 'EvaluateOnAllUnseen', NaN);
        ecosystem_type = '';
        if include_ecosystem
            ecosystem_type = normalize_ecosystem_type( ...
                text_field(results(i), 'EcosystemType', 'Unknown'));
        end

        key = sprintf('K=%g|Train=%.12g|Backbone=%.12g|CvK=%g|ThresholdMode=%s|Threshold=%.12g|EvalAll=%g', ...
            k, train_ratio, backbone_ratio, cv_k, threshold_mode, threshold, evaluate_all);
        if include_ecosystem
            key = sprintf('%s|Ecosystem=%s', key, ecosystem_type);
        end

        group_idx = find(strcmp({groups.Key}, key), 1, 'first');

        if isempty(group_idx)
            group_idx = numel(groups) + 1;
            groups(group_idx).Key = key;
            groups(group_idx).Indices = i;
            groups(group_idx).EcosystemType = ecosystem_type;
            groups(group_idx).K = k;
            groups(group_idx).TrainRatio = train_ratio;
            groups(group_idx).BackboneRatio = backbone_ratio;
            groups(group_idx).CvK = cv_k;
            groups(group_idx).ThresholdMode = threshold_mode;
            groups(group_idx).Threshold = threshold;
            groups(group_idx).EvaluateOnAllUnseen = evaluate_all;
        else
            groups(group_idx).Indices(end+1) = i;
        end
    end
end

function means = foodweb_level_means(rows, field)
    if isempty(rows)
        means = [];
        return;
    end

    foodwebs = cell(numel(rows), 1);
    for i = 1:numel(rows)
        foodwebs{i} = text_field(rows(i), 'Foodweb', sprintf('row_%d', i));
    end

    unique_foodwebs = unique(foodwebs, 'stable');
    means = NaN(numel(unique_foodwebs), 1);

    for f = 1:numel(unique_foodwebs)
        vals = [];
        for i = 1:numel(rows)
            if strcmp(foodwebs{i}, unique_foodwebs{f})
                vals(end+1, 1) = scalar_field(rows(i), field, NaN); %#ok<AGROW>
            end
        end

        vals = vals(isfinite(vals));
        if ~isempty(vals)
            means(f) = mean(vals);
        end
    end

    means = means(isfinite(means));
end

function value = mean_or_nan(x)
    x = x(:);
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = mean(x);
    end
end

function value = normalize_ecosystem_type(raw_value)
    value = char(strtrim(string(raw_value)));
    if isempty(value) || strcmpi(value, 'nan') || ...
            strcmpi(value, 'missing') || strcmpi(value, '<missing>')
        value = 'Unknown';
    end
end

function p_value = student_t_cdf(t_stat, df)
    if ~isfinite(df) || df <= 0 || isnan(t_stat)
        p_value = NaN;
        return;
    end

    if isinf(t_stat)
        if t_stat > 0
            p_value = 1;
        else
            p_value = 0;
        end
        return;
    end

    x = df / (df + t_stat.^2);
    ib = betainc(x, df / 2, 0.5);

    if t_stat >= 0
        p_value = 1 - 0.5 * ib;
    else
        p_value = 0.5 * ib;
    end

    p_value = max(0, min(1, p_value));
end

function q = student_t_inv_cdf(p, df)
    if ~isfinite(df) || df <= 0 || ~isfinite(p) || p <= 0 || p >= 1
        q = NaN;
        return;
    end

    if abs(p - 0.5) < eps
        q = 0;
        return;
    end

    if p < 0.5
        q = -student_t_inv_cdf(1 - p, df);
        return;
    end

    lo = 0;
    hi = 1;
    while student_t_cdf(hi, df) < p
        hi = hi * 2;
        if hi > 1e6
            q = NaN;
            return;
        end
    end

    q = fzero(@(t) student_t_cdf(t, df) - p, [lo hi]);
end

function value = scalar_field(s, field, default_value)
    if isstruct(s) && isfield(s, field) && ~isempty(s.(field)) && isscalar(s.(field))
        value = double(s.(field));
    else
        value = default_value;
    end
end

function value = text_field(s, field, default_value)
    if isstruct(s) && isfield(s, field) && ~isempty(s.(field))
        value = char(string(s.(field)));
    else
        value = char(string(default_value));
    end
end

function out = empty_output_struct(include_ecosystem)
    out = struct( ...
        'Version', {{}}, ...
        'TestType', {{}}, ...
        'Metric', {{}}, ...
        'K', [], ...
        'TrainRatio', [], ...
        'BackboneRatio', [], ...
        'CvK', [], ...
        'ThresholdMode', {{}}, ...
        'Threshold', [], ...
        'EvaluateOnAllUnseen', [], ...
        'NumFoodWebs', [], ...
        'NumResultRows', [], ...
        'MeanEmpirical', [], ...
        'MeanPseudo', [], ...
        'MeanDelta', [], ...
        'StdDelta', [], ...
        'SEDelta', [], ...
        'LowerMargin', [], ...
        'UpperMargin', [], ...
        'MarginMode', {{}}, ...
        'MarginJustification', {{}}, ...
        'TLower', [], ...
        'PLower', [], ...
        'TUpper', [], ...
        'PUpper', [], ...
        'TOSTPValue', [], ...
        'CILevel', [], ...
        'CILower', [], ...
        'CIUpper', [], ...
        'DF', [], ...
        'Alpha', [], ...
        'Equivalent', [] ...
    );

    if include_ecosystem
        out.EcosystemType = {};
    end
end

function summary = output_struct_to_table(out, include_ecosystem)
    if include_ecosystem
        summary = table( ...
            out.Version, ...
            out.TestType, ...
            out.Metric, ...
            out.EcosystemType, ...
            out.K, ...
            out.TrainRatio, ...
            out.BackboneRatio, ...
            out.CvK, ...
            out.ThresholdMode, ...
            out.Threshold, ...
            out.EvaluateOnAllUnseen, ...
            out.NumFoodWebs, ...
            out.NumResultRows, ...
            out.MeanEmpirical, ...
            out.MeanPseudo, ...
            out.MeanDelta, ...
            out.StdDelta, ...
            out.SEDelta, ...
            out.LowerMargin, ...
            out.UpperMargin, ...
            out.MarginMode, ...
            out.MarginJustification, ...
            out.TLower, ...
            out.PLower, ...
            out.TUpper, ...
            out.PUpper, ...
            out.TOSTPValue, ...
            out.CILevel, ...
            out.CILower, ...
            out.CIUpper, ...
            out.DF, ...
            out.Alpha, ...
            out.Equivalent, ...
            'VariableNames', output_variable_names(true));
    else
        summary = table( ...
            out.Version, ...
            out.TestType, ...
            out.Metric, ...
            out.K, ...
            out.TrainRatio, ...
            out.BackboneRatio, ...
            out.CvK, ...
            out.ThresholdMode, ...
            out.Threshold, ...
            out.EvaluateOnAllUnseen, ...
            out.NumFoodWebs, ...
            out.NumResultRows, ...
            out.MeanEmpirical, ...
            out.MeanPseudo, ...
            out.MeanDelta, ...
            out.StdDelta, ...
            out.SEDelta, ...
            out.LowerMargin, ...
            out.UpperMargin, ...
            out.MarginMode, ...
            out.MarginJustification, ...
            out.TLower, ...
            out.PLower, ...
            out.TUpper, ...
            out.PUpper, ...
            out.TOSTPValue, ...
            out.CILevel, ...
            out.CILower, ...
            out.CIUpper, ...
            out.DF, ...
            out.Alpha, ...
            out.Equivalent, ...
            'VariableNames', output_variable_names(false));
    end
end

function names = output_variable_names(include_ecosystem)
    base = { ...
        'Version', ...
        'TestType', ...
        'Metric', ...
        'K', ...
        'TrainRatio', ...
        'BackboneRatio', ...
        'CvK', ...
        'ThresholdMode', ...
        'Threshold', ...
        'EvaluateOnAllUnseen', ...
        'NumFoodWebs', ...
        'NumResultRows', ...
        'MeanEmpirical', ...
        'MeanPseudo', ...
        'MeanDelta', ...
        'StdDelta', ...
        'SEDelta', ...
        'LowerMargin', ...
        'UpperMargin', ...
        'MarginMode', ...
        'MarginJustification', ...
        'TLower', ...
        'PLower', ...
        'TUpper', ...
        'PUpper', ...
        'TOSTPValue', ...
        'CILevel', ...
        'CILower', ...
        'CIUpper', ...
        'DF', ...
        'Alpha', ...
        'Equivalent' ...
    };

    if include_ecosystem
        names = [base(1:3), {'EcosystemType'}, base(4:end)];
    else
        names = base;
    end
end

function summary = empty_summary_table()
    summary = cell2table(cell(0, numel(output_variable_names(false))), ...
        'VariableNames', output_variable_names(false));
end

function summary = empty_ecosystem_summary_table()
    summary = cell2table(cell(0, numel(output_variable_names(true))), ...
        'VariableNames', output_variable_names(true));
end

function names = extended_structural_metric_names()
    names = { ...
        'DegreeStd', ...
        'DegreeCV', ...
        'DegreeGini', ...
        'GeneralityStd', ...
        'GeneralityCV', ...
        'GeneralityGini', ...
        'VulnerabilityStd', ...
        'VulnerabilityCV', ...
        'VulnerabilityGini', ...
        'TrophicLevelStd', ...
        'TrophicLevelCV', ...
        'TrophicLevelRange', ...
        'NetworkXMeanTrophicLevel', ...
        'MeanLocalClustering', ...
        'Transitivity', ...
        'NumTriangles', ...
        'TriangleDensity', ...
        'MeanDietOverlap' ...
    };
end

function write_summary_table(summary_file, summary)
    out_dir = fileparts(summary_file);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    writetable(summary, summary_file);
end
