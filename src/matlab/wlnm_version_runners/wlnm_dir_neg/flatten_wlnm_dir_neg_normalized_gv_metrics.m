function flat = flatten_wlnm_dir_neg_normalized_gv_metrics(aux)
%FLATTEN_WLNM_DIR_NEG_NORMALIZED_GV_METRICS Build the 20 standard-run fields.
%
% Returns five normalized generality/vulnerability fields for Empirical,
% Train, Pseudo, and Delta. Missing source metric structs remain NaN. Delta
% is always Pseudo minus Empirical, matching the other WLNM_dir_neg fields.

    suffixes = normalized_gv_metric_suffixes();
    prefixes = {'Empirical', 'Train', 'Pseudo', 'Delta'};
    flat = struct();

    for i = 1:numel(prefixes)
        for j = 1:numel(suffixes)
            flat.([prefixes{i} suffixes{j}]) = NaN;
        end
    end

    if nargin == 0 || isempty(aux)
        return;
    end

    sources = {'empirical_metrics', 'train_metrics', 'pseudo_metrics'};
    source_prefixes = {'Empirical', 'Train', 'Pseudo'};
    for i = 1:numel(sources)
        source = sources{i};
        if ~isfield(aux, source) || isempty(aux.(source))
            continue;
        end

        normalized = compute_wlnm_dir_neg_normalized_gv_metrics(aux.(source));
        for j = 1:numel(suffixes)
            suffix = suffixes{j};
            flat.([source_prefixes{i} suffix]) = normalized.(suffix);
        end
    end

    for j = 1:numel(suffixes)
        suffix = suffixes{j};
        flat.(['Delta' suffix]) = ...
            flat.(['Pseudo' suffix]) - flat.(['Empirical' suffix]);
    end
end

function suffixes = normalized_gv_metric_suffixes()
    suffixes = { ...
        'LinkageDensity', ...
        'MeanNormalizedGeneralityConsumersOnly', ...
        'MeanNormalizedVulnerabilityResourcesOnly', ...
        'NormalizedGeneralityStdAllSpecies', ...
        'NormalizedVulnerabilityStdAllSpecies' ...
    };
end
