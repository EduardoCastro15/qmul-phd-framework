function normalized = compute_wlnm_dir_neg_normalized_gv_metrics(metrics)
%COMPUTE_WLNM_DIR_NEG_NORMALIZED_GV_METRICS Normalized food-web G/V metrics.
%
% This helper is intentionally owned by the standard WLNM_dir_neg runner.
% It does not change compute_foodweb_metrics, WLNM_original, or the k-fold
% runner. The input metrics must come from compute_foodweb_metrics, which
% removes self-links and uses A(resource, consumer) = 1.
%
% For S species, L off-diagonal links, d = L/S, resources consumed g_i,
% and consumers of resource i v_i:
%   G_i = g_i/d
%   V_i = v_i/d
%
% The positive-only means implement the requested consumer/resource
% interpretation. The all-species standard deviations implement the
% Williams and Martinez (2000) GenSD and VulSD interpretation.

    normalized = empty_normalized_metrics();

    required_fields = {'NumSpecies', 'NumLinks', 'Generality', 'Vulnerability'};
    for i = 1:numel(required_fields)
        field = required_fields{i};
        if ~isstruct(metrics) || ~isfield(metrics, field)
            error('WLNMDirNegNormalizedGV:MissingField', ...
                'Input metrics is missing required field %s.', field);
        end
    end

    S = double(metrics.NumSpecies);
    L = double(metrics.NumLinks);
    generality = double(metrics.Generality(:));
    vulnerability = double(metrics.Vulnerability(:));

    if ~isscalar(S) || ~isfinite(S) || S < 0 || floor(S) ~= S
        error('WLNMDirNegNormalizedGV:InvalidNumSpecies', ...
            'NumSpecies must be one finite non-negative integer.');
    end
    if ~isscalar(L) || ~isfinite(L) || L < 0 || floor(L) ~= L
        error('WLNMDirNegNormalizedGV:InvalidNumLinks', ...
            'NumLinks must be one finite non-negative integer.');
    end
    if numel(generality) ~= S || numel(vulnerability) ~= S
        error('WLNMDirNegNormalizedGV:VectorLengthMismatch', ...
            ['Generality and Vulnerability must each contain NumSpecies ' ...
             'values.']);
    end
    if any(~isfinite(generality)) || any(generality < 0) || ...
            any(~isfinite(vulnerability)) || any(vulnerability < 0)
        error('WLNMDirNegNormalizedGV:InvalidNodeCounts', ...
            'Generality and Vulnerability must contain finite non-negative values.');
    end
    if abs(sum(generality) - L) > 1e-9 || ...
            abs(sum(vulnerability) - L) > 1e-9
        error('WLNMDirNegNormalizedGV:InconsistentLinkCounts', ...
            ['The per-species generality and vulnerability counts must each ' ...
             'sum to NumLinks.']);
    end

    if S == 0
        return;
    end

    linkage_density = L / S;
    normalized.LinkageDensity = linkage_density;

    if linkage_density == 0
        return;
    end

    normalized_generality = generality / linkage_density;
    normalized_vulnerability = vulnerability / linkage_density;

    consumer_mask = generality > 0;
    resource_mask = vulnerability > 0;

    if any(consumer_mask)
        normalized.MeanNormalizedGeneralityConsumersOnly = ...
            mean(normalized_generality(consumer_mask));
    end
    if any(resource_mask)
        normalized.MeanNormalizedVulnerabilityResourcesOnly = ...
            mean(normalized_vulnerability(resource_mask));
    end

    normalized.NormalizedGeneralityStdAllSpecies = ...
        sample_std(normalized_generality);
    normalized.NormalizedVulnerabilityStdAllSpecies = ...
        sample_std(normalized_vulnerability);
end

function normalized = empty_normalized_metrics()
    normalized = struct( ...
        'LinkageDensity', NaN, ...
        'MeanNormalizedGeneralityConsumersOnly', NaN, ...
        'MeanNormalizedVulnerabilityResourcesOnly', NaN, ...
        'NormalizedGeneralityStdAllSpecies', NaN, ...
        'NormalizedVulnerabilityStdAllSpecies', NaN);
end

function value = sample_std(values)
    if numel(values) <= 1
        value = 0;
    else
        value = std(values, 0);
    end
end
