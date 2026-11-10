function metrics = compute_dir_neg_foodweb_metrics(A, protocol, highPrecision)
% Preserve all shared ecological calculations; replace only NetworkX trophic fields.
    metrics = compute_foodweb_metrics(A);
    if strcmp(protocol,'legacy_v1'), return; end
    legacy = metrics;
    [levels,stats] = compute_networkx_trophic_levels_v2(A,highPrecision);
    metrics.NetworkXTrophicLevel = levels;
    finiteLevels = levels(isfinite(levels));
    metrics.NetworkXMeanTrophicLevel = NaN;
    metrics.NetworkXTrophicLevelStd = NaN;
    metrics.NetworkXTrophicLevelRange = NaN;
    if ~isempty(finiteLevels)
        metrics.NetworkXMeanTrophicLevel = mean(finiteLevels);
        metrics.NetworkXTrophicLevelStd = std(finiteLevels);
        metrics.NetworkXTrophicLevelRange = max(finiteLevels)-min(finiteLevels);
    end
    fields = {'NumSpeciesFull','NumSpeciesLargest','NumSpeciesWithLevel','LargestFraction','StatusCode'};
    for k=1:numel(fields)
        metrics.(['NetworkXTrophicLevel' fields{k}]) = stats.(fields{k});
    end
    stats.LegacyMean = legacy.NetworkXMeanTrophicLevel;
    stats.LegacyStatusCode = legacy.NetworkXTrophicLevelStatusCode;
    metrics.TrophicV2Diagnostics = stats;
end
