function [features, diagnostics] = encode_focal_mass_features(pairs, profile, mode)
%ENCODE_FOCAL_MASS_FEATURES Encode ordered resource-consumer focal pairs.
% pairs(:,1) is resource; pairs(:,2) is consumer.

    mode = normalize_focal_mass_feature_mode(mode);
    if isempty(pairs)
        pair_count = 0;
    else
        if size(pairs, 2) ~= 2
            error('WLNMDirNegFocalMass:InvalidPairs', ...
                'Candidate pairs must be an N-by-2 matrix.');
        end
        pair_count = size(pairs, 1);
    end

    feature_dim = focal_mass_feature_dim(mode);
    if feature_dim == 0
        features = zeros(pair_count, 0);
        diagnostics = make_diagnostics(pairs, profile, feature_dim);
        return;
    end

    resource = pairs(:, 1);
    consumer = pairs(:, 2);
    n = numel(profile.standardized);
    if any(resource < 1 | resource > n | consumer < 1 | consumer > n | ...
            resource ~= floor(resource) | consumer ~= floor(consumer))
        error('WLNMDirNegFocalMass:PairIndexOutOfRange', ...
            'Candidate pair indices must be integer node indices in [1,%d].', n);
    end

    resource_z = profile.standardized(resource);
    consumer_z = profile.standardized(consumer);
    resource_missing = profile.missing(resource);
    consumer_missing = profile.missing(consumer);
    ratio_missing = resource_missing | consumer_missing;
    log_ratio_z = consumer_z - resource_z;
    log_ratio_z(ratio_missing) = 0;

    switch mode
        case 'endpoint_masses'
            features = [resource_z, consumer_z, ...
                double(resource_missing), double(consumer_missing)];
        case 'mass_ratio'
            features = [log_ratio_z, double(ratio_missing)];
        case 'endpoint_masses_ratio'
            features = [resource_z, consumer_z, ...
                double(resource_missing), double(consumer_missing), log_ratio_z];
        otherwise
            error('WLNMDirNegFocalMass:UnsupportedFeatureMode', ...
                'Unsupported feature mode: %s.', mode);
    end

    if size(features, 2) ~= feature_dim || any(~isfinite(features(:)))
        error('WLNMDirNegFocalMass:InvalidFeatureMatrix', ...
            'Focal-mass feature construction produced invalid output.');
    end
    diagnostics = make_diagnostics(pairs, profile, feature_dim);
end

function value = focal_mass_feature_dim(mode)
    switch mode
        case 'structure_only'
            value = 0;
        case 'endpoint_masses'
            value = 4;
        case 'mass_ratio'
            value = 2;
        case 'endpoint_masses_ratio'
            value = 5;
        otherwise
            value = NaN;
    end
end

function diagnostics = make_diagnostics(pairs, profile, feature_dim)
    if isempty(pairs)
        both_valid = 0;
    else
        both_valid = sum(~profile.missing(pairs(:,1)) & ...
            ~profile.missing(pairs(:,2)));
    end
    diagnostics = struct( ...
        'FeatureDim', double(feature_dim), ...
        'PairCount', double(size(pairs, 1)), ...
        'BothMassValidCount', double(both_valid) ...
    );
end
