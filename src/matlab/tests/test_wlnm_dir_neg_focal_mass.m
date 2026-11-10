function tests = test_wlnm_dir_neg_focal_mass
%TEST_WLNM_DIR_NEG_FOCAL_MASS Unit and regression tests for the new route.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    matlab_root = fileparts(fileparts(mfilename('fullpath')));
    testCase.TestData.MatlabRoot = matlab_root;
    addpath(genpath(matlab_root));
end

function testFrozenMassPreprocessingAndFeatureOrder(testCase)
    profile = prepare_focal_mass_profile([1; 10; 100; -999; 0; NaN; Inf], 7);
    verifyEqual(testCase, profile.valid_count, 3);
    verifyEqual(testCase, profile.missing_count, 4);
    verifyEqual(testCase, profile.standardized(4:7), zeros(4,1));
    verifyTrue(testCase, all(profile.missing(4:7)));
    verifyEqual(testCase, profile.protocol, 'log10_within_web_median_iqr_v1');

    pairs = [1 3; 4 2];
    [endpoint, endpoint_diag] = encode_focal_mass_features( ...
        pairs, profile, 'endpoint_masses');
    verifyEqual(testCase, endpoint(1,:), ...
        [profile.standardized(1), profile.standardized(3), 0, 0], ...
        'AbsTol', 1e-12);
    verifyEqual(testCase, endpoint(2,:), ...
        [0, profile.standardized(2), 1, 0], 'AbsTol', 1e-12);
    verifyEqual(testCase, endpoint_diag.FeatureDim, 4);
    verifyEqual(testCase, endpoint_diag.BothMassValidCount, 1);

    [ratio, ratio_diag] = encode_focal_mass_features(pairs, profile, 'mass_ratio');
    verifyEqual(testCase, ratio(1,1), ...
        profile.standardized(3) - profile.standardized(1), 'AbsTol', 1e-12);
    verifyEqual(testCase, ratio(1,2), 0);
    verifyEqual(testCase, ratio(2,:), [0 1]);
    verifyEqual(testCase, ratio_diag.FeatureDim, 2);

    combined = encode_focal_mass_features(pairs, profile, 'endpoint_masses_ratio');
    verifyEqual(testCase, combined(:,1:4), endpoint, 'AbsTol', 1e-12);
    verifyEqual(testCase, combined(:,5), ratio(:,1), 'AbsTol', 1e-12);
end

function testFallbacksAndDimensions(testCase)
    missing_profile = prepare_focal_mass_profile([-999; 0; NaN; Inf], 4);
    verifyEqual(testCase, missing_profile.log10_median, 0);
    verifyEqual(testCase, missing_profile.log10_iqr, 1);
    verifyEqual(testCase, missing_profile.standardized, zeros(4,1));

    constant_profile = prepare_focal_mass_profile(10 * ones(4,1), 4);
    verifyEqual(testCase, constant_profile.log10_iqr, 0);
    verifyEqual(testCase, constant_profile.scale, 1);
    verifyEqual(testCase, constant_profile.standardized, zeros(4,1));

    modes = {'structure_only','endpoint_masses','mass_ratio','endpoint_masses_ratio'};
    extra_dims = [0 4 2 5];
    total_dims = [90 94 92 95];
    for i = 1:numel(modes)
        features = encode_focal_mass_features([1 2], constant_profile, modes{i});
        verifySize(testCase, features, [1 extra_dims(i)]);
        verifyEqual(testCase, 90 + size(features,2), total_dims(i));
    end
end

function testOrderedResourceConsumerRatio(testCase)
    profile = prepare_focal_mass_profile([1; 100; 10], 3);
    forward = encode_focal_mass_features([1 2], profile, 'mass_ratio');
    reverse = encode_focal_mass_features([2 1], profile, 'mass_ratio');
    verifyEqual(testCase, forward(1), -reverse(1), 'AbsTol', 1e-12);
    verifyGreaterThan(testCase, forward(1), 0);
end

function testCopiedEncoderMatchesHistoricalEncoder(testCase)
    adjacency = sparse([1 1 2 3 4 5], [2 3 4 4 5 1], 1, 5, 5);
    positives = [1 2; 2 4];
    negatives = [2 5; 4 1];
    [historical_data, historical_labels] = graph2vector_dir_neg( ...
        positives, negatives, adjacency, 4, false, 'encoder_parity', false);
    [new_data, new_labels] = graph2vector_dir_neg_focal_mass( ...
        positives, negatives, adjacency, 4, false, 'encoder_parity', false);
    verifyEqual(testCase, new_data, historical_data, 'AbsTol', 0);
    verifyEqual(testCase, new_labels, historical_labels);
end

function testRouterAndHistoricalDefault(testCase)
    registry = get_version_registry();
    historical = resolve_runner(registry, 'WLNM_dir_neg');
    focal = resolve_runner(registry, 'WLNM_dir_neg_focal_mass');
    verifyEqual(testCase, func2str(historical), 'run_wlnm_dir_neg');
    verifyEqual(testCase, func2str(focal), 'run_wlnm_dir_neg_focal_mass');
end

function testFrozenNegativeProtocolRejectsMassEligibility(testCase)
    verifyError(testCase, @() validate_dir_neg_focal_mass_scope( ...
        'WLNM_dir_neg_focal_mass', 'legacy_v1', 'off', 'mass_only', 2, ...
        'uniform_without_replacement', 'uniform_remaining_nonlinks', true), ...
        'WLNMDirNegFocalMass:Protocol');
    verifyError(testCase, @() normalize_focal_mass_feature_mode('ecosystem_type'), ...
        'WLNMDirNegFocalMass:InvalidFeatureMode');
end

function testModelProtocolRejectsNonTenK(testCase)
    verifyError(testCase, @() WLNM_dir_neg_focal_mass( ...
        'toy', sparse(3,3), sparse(3,3), 9, {'a','b','c'}, ...
        ones(3,1), cell(3,1), 'random', 0.60), ...
        'WLNMDirNegFocalMass:SubgraphSize');
end

function testFocalMassCsvSchemaIsConditional(testCase)
    output_dir = tempname;
    mkdir(output_dir);
    cleanup = onCleanup(@() rmdir(output_dir, 's')); %#ok<NASGU>
    focal_log = fullfile(output_dir, 'focal.csv');
    old_log = fullfile(output_dir, 'old.csv');

    init_log_file(focal_log, false, false, 'WLNM_dir_neg_focal_mass');
    result = struct( ...
        'Version','WLNM_dir_neg_focal_mass', 'TrainRatio',0.6, ...
        'FocalMassFeatureMode','endpoint_masses_ratio', ...
        'FocalMassFeatureDim',5, 'ModelInputDim',95, ...
        'MassPreprocessingProtocol','log10_within_web_median_iqr_v1', ...
        'ValidMassNodeCount',7, 'MissingMassNodeCount',2, ...
        'MassLog10Median',1.5, 'MassLog10IQR',0.75, ...
        'TrainPairsBothMassValid',20, 'TestPairsBothMassValid',8, ...
        'PositiveSplitHash',repmat('a',1,64), ...
        'NegativeSplitHash',repmat('b',1,64), ...
        'TargetMaskingProtocol','historical_weighted_v1');
    append_results(focal_log, result, false);
    output = readtable(focal_log, 'VariableNamingRule','preserve');
    verifyEqual(testCase, string(output.FocalMassFeatureMode), "endpoint_masses_ratio");
    verifyEqual(testCase, output.ModelInputDim, 95);
    verifyEqual(testCase, strlength(string(output.NegativeSplitHash)), 64);

    init_log_file(old_log, false, false, 'WLNM_dir_neg');
    old_output = readtable(old_log, 'VariableNamingRule','preserve');
    verifyFalse(testCase, ismember('FocalMassFeatureMode', ...
        old_output.Properties.VariableNames));
    verifyFalse(testCase, ismember('NegativeSplitHash', ...
        old_output.Properties.VariableNames));
end

function testHashIsOrderInvariantWithinEachSplit(testCase)
    hash_a = make_focal_mass_link_split_hash(5, [1 2; 3 4], [2 5; 4 1]);
    hash_b = make_focal_mass_link_split_hash(5, [3 4; 1 2], [4 1; 2 5]);
    verifyEqual(testCase, hash_a, hash_b);
    verifyEqual(testCase, strlength(string(hash_a)), 64);
end
