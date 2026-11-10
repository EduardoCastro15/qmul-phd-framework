function tests = test_wlnm_dir_neg_node_ecology
%TEST_WLNM_DIR_NEG_NODE_ECOLOGY Unit tests for the isolated node route.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    matlab_root = fileparts(fileparts(mfilename('fullpath')));
    testCase.TestData.MatlabRoot = matlab_root;
    addpath(genpath(matlab_root));
end

function testRolesAreDerivedFromTrainingAdjacencyIncludingIsolate(testCase)
    % Orientation is train(resource, consumer)=1.
    train = sparse([1 2], [2 3], 1, 4, 4);
    actual = derive_train_roles_node_ecology(train);

    verifyEqual(testCase, actual, [ ...
        "resource"; ...
        "consumer-resource"; ...
        "consumer"; ...
        "isolate"]);
end

function testMassPreprocessorUsesTrainingNodesOnly(testCase)
    roles = ["resource"; "consumer-resource"; "consumer"; "isolate"];
    masses = [1; exp(2); -999; exp(10)];
    train_node_ids = int32([1 2 3 3 0 0 0 0 0 0]);

    state = fit_node_ecology_preprocessor(masses, roles, train_node_ids);
    expected_scale = sqrt(2 / 3);

    verifyEqual(testCase, state.training_node_ids, [1 2 3]);
    verifyEqual(testCase, state.mass_median, 1, 'AbsTol', 1e-12);
    verifyEqual(testCase, state.mass_center, 1, 'AbsTol', 1e-12);
    verifyEqual(testCase, state.mass_scale, expected_scale, 'AbsTol', 1e-12);
    verifyEqual(testCase, state.feature_dim_per_node, 7);
    verifyEqual(testCase, state.protocol, ...
        'ln_train_subgraph_median_mean_std_v1');

    % A held-out node must not alter any fitted statistic.
    changed_test_mass = masses;
    changed_test_mass(4) = exp(100);
    changed = fit_node_ecology_preprocessor( ...
        changed_test_mass, roles, train_node_ids);
    verifyEqual(testCase, changed.mass_median, state.mass_median, ...
        'AbsTol', 0);
    verifyEqual(testCase, changed.mass_center, state.mass_center, ...
        'AbsTol', 0);
    verifyEqual(testCase, changed.mass_scale, state.mass_scale, ...
        'AbsTol', 0);
end

function testTransformHasFrozenSevenChannelOrder(testCase)
    roles = ["resource"; "consumer-resource"; "consumer"; "isolate"];
    masses = [1; exp(2); -999; exp(10)];
    train_node_ids = int32([1 2 3 0 0 0 0 0 0 0]);
    state = fit_node_ecology_preprocessor(masses, roles, train_node_ids);
    features = transform_node_ecology_features(masses, roles, state);
    scale = sqrt(2 / 3);

    % [mass_z, mass_missing, resource, consumer-resource, consumer,
    %  isolate, real_node]
    expected = [ ...
        -1 / scale, 0, 1, 0, 0, 0, 1; ...
         1 / scale, 0, 0, 1, 0, 0, 1; ...
         0,         1, 0, 0, 1, 0, 1; ...
         9 / scale, 0, 0, 0, 0, 1, 1];

    verifySize(testCase, features, [4 7]);
    verifyEqual(testCase, features, expected, 'AbsTol', 1e-12);
    verifyTrue(testCase, all(isfinite(features(:))));
end

function testAlignmentIsNodeMajorAndPaddingIsZero(testCase)
    node_features = [ ...
         1:7; ...
        11:17; ...
        21:27; ...
        31:37];
    ids = int32([ ...
        3 1 0 2 0 0 0 0 0 0; ...
        4 0 0 0 0 0 0 0 0 0]);

    actual = align_node_ecology_features(ids, node_features);
    expected_first = [node_features(3,:), node_features(1,:), zeros(1,7), ...
        node_features(2,:), zeros(1,42)];
    expected_second = [node_features(4,:), zeros(1,63)];

    verifySize(testCase, actual, [2 70]);
    verifyEqual(testCase, actual(1,:), expected_first, 'AbsTol', 0);
    verifyEqual(testCase, actual(2,:), expected_second, 'AbsTol', 0);
end

function testEncoderReturnsNinetyFeaturesIDsAndMasksWeightedCandidate(testCase)
    % The focal link 1->2 is present. An alternative 1-3-4-2 route keeps
    % the enclosing graph informative after the focal link is hidden.
    adjacency = sparse( ...
        [1 1 1 2 3 3 4 4 5 5 6 6 7 7 8 8 9], ...
        [2 3 4 4 4 5 5 6 6 7 7 8 8 9 9 10 10], ...
        1, 10, 10);
    positive = [1 2];
    negative = zeros(0,2);

    [encoded, labels, canonical_ids, node_counts] = ...
        graph2vector_dir_neg_node_ecology( ...
            positive, negative, adjacency, 10, false, 'node_ecology_test');

    verifySize(testCase, encoded, [1 90]);
    verifyEqual(testCase, labels, 1);
    verifySize(testCase, canonical_ids, [1 10]);
    verifyGreaterThanOrEqual(testCase, node_counts, 2);
    verifyLessThanOrEqual(testCase, node_counts, 10);

    count = node_counts(1);
    ordered_ids = double(canonical_ids(1,1:count));
    verifyEqual(testCase, sort(ordered_ids(1:count)), ...
        sort(unique(ordered_ids)));
    verifyEqual(testCase, canonical_ids(1,count+1:end), ...
        zeros(1,10-count,'int32'));
    resource_position = find(ordered_ids == positive(1), 1);
    consumer_position = find(ordered_ids == positive(2), 1);
    verifyNotEmpty(testCase, resource_position);
    verifyNotEmpty(testCase, consumer_position);

    off_diagonal = ~eye(count);
    target_linear = sub2ind([count count], ...
        resource_position, consumer_position);
    encoded_position = find(find(off_diagonal) == target_linear, 1);
    verifyNotEmpty(testCase, encoded_position);
    % The first structural entry is deliberately set to eps as a legacy
    % file-format sentinel; otherwise the masked value is exactly zero.
    verifyLessThanOrEqual(testCase, ...
        abs(encoded(1,encoded_position)), eps);

    adjacency_without_candidate = adjacency;
    adjacency_without_candidate(positive(1),positive(2)) = 0;
    [encoded_without, labels_without, ids_without, counts_without] = ...
        graph2vector_dir_neg_node_ecology( ...
            positive, negative, adjacency_without_candidate, 10, false, ...
            'node_ecology_test_masked');
    verifyEqual(testCase, encoded, encoded_without, 'AbsTol', 0);
    verifyEqual(testCase, labels, labels_without);
    verifyEqual(testCase, canonical_ids, ids_without);
    verifyEqual(testCase, node_counts, counts_without);
end

function testRouterPreservesHistoricalDefaultAndAddsNodeEcology(testCase)
    registry = get_version_registry();
    historical = resolve_runner(registry, 'WLNM_dir_neg');
    node_ecology = resolve_runner(registry, 'WLNM_dir_neg_node_ecology');

    verifyEqual(testCase, func2str(historical), 'run_wlnm_dir_neg');
    verifyEqual(testCase, func2str(node_ecology), ...
        'run_wlnm_dir_neg_node_ecology');

    main_source = fileread(fullfile(testCase.TestData.MatlabRoot, 'Main.m'));
    default_pattern = '''version''\s*,\s*''WLNM_dir_neg''';
    verifyNotEmpty(testCase, regexp(main_source, default_pattern, 'once'));
end

function testFrozenScopeAcceptsProtocolAndRejectsMassOnly(testCase)
    fixed_protocol = @() validate_dir_neg_node_ecology_scope( ...
        'WLNM_dir_neg_node_ecology', 10, false, 'fixed', 0.50, false, ...
        'role_only', 2, 'uniform_without_replacement', ...
        'uniform_remaining_nonlinks', false);
    verifyWarningFree(testCase, fixed_protocol);

    verifyError(testCase, @() validate_dir_neg_node_ecology_scope( ...
        'WLNM_dir_neg_node_ecology', 10, false, 'fixed', 0.50, false, ...
        'mass_only', 2, 'uniform_without_replacement', ...
        'uniform_remaining_nonlinks', false), ...
        'WLNMDirNegNodeEcology:NegativeProtocol');
end

function testNodeEcologyCsvSchemaIsConditionalAndWritesFields(testCase)
    output_dir = tempname;
    mkdir(output_dir);
    cleanup = onCleanup(@() rmdir(output_dir, 's')); %#ok<NASGU>
    node_log = fullfile(output_dir, 'node_ecology.csv');
    baseline_log = fullfile(output_dir, 'baseline.csv');

    init_log_file(node_log, false, false, 'WLNM_dir_neg_node_ecology');
    result = struct( ...
        'Version','WLNM_dir_neg_node_ecology', ...
        'TrainRatio',0.60, ...
        'NodeEcologyFeatureDim',70, ...
        'ModelInputDim',160, ...
        'MassPreprocessingProtocol','ln_train_subgraph_median_mean_std_v1', ...
        'ValidMassNodeCount',8, ...
        'MissingMassNodeCount',2, ...
        'TrainingNodeCount',7, ...
        'NegativeSplitHash',repmat('b',1,64), ...
        'NegativeRoleSource','train_positive_adjacency_v1', ...
        'NodeAlignmentProtocol','canonical_node_order_node_major_padding0_v1', ...
        'TargetMaskingProtocol','binary_and_weighted_candidate_mask_v1');
    append_results(node_log, result, false);

    output = readtable(node_log, 'VariableNamingRule','preserve');
    verifyEqual(testCase, height(output), 1);
    verifyEqual(testCase, output.NodeEcologyFeatureDim, 70);
    verifyEqual(testCase, output.ModelInputDim, 160);
    verifyEqual(testCase, output.TrainingNodeCount, 7);
    verifyEqual(testCase, string(output.MassPreprocessingProtocol), ...
        "ln_train_subgraph_median_mean_std_v1");
    verifyEqual(testCase, string(output.NegativeRoleSource), ...
        "train_positive_adjacency_v1");
    verifyEqual(testCase, string(output.NodeAlignmentProtocol), ...
        "canonical_node_order_node_major_padding0_v1");
    verifyEqual(testCase, string(output.TargetMaskingProtocol), ...
        "binary_and_weighted_candidate_mask_v1");
    verifyEqual(testCase, strlength(string(output.NegativeSplitHash)), 64);

    init_log_file(baseline_log, false, false, 'WLNM_dir_neg');
    baseline = readtable(baseline_log, 'VariableNamingRule','preserve');
    verifyFalse(testCase, ismember('NodeEcologyFeatureDim', ...
        baseline.Properties.VariableNames));
    verifyFalse(testCase, ismember('NegativeRoleSource', ...
        baseline.Properties.VariableNames));
    verifyFalse(testCase, ismember('NodeAlignmentProtocol', ...
        baseline.Properties.VariableNames));
end
