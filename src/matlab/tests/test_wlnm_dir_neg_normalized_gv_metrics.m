function tests = test_wlnm_dir_neg_normalized_gv_metrics
%TEST_WLNM_DIR_NEG_NORMALIZED_GV_METRICS Verify Athen and W&M definitions.
    tests = functiontests(localfunctions);
end

function testAsymmetricWebMatchesPerSpeciesDefinitions(testCase)
    % A(resource, consumer)=1. Generality=[0 0 1 3 0] and
    % vulnerability=[2 1 1 0 0].
    A = sparse([1 1 2 3], [3 4 4 4], 1, 5, 5);
    raw = compute_foodweb_metrics(A);
    actual = compute_wlnm_dir_neg_normalized_gv_metrics(raw);

    expected_density = 4 / 5;
    expected_g = [0; 0; 1; 3; 0] / expected_density;
    expected_v = [2; 1; 1; 0; 0] / expected_density;

    verifyEqual(testCase, actual.LinkageDensity, expected_density, 'AbsTol', 1e-12);
    verifyEqual(testCase, actual.MeanNormalizedGeneralityConsumersOnly, ...
        mean(expected_g(expected_g > 0)), 'AbsTol', 1e-12);
    verifyEqual(testCase, actual.MeanNormalizedVulnerabilityResourcesOnly, ...
        mean(expected_v(expected_v > 0)), 'AbsTol', 1e-12);
    verifyEqual(testCase, actual.NormalizedGeneralityStdAllSpecies, ...
        std(expected_g, 0), 'AbsTol', 1e-12);
    verifyEqual(testCase, actual.NormalizedVulnerabilityStdAllSpecies, ...
        std(expected_v, 0), 'AbsTol', 1e-12);

    % W&M normalization invariant across all S species.
    verifyEqual(testCase, mean(expected_g), 1, 'AbsTol', 1e-12);
    verifyEqual(testCase, mean(expected_v), 1, 'AbsTol', 1e-12);

    % Positive-only equivalences are validation identities, not the
    % implementation used by the helper.
    verifyEqual(testCase, actual.MeanNormalizedGeneralityConsumersOnly, ...
        raw.NumSpecies / nnz(raw.Generality > 0), 'AbsTol', 1e-12);
    verifyEqual(testCase, actual.MeanNormalizedVulnerabilityResourcesOnly, ...
        raw.NumSpecies / nnz(raw.Vulnerability > 0), 'AbsTol', 1e-12);
    verifyEqual(testCase, actual.NormalizedGeneralityStdAllSpecies, ...
        raw.GeneralityStd / expected_density, 'AbsTol', 1e-12);
    verifyEqual(testCase, actual.NormalizedVulnerabilityStdAllSpecies, ...
        raw.VulnerabilityStd / expected_density, 'AbsTol', 1e-12);
end

function testSelfLinksDoNotChangeMetrics(testCase)
    A = sparse([1 1 2 3], [3 4 4 4], 1, 5, 5);
    with_self_links = A + speye(5);

    without_loops = compute_wlnm_dir_neg_normalized_gv_metrics( ...
        compute_foodweb_metrics(A));
    with_loops = compute_wlnm_dir_neg_normalized_gv_metrics( ...
        compute_foodweb_metrics(with_self_links));

    verifyEqual(testCase, with_loops, without_loops, 'AbsTol', 1e-12);
end

function testRunnerFlatteningComputesPrefixesAndDeltas(testCase)
    empirical_A = sparse([1 1 2 3], [3 4 4 4], 1, 5, 5);
    train_A = sparse([1 1 2], [3 4 4], 1, 5, 5);
    pseudo_A = sparse([1 1 2 2 3], [3 4 3 4 4], 1, 5, 5);

    aux = struct( ...
        'empirical_metrics', compute_foodweb_metrics(empirical_A), ...
        'train_metrics', compute_foodweb_metrics(train_A), ...
        'pseudo_metrics', compute_foodweb_metrics(pseudo_A));

    actual = flatten_wlnm_dir_neg_normalized_gv_metrics(aux);

    suffixes = { ...
        'LinkageDensity', ...
        'MeanNormalizedGeneralityConsumersOnly', ...
        'MeanNormalizedVulnerabilityResourcesOnly', ...
        'NormalizedGeneralityStdAllSpecies', ...
        'NormalizedVulnerabilityStdAllSpecies' ...
    };
    sources = {'empirical_metrics', 'train_metrics', 'pseudo_metrics'};
    prefixes = {'Empirical', 'Train', 'Pseudo'};

    for i = 1:numel(sources)
        expected = compute_wlnm_dir_neg_normalized_gv_metrics(aux.(sources{i}));
        for j = 1:numel(suffixes)
            suffix = suffixes{j};
            verifyEqual(testCase, actual.([prefixes{i} suffix]), ...
                expected.(suffix), 'AbsTol', 1e-12);
        end
    end

    for j = 1:numel(suffixes)
        suffix = suffixes{j};
        verifyEqual(testCase, actual.(['Delta' suffix]), ...
            actual.(['Pseudo' suffix]) - actual.(['Empirical' suffix]), ...
            'AbsTol', 1e-12);
    end

    % Smoke-test the values through the real header-driven CSV writer.
    output_dir = tempname;
    mkdir(output_dir);
    cleanup = onCleanup(@() rmdir(output_dir, 's')); %#ok<NASGU>
    log_file = fullfile(output_dir, 'normalized_metrics.csv');
    init_log_file(log_file, false, false, 'WLNM_dir_neg');
    actual.Version = 'WLNM_dir_neg';
    actual.TrainRatio = 0.60;
    append_results(log_file, actual, false);
    output = readtable(log_file, 'VariableNamingRule', 'preserve');

    for i = 1:numel(prefixes)
        for j = 1:numel(suffixes)
            column = [prefixes{i} suffixes{j}];
            verifyEqual(testCase, output{1, column}, actual.(column), ...
                'AbsTol', 1e-10);
        end
    end
    for j = 1:numel(suffixes)
        column = ['Delta' suffixes{j}];
        verifyEqual(testCase, output{1, column}, actual.(column), ...
            'AbsTol', 1e-10);
    end
end

function testZeroLinkAndZeroSpeciesCasesAreFiniteSafe(testCase)
    zero_link_raw = compute_foodweb_metrics(sparse(5, 5));
    zero_link = compute_wlnm_dir_neg_normalized_gv_metrics(zero_link_raw);

    verifyEqual(testCase, zero_link.LinkageDensity, 0);
    verifyTrue(testCase, isnan(zero_link.MeanNormalizedGeneralityConsumersOnly));
    verifyTrue(testCase, isnan(zero_link.MeanNormalizedVulnerabilityResourcesOnly));
    verifyTrue(testCase, isnan(zero_link.NormalizedGeneralityStdAllSpecies));
    verifyTrue(testCase, isnan(zero_link.NormalizedVulnerabilityStdAllSpecies));

    zero_species_raw = struct( ...
        'NumSpecies', 0, ...
        'NumLinks', 0, ...
        'Generality', zeros(0, 1), ...
        'Vulnerability', zeros(0, 1));
    zero_species = compute_wlnm_dir_neg_normalized_gv_metrics(zero_species_raw);

    values = [ ...
        zero_species.LinkageDensity, ...
        zero_species.MeanNormalizedGeneralityConsumersOnly, ...
        zero_species.MeanNormalizedVulnerabilityResourcesOnly, ...
        zero_species.NormalizedGeneralityStdAllSpecies, ...
        zero_species.NormalizedVulnerabilityStdAllSpecies ...
    ];
    verifyTrue(testCase, all(isnan(values)));
    verifyFalse(testCase, any(isinf(values)));
end
