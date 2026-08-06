function tests = test_sample_neg_dir_neg_role_only
%TEST_SAMPLE_NEG_DIR_NEG_ROLE_ONLY Verify the frozen role-only protocol.
    tests = functiontests(localfunctions);
end

function testRolePoolIsSampledWithoutReplacement(testCase)
    [train, test, role, mass] = toy_network();

    rng(11, 'twister');
    [train_pos, train_neg, test_pos, test_neg, diagnostics] = sample_neg_dir_neg( ...
        train, test, role, 2, 1, false, true, mass, false, 1.0, ...
        'uniform_remaining_nonlinks', 'uniform_without_replacement');

    selected = [train_neg; test_neg];
    observed = [train_pos; test_pos];
    expected_role_pool = [2, 3; 3, 1];

    verifyEqual(testCase, sortrows(selected), sortrows(expected_role_pool));
    verifyEqual(testCase, size(unique(selected, 'rows'), 1), size(selected, 1));
    verifyFalse(testCase, any(ismember(selected, observed, 'rows')));
    verifyFalse(testCase, any(selected(:, 1) == selected(:, 2)));
    verifyEqual(testCase, diagnostics.EligibilityMode, 'role_only');
    verifyEqual(testCase, diagnostics.NegativePositiveRatio, 2);
    verifyEqual(testCase, diagnostics.RolePoolSize, 2);
    verifyEqual(testCase, diagnostics.EligiblePoolSize, 2);
    verifyEqual(testCase, diagnostics.RandomTopupCount, 0);
    verifyEqual(testCase, diagnostics.TopupProportion, 0);
end

function testRandomTopupUsesRemainingNonlinks(testCase)
    [train, test, role, mass] = toy_network();

    rng(12, 'twister');
    [~, train_neg, ~, test_neg, diagnostics] = sample_neg_dir_neg( ...
        train, test, role, 3, 1, false, true, mass, false, 1.0, ...
        'uniform_remaining_nonlinks', 'uniform_without_replacement');

    selected = [train_neg; test_neg];
    expected_role_pool = [2, 3; 3, 1];

    verifyEqual(testCase, size(selected, 1), 3);
    verifyTrue(testCase, all(ismember(expected_role_pool, selected, 'rows')));
    verifyEqual(testCase, size(unique(selected, 'rows'), 1), 3);
    verifyEqual(testCase, diagnostics.EligibleNegativeCount, 2);
    verifyEqual(testCase, diagnostics.EligibleShortfall, 1);
    verifyEqual(testCase, diagnostics.RandomTopupCount, 1);
    verifyEqual(testCase, diagnostics.TopupProportion, 1 / 3, 'AbsTol', 1e-12);
end

function testTopupCanBeDisabledExplicitly(testCase)
    [train, test, role, mass] = toy_network();

    call = @() sample_neg_dir_neg( ...
        train, test, role, 3, 1, false, true, mass, false, 1.0, ...
        'error', 'uniform_without_replacement');

    verifyError(testCase, call, 'sample_neg_dir_neg:EligiblePoolShortfall');
end

function testSamplingIsDeterministicForFixedSeed(testCase)
    [train, test, role, mass] = toy_network();

    rng(99, 'twister');
    [~, train_neg_a, ~, test_neg_a] = sample_neg_dir_neg( ...
        train, test, role, 3, 1, false, true, mass, false, 1.0);
    rng(99, 'twister');
    [~, train_neg_b, ~, test_neg_b] = sample_neg_dir_neg( ...
        train, test, role, 3, 1, false, true, mass, false, 1.0);

    verifyEqual(testCase, train_neg_a, train_neg_b);
    verifyEqual(testCase, test_neg_a, test_neg_b);
end

function testCanonicalModeOverridesLegacyMassFlag(testCase)
    protocol = resolve_negative_sampling_protocol( ...
        'role_only', 2, 'uniform_without_replacement', ...
        'uniform_remaining_nonlinks', true, true);

    verifyEqual(testCase, protocol.eligibility_mode, 'role_only');
    verifyTrue(testCase, protocol.use_role_filter);
    verifyFalse(testCase, protocol.use_mass_constraint);
end

function [train, test, role, mass] = toy_network()
    % One positive leaves five directed non-links. Exactly two satisfy the
    % configured role constraints: 2 -> 3 and 3 -> 1.
    train = sparse(2, 1, 1, 3, 3);
    test = sparse(3, 3);
    role = {'resource'; 'consumer'; 'consumer-resource'};
    mass = [10; 1; NaN];
end
