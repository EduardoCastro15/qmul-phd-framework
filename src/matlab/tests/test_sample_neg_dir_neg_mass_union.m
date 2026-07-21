function tests = test_sample_neg_dir_neg_mass_union
%TEST_SAMPLE_NEG_DIR_NEG_MASS_UNION Verify role OR mass eligibility sampling.
    tests = functiontests(localfunctions);
end

function testMassOnlyCandidateJoinsEligiblePool(testCase)
    [train, test, role, mass] = toy_network();

    rng(1, 'twister');
    output = evalc(['[~, train_neg, ~, test_neg] = sample_neg_dir_neg(' ...
        'train, test, role, 3, 1, false, true, mass, true, 1.0);']);

    selected = [train_neg; test_neg];
    verifyEqual(testCase, size(selected, 1), 3);
    verifyTrue(testCase, ismember([1, 2], selected, 'rows'), ...
        'The non-role candidate satisfying the mass rule must be eligible.');
    verifySubstring(testCase, output, 'eligibility=role_or_mass');
    verifySubstring(testCase, output, 'mass_pool=1 eligible_pool=3');
    verifySubstring(testCase, output, 'eligible_shortfall=0');
    verifySubstring(testCase, output, 'eligible_neg=3 random_topup=0');
end

function testTopupUsesNeitherConstraintPool(testCase)
    [train, test, role, mass] = toy_network();

    rng(2, 'twister');
    output = evalc(['[~, train_neg, ~, test_neg] = sample_neg_dir_neg(' ...
        'train, test, role, 4, 1, false, true, mass, true, 1.0);']);

    selected = [train_neg; test_neg];
    verifyEqual(testCase, size(selected, 1), 4);
    verifySubstring(testCase, output, 'eligible_pool=3');
    verifySubstring(testCase, output, 'eligible_shortfall=1');
    verifySubstring(testCase, output, 'eligible_neg=3 random_topup=1');
end

function testEvaluateAllUnseenEnumeratesUnionOnly(testCase)
    [train, test, role, mass] = toy_network();

    rng(3, 'twister');
    output = evalc(['[~, train_neg, ~, test_neg] = sample_neg_dir_neg(' ...
        'train, test, role, 1, 1, true, true, mass, true, 1.0);']);

    selected = [train_neg; test_neg];
    verifyEqual(testCase, size(selected, 1), 3);
    verifyTrue(testCase, ismember([1, 2], selected, 'rows'));
    verifySubstring(testCase, output, 'eval_all=1');
    verifySubstring(testCase, output, 'eligible_neg=3 random_topup=0');
end

function [train, test, role, mass] = toy_network()
    % One positive leaves five directed non-links. Of those, two satisfy the
    % role rules and one additional candidate, 1 -> 2, satisfies only mass.
    train = sparse(2, 1, 1, 3, 3);
    test = sparse(3, 3);
    role = {'resource'; 'consumer'; 'consumer-resource'};
    mass = [10; 1; NaN];
end
