function tests = test_sample_neg_dir_neg_observed_zero
%TEST_SAMPLE_NEG_DIR_NEG_OBSERVED_ZERO Freeze Yujie's explicit-zero protocol.
    tests = functiontests(localfunctions);
end

function testOnlyObservedZerosAreSampledAtExactTwoToOne(testCase)
    [train, test, role, observed, candidate] = toy_network();

    rng(77, 'twister');
    [train_pos, train_neg, test_pos, test_neg, diagnostics] = sample_neg_dir_neg( ...
        train, test, role, 2, 1, false, false, NaN(5,1), false, 1.0, ...
        'error', 'uniform_without_replacement', 'observed_zero', observed, candidate);

    selected = [train_neg; test_neg];
    [zero_i, zero_j] = find(observed);
    explicit_zeroes = [zero_i, zero_j];

    verifyEqual(testCase, sortrows(selected), sortrows(explicit_zeroes));
    verifyEqual(testCase, size(unique(selected, 'rows'), 1), size(selected, 1));
    verifyEqual(testCase, size(train_neg, 1), 2 * size(train_pos, 1));
    verifyEqual(testCase, size(test_neg, 1), 2 * size(test_pos, 1));
    verifyFalse(testCase, ismember([3 4], selected, 'rows'), ...
        'An unresolved candidate must not be sampled.');
    verifyFalse(testCase, ismember([5 1], selected, 'rows'), ...
        'A pair outside candidate_mask must not be sampled.');
    verifyEqual(testCase, diagnostics.EligibilityMode, 'observed_zero');
    verifyEqual(testCase, diagnostics.CandidatePairCount, nnz(candidate));
    verifyEqual(testCase, diagnostics.ObservedZeroPoolSize, nnz(observed));
    verifyEqual(testCase, diagnostics.RequestedNegativeCount, 8);
    verifyEqual(testCase, diagnostics.SelectedNegativeCount, 8);
    verifyEqual(testCase, diagnostics.RandomTopupCount, 0);
    verifyEqual(testCase, diagnostics.TrainTestNegativeOverlapCount, 0);
    verifyEqual(testCase, diagnostics.PositiveTrainTestOverlapCount, 0);
    verifyEqual(testCase, diagnostics.TestPositiveOneEndpointVisible, 1);
    verifyEqual(testCase, diagnostics.TestPositiveNeitherEndpointVisible, 1);
end

function testFixedSeedIsReproducible(testCase)
    [train, test, role, observed, candidate] = toy_network_with_extra_zeroes();

    rng(1234, 'twister');
    [~, train_a, ~, test_a] = sample_once(train, test, role, observed, candidate);
    rng(1234, 'twister');
    [~, train_b, ~, test_b] = sample_once(train, test, role, observed, candidate);

    verifyEqual(testCase, train_a, train_b);
    verifyEqual(testCase, test_a, test_b);
end

function testPositiveSplitAndNegativesAreReproducibleForSeed(testCase)
    [train_seed, test_seed, role, observed, candidate] = toy_network();
    full_net = spones(train_seed + test_seed);

    rng(2468, 'twister');
    [train_a, test_a] = DivideNet_dir_neg(full_net, 0.5, false, false);
    [~, train_neg_a, ~, test_neg_a] = sample_once( ...
        train_a, test_a, role, observed, candidate);

    rng(2468, 'twister');
    [train_b, test_b] = DivideNet_dir_neg(full_net, 0.5, false, false);
    [~, train_neg_b, ~, test_neg_b] = sample_once( ...
        train_b, test_b, role, observed, candidate);

    verifyEqual(testCase, train_a, train_b);
    verifyEqual(testCase, test_a, test_b);
    verifyEqual(testCase, train_neg_a, train_neg_b);
    verifyEqual(testCase, test_neg_a, test_neg_b);
end

function testInsufficientObservedPoolFailsWithoutFallback(testCase)
    [train, test, role, observed, candidate] = toy_network();
    observed(3, 2) = 0;

    call = @() sample_once(train, test, role, observed, candidate);
    verifyError(testCase, call, 'sample_neg_dir_neg:ObservedZeroPoolShortfall');
end

function testObservedZeroRejectsFallbackPolicy(testCase)
    [train, test, role, observed, candidate] = toy_network();
    call = @() sample_neg_dir_neg( ...
        train, test, role, 2, 1, false, false, NaN(5,1), false, 1.0, ...
        'uniform_remaining_nonlinks', 'uniform_without_replacement', ...
        'observed_zero', observed, candidate);
    verifyError(testCase, call, ...
        'resolve_negative_sampling_protocol:ObservedZeroRequiresErrorTopup');
end

function testMasksMustBeDisjointAndInsideCandidateDomain(testCase)
    [train, test, role, observed, candidate] = toy_network();
    outside = observed;
    outside(5, 1) = 1;
    verifyError(testCase, ...
        @() sample_once(train, test, role, outside, candidate), ...
        'sample_neg_dir_neg:ObservedZeroOutsideCandidateMask');

    overlap = observed;
    overlap(1, 4) = 1;
    candidate(1, 4) = 1;
    verifyError(testCase, ...
        @() sample_once(train, test, role, overlap, candidate), ...
        'sample_neg_dir_neg:ObservedZeroPositiveOverlap');
end

function testEvaluateAllUnseenIsRejected(testCase)
    [train, test, role, observed, candidate] = toy_network();
    call = @() sample_neg_dir_neg( ...
        train, test, role, 2, 1, true, false, NaN(5,1), false, 1.0, ...
        'error', 'uniform_without_replacement', 'observed_zero', observed, candidate);
    verifyError(testCase, call, ...
        'sample_neg_dir_neg:ObservedZeroEvaluateAllUnsupported');
end

function [train_pos, train_neg, test_pos, test_neg, diagnostics] = ...
        sample_once(train, test, role, observed, candidate)
    [train_pos, train_neg, test_pos, test_neg, diagnostics] = sample_neg_dir_neg( ...
        train, test, role, 2, 1, false, false, NaN(5,1), false, 1.0, ...
        'error', 'uniform_without_replacement', 'observed_zero', observed, candidate);
end

function [train, test, role, observed, candidate] = toy_network()
    n = 5;
    train = sparse([1 2], [4 4], 1, n, n);
    test = sparse([3 4], [5 5], 1, n, n);
    zeroes = [1 2; 1 3; 1 5; 2 1; 2 3; 2 5; 3 1; 3 2];
    observed = sparse(zeroes(:,1), zeroes(:,2), 1, n, n);
    candidate = spones(train + test + observed + sparse(3, 4, 1, n, n));
    role = {'resource'; 'resource'; 'resource'; 'consumer'; 'consumer'};
end

function [train, test, role, observed, candidate] = toy_network_with_extra_zeroes()
    [train, test, role, observed, candidate] = toy_network();
    observed(4, 1) = 1;
    observed(5, 2) = 1;
    candidate = spones(candidate + observed);
end
