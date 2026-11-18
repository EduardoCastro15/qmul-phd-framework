function tests = test_compute_pr_auc
% Frozen references for non-interpolated average precision.
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    matlab_root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(matlab_root));
end

function testPerfectDistinctRanking(testCase)
    labels = [1; 1; 0; 0];
    scores = [0.9; 0.8; 0.2; 0.1];
    verifyEqual(testCase, compute_pr_auc(labels, scores), 1, ...
        'AbsTol', 1e-12);
end

function testPerfectTiedRanking(testCase)
    labels = [1; 1; 0; 0];
    scores = [1; 1; 0; 0];
    verifyEqual(testCase, compute_pr_auc(labels, scores), 1, ...
        'AbsTol', 1e-12);
end

function testReversedRankingMatchesSklearnReference(testCase)
    labels = [1; 1; 0; 0];
    scores = [0.1; 0.2; 0.8; 0.9];
    verifyEqual(testCase, compute_pr_auc(labels, scores), 5/12, ...
        'AbsTol', 1e-12);
end

function testConstantScoresEqualPositivePrevalence(testCase)
    labels = [1; 0; 0];
    scores = 0.5 * ones(3, 1);
    verifyEqual(testCase, compute_pr_auc(labels, scores), 1/3, ...
        'AbsTol', 1e-12);
end

function testMixedTopTieMatchesSklearnReference(testCase)
    labels = [1; 1; 0; 0];
    scores = [0.9; 0.9; 0.9; 0.1];
    verifyEqual(testCase, compute_pr_auc(labels, scores), 2/3, ...
        'AbsTol', 1e-12);
end

function testSklearnDocumentationExample(testCase)
    labels = [0; 0; 1; 1];
    scores = [0.1; 0.4; 0.35; 0.8];
    verifyEqual(testCase, compute_pr_auc(labels, scores), 5/6, ...
        'AbsTol', 1e-12);
end

function testOrderingWithinTieDoesNotMatter(testCase)
    scores = [0.8; 0.8; 0.8; 0.2];
    first = compute_pr_auc([1; 0; 1; 0], scores);
    second = compute_pr_auc([0; 1; 1; 0], scores);
    verifyEqual(testCase, first, second, 'AbsTol', 1e-12);
end

function testSingleClassReturnsNaN(testCase)
    verifyTrue(testCase, isnan(compute_pr_auc(ones(5,1), (1:5)')));
    verifyTrue(testCase, isnan(compute_pr_auc(zeros(5,1), (1:5)')));
end

function testInvalidInputsAreRejected(testCase)
    verifyError(testCase, @() compute_pr_auc([0; 1], 0.5), ...
        'WLNMMetrics:InvalidPRAUCInput');
    verifyError(testCase, @() compute_pr_auc([0; 2], [0.1; 0.9]), ...
        'WLNMMetrics:InvalidPRAUCInput');
    verifyError(testCase, @() compute_pr_auc([0; 1], [0.1; NaN]), ...
        'WLNMMetrics:InvalidPRAUCInput');
end
