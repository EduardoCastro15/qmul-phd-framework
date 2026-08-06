function tests = test_wlnm_original_undirected_split
%TEST_WLNM_ORIGINAL_UNDIRECTED_SPLIT Validate legacy WLNM_original splitting.
    tests = functiontests(localfunctions);
end

function testLowerTriangleDirectedEdgeIsDiscarded(testCase)
    % Historical WLNM_original keeps only the supplied upper triangle.
    net = sparse([3, 4], [1, 2], 1, 4, 4);

    [train, test] = DivideNet_original( ...
        net, 1.0, "random", true, false, false);

    verifyEqual(testCase, nnz(train), 0);
    verifyEqual(testCase, nnz(test), 0);
    verifyEqual(testCase, train, train');
    verifyEqual(testCase, nnz(diag(train)), 0);
end

function testUpperTriangleIsRetainedAndSelfLoopsAreRemoved(testCase)
    net = sparse([1, 2, 4, 3], [2, 3, 1, 3], 1, 4, 4);

    [train, test] = DivideNet_original( ...
        net, 1.0, "random", true, false, false);

    expectedUpper = sparse([1, 2], [2, 3], 1, 4, 4);
    expected = expectedUpper + expectedUpper';

    verifyEqual(testCase, train, expected);
    verifyEqual(testCase, nnz(test), 0);
    verifyEqual(testCase, train, train');
    verifyEqual(testCase, nnz(diag(train)), 0);
end

function testTrainTestAreDisjointAndCoverLegacyProjection(testCase)
    net = sparse([4, 1, 3], [1, 2, 2], 1, 4, 4);
    legacyUpper = triu(net, 1);
    expected = legacyUpper + legacyUpper';

    rng(12345, 'twister');
    [train, test] = DivideNet_original( ...
        net, 0.5, "random", true, false, false);

    verifyEqual(testCase, nnz(train & test), 0);
    verifyEqual(testCase, spones(train + test), expected);
    verifyEqual(testCase, train, train');
    verifyEqual(testCase, test, test');
    verifyEqual(testCase, ...
        nnz(triu(train, 1)) + nnz(triu(test, 1)), ...
        nnz(legacyUpper));
end
