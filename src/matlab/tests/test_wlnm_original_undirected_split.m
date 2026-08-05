function tests = test_wlnm_original_undirected_split
%TEST_WLNM_ORIGINAL_UNDIRECTED_SPLIT Validate directed-to-undirected conversion.
    tests = functiontests(localfunctions);
end

function testLowerTriangleDirectedEdgeIsNotDiscarded(testCase)
    % Both directed links lie below the diagonal. A direct triu(A) would
    % discard them, whereas the undirected baseline must retain both pairs.
    net = sparse([3, 4], [1, 2], 1, 4, 4);

    [train, test] = DivideNet_original( ...
        net, 1.0, "random", true, false, false);

    expected = sparse([1, 3, 2, 4], [3, 1, 4, 2], 1, 4, 4);

    verifyEqual(testCase, train, expected);
    verifyEqual(testCase, nnz(test), 0);
    verifyEqual(testCase, train, train');
    verifyEqual(testCase, nnz(diag(train)), 0);
end

function testTrainTestAreDisjointAndCoverSymmetrizedGraph(testCase)
    net = sparse([4, 1, 3], [1, 2, 2], 1, 4, 4);
    expected = spones(net + net');

    rng(12345, 'twister');
    [train, test] = DivideNet_original( ...
        net, 0.5, "random", true, false, false);

    verifyEqual(testCase, nnz(train & test), 0);
    verifyEqual(testCase, spones(train + test), expected);
    verifyEqual(testCase, train, train');
    verifyEqual(testCase, test, test');
    verifyEqual(testCase, nnz(triu(train, 1)), 1);
    verifyEqual(testCase, nnz(triu(test, 1)), 2);
end

