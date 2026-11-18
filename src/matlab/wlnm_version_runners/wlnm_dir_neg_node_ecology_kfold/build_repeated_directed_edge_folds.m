function folds = build_repeated_directed_edge_folds(net, k, seed)
%BUILD_REPEATED_DIRECTED_EDGE_FOLDS Balanced directed-positive CV folds.
% Each positive off-diagonal edge is assigned exactly once to one fold.

    if nargin < 2 || isempty(k), k = 5; end
    if nargin < 3 || isempty(seed), seed = 1; end
    if ~isscalar(k) || ~isfinite(k) || k < 2 || floor(k) ~= k
        error('WLNMDirNegNodeEcologyKfold:InvalidK', ...
            'k must be an integer greater than or equal to two.');
    end
    if ~isscalar(seed) || ~isfinite(seed) || seed < 1 || floor(seed) ~= seed
        error('WLNMDirNegNodeEcologyKfold:InvalidSeed', ...
            'seed must be one positive integer.');
    end

    net = spones(sparse(net));
    n = size(net, 1);
    if size(net, 2) ~= n
        error('WLNMDirNegNodeEcologyKfold:NonSquareAdjacency', ...
            'The adjacency matrix must be square.');
    end
    net = net - spdiags(diag(net), 0, n, n);
    net = spones(net);
    [i, j] = find(net);
    m = numel(i);
    if m < k
        error('WLNMDirNegNodeEcologyKfold:InsufficientEdges', ...
            'Requested %d folds for only %d positive edges.', k, m);
    end

    previous_rng = rng;
    cleanup = onCleanup(@() rng(previous_rng));
    rng(double(seed), 'twister');
    permutation = randperm(m);
    fold_id = zeros(m, 1);
    base = floor(m / k);
    remainder = mod(m, k);
    cursor = 1;
    for fold = 1:k
        count = base + double(fold <= remainder);
        selected = permutation(cursor:(cursor + count - 1));
        fold_id(selected) = fold;
        cursor = cursor + count;
    end

    if any(fold_id < 1) || numel(fold_id) ~= m
        error('WLNMDirNegNodeEcologyKfold:IncompletePartition', ...
            'Every positive edge must be assigned to exactly one fold.');
    end
    counts = accumarray(fold_id, 1, [k 1]);
    if max(counts) - min(counts) > 1
        error('WLNMDirNegNodeEcologyKfold:UnbalancedPartition', ...
            'Fold sizes must differ by at most one edge.');
    end

    folds = struct( ...
        'i', i, ...
        'j', j, ...
        'fold_id', fold_id, ...
        'k', k, ...
        'n', n, ...
        'seed', double(seed));
end
