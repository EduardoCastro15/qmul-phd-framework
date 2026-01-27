function folds = kfold_edges_dir(net, k, seed, backbone_mask, stratify)
% Assign each positive directed edge to a fold in 1..k.
% Optional stratification by backbone/non-backbone.

    if nargin < 2 || isempty(k), k = 5; end
    if nargin < 3 || isempty(seed), seed = 1; end
    if nargin < 4, backbone_mask = []; end
    if nargin < 5, stratify = false; end

    rng(seed, 'twister');

    net = sparse(net);
    n   = size(net, 1);
    net = net - spdiags(diag(net), 0, n, n);

    [i, j] = find(net);
    m = numel(i);

    if m == 0
        folds = struct('i', i, 'j', j, 'fold_id', zeros(0,1), 'k', 0, 'n', n);
        return;
    end

    k = min(k, m);
    fold_id = zeros(m, 1);

    if stratify && ~isempty(backbone_mask)
        B = logical(sparse(backbone_mask));
        B = B & (net > 0);

        lin = sub2ind([n,n], i, j);
        is_bb = full(B(lin));

        idx_bb = find(is_bb);
        idx_nb = find(~is_bb);

        fold_id(idx_bb) = assign_balanced(numel(idx_bb), k);
        fold_id(idx_nb) = assign_balanced(numel(idx_nb), k);
    else
        fold_id = assign_balanced(m, k);
    end

    folds = struct('i', i, 'j', j, 'fold_id', fold_id, 'k', k, 'n', n);
end

function fold_id = assign_balanced(m, k)
    perm = randperm(m);
    fold_id = zeros(m,1);

    base = floor(m / k);
    remn = mod(m, k);

    s = 1;
    for f = 1:k
        c = base + (f <= remn);
        sel = perm(s:(s+c-1));
        fold_id(sel) = f;
        s = s + c;
    end
end
