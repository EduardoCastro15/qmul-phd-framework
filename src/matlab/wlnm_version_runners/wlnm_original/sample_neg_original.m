function [train_pos, train_neg, test_pos, test_neg] = sample_neg_original(train, test, a, portion, evaluate_on_all_unseen, use_role_filter)
    %SAMPLE_NEG_ORIGINAL Negative sampling for the original WLNM pipeline.
    %
    % The original WLNM path works on the upper triangular half of an
    % undirected graph. The common path now samples only the requested number
    % of negative pairs by rejection instead of materializing the full upper
    % triangular complement.

    if nargin < 3 || isempty(a), a = 1; end
    if nargin < 4 || isempty(portion), portion = 1; end
    if nargin < 5 || isempty(evaluate_on_all_unseen), evaluate_on_all_unseen = false; end
    if nargin < 6 || isempty(use_role_filter), use_role_filter = false; end

    train = sparse(triu(train, 1));
    test = sparse(triu(test, 1));

    [i, j] = find(train);
    train_pos = [i, j];
    train_size = size(train_pos, 1);

    [i, j] = find(test);
    test_pos = [i, j];
    test_size = size(test_pos, 1);

    if nnz(train & test) ~= 0
        error('sample_neg_original:Overlap', 'Train and test must not overlap.');
    end

    net = spones(train + test);
    pool_size = upper_negative_pool_size(net);
    pos_total = train_size + test_size;
    need_total_requested = floor(a * pos_total);

    if pool_size < need_total_requested
        warning('Not enough negative links available. Reducing the sample size.');
    end

    need_total = min(need_total_requested, pool_size);

    if pool_size == 0 || need_total == 0
        warning('[sample_neg_original] No negatives available. Returning empties.');
        train_neg = zeros(0, 2);
        test_neg = zeros(0, 2);
        return;
    end

    if evaluate_on_all_unseen
        neg_links = enumerate_upper_negative_links(net);
        pool_size = size(neg_links, 1);

        k_train = min(floor(a * train_size), pool_size);
        idx_train = randperm(pool_size, k_train);
        train_neg = neg_links(idx_train, :);

        keep_test = true(pool_size, 1);
        keep_test(idx_train) = false;
        test_neg = neg_links(keep_test, :);
    else
        [k_train, k_test] = split_negative_counts_original( ...
            need_total, train_size, test_size, floor(a * train_size), floor(a * test_size));

        neg_links = sample_upper_negative_links(net, k_train + k_test, pool_size);
        train_neg = neg_links(1:k_train, :);
        test_neg = neg_links(k_train+1:end, :);
    end

    % Sample a portion of the links if specified
    if portion < 1
        train_pos = train_pos(1:min(size(train_pos,1), ceil(size(train_pos, 1) * portion)), :);
        train_neg = train_neg(1:min(size(train_neg,1), ceil(size(train_neg, 1) * portion)), :);
        test_pos = test_pos(1:min(size(test_pos,1), ceil(size(test_pos, 1) * portion)), :);
        test_neg = test_neg(1:min(size(test_neg,1), ceil(size(test_neg, 1) * portion)), :);
    elseif portion > 1
        train_pos = train_pos(1:min(size(train_pos,1), portion), :);
        train_neg = train_neg(1:min(size(train_neg,1), portion), :);
        test_pos = test_pos(1:min(size(test_pos,1), portion), :);
        test_neg = test_neg(1:min(size(test_neg,1), portion), :);
    end

    fprintf('[NegPool] pool=%d need_total=%d a=%g eval_all=%d role_filter=%d | k_train=%d k_test=%d\n', ...
        pool_size, need_total, a, logical(evaluate_on_all_unseen), logical(use_role_filter), ...
        size(train_neg,1), size(test_neg,1));

    fprintf('[sample_neg] Final link counts (use_role_filter = %d):\n', logical(use_role_filter));
    fprintf('    Train Positive: %d\n', size(train_pos, 1));
    fprintf('    Train Negative: %d\n', size(train_neg, 1));
    fprintf('    Test  Positive: %d\n', size(test_pos, 1));
    fprintf('    Test  Negative: %d\n', size(test_neg, 1));
end

function pool_size = upper_negative_pool_size(net)
    n = size(net, 1);
    upper_positive = nnz(triu(net, 1));
    pool_size = n * max(0, n - 1) / 2 - upper_positive;
    pool_size = max(0, double(pool_size));
end

function [k_train, k_test] = split_negative_counts_original(need_total, train_size, test_size, k_train_target, k_test_target)
    if k_train_target + k_test_target <= need_total
        k_train = k_train_target;
        k_test = k_test_target;
        return;
    end

    ratio = train_size / max(1, train_size + test_size);
    k_train = floor(ratio * need_total);
    k_test = need_total - k_train;

    if train_size > 0 && test_size > 0 && need_total >= 2
        if k_train == 0 && k_test > 1
            k_train = 1;
            k_test = need_total - 1;
        elseif k_test == 0 && k_train > 1
            k_test = 1;
            k_train = need_total - 1;
        end
    end
end

function neg_links = sample_upper_negative_links(net, k, pool_size)
    if k <= 0
        neg_links = zeros(0, 2);
        return;
    end

    if k > 0.25 * pool_size
        pool = enumerate_upper_negative_links(net);
        neg_links = pool(randperm(size(pool, 1), k), :);
        return;
    end

    n = size(net, 1);
    lin = zeros(0, 1);
    max_rounds = 25;

    for round_id = 1:max_rounds
        remaining = k - numel(lin);
        if remaining <= 0
            break;
        end

        batch_size = max(1000, ceil(remaining * 2.5));
        cand = draw_upper_pairs(n, batch_size);
        cand_lin = sub2ind([n, n], cand(:,1), cand(:,2));
        cand_lin = cand_lin(net(cand_lin) == 0);
        if isempty(cand_lin)
            continue;
        end

        lin = unique([lin; cand_lin(:)], 'stable');

        if round_id > 5 && numel(lin) < 0.5 * k
            break;
        end
    end

    if numel(lin) < k
        pool = enumerate_upper_negative_links(net);
        pool_lin = sub2ind([n, n], pool(:,1), pool(:,2));
        pool = pool(~ismember(pool_lin, lin), :);
        extra = pool(randperm(size(pool, 1), k - numel(lin)), :);

        [i, j] = ind2sub([n, n], lin);
        neg_links = [[i(:), j(:)]; extra];
    else
        lin = lin(1:k);
        [i, j] = ind2sub([n, n], lin);
        neg_links = [i(:), j(:)];
    end
end

function pairs = draw_upper_pairs(n, batch_size)
    i = randi(n, batch_size, 1);
    j = randi(n, batch_size, 1);

    same = i == j;
    while any(same)
        j(same) = randi(n, sum(same), 1);
        same = i == j;
    end

    pairs = [min(i, j), max(i, j)];
end

function neg_links = enumerate_upper_negative_links(net)
    n = size(net, 1);
    [i, j] = find(triu(true(n), 1));
    lin = sub2ind([n, n], i, j);
    keep = net(lin) == 0;
    neg_links = [i(keep), j(keep)];
end
