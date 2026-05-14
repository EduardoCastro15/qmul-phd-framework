function [train_pos, train_neg, test_pos, test_neg] = sample_neg_directed(train, test, role, a, portion, evaluate_on_all_unseen, use_role_filter)
    %SAMPLE_NEG_DIRECTED Sample directed negative links for WLNM_directed.
    %
    % The common path samples only the requested number of directed
    % non-links by rejection, avoiding materializing the full n-by-n
    % complement. Full enumeration is kept for evaluate_on_all_unseen and
    % for fallback paths where it is faster or unavoidable.

    if nargin < 3, role = []; end
    if nargin < 4 || isempty(a), a = 1; end
    if nargin < 5 || isempty(portion), portion = 1; end
    if nargin < 6 || isempty(evaluate_on_all_unseen), evaluate_on_all_unseen = false; end
    if nargin < 7 || isempty(use_role_filter), use_role_filter = false; end

    train = sparse(train);
    test = sparse(test);
    n = size(train, 1);

    [i, j] = find(train);
    train_pos = [i, j];
    train_size = size(train_pos, 1);

    [i, j] = find(test);
    test_pos = [i, j];
    test_size = size(test_pos, 1);

    if nnz(train & test) ~= 0
        error('sample_neg_directed:Overlap', 'Train and test must not overlap.');
    end

    net = spones(train + test);
    net = net - spdiags(diag(net), 0, n, n);
    net = spones(net);

    role_code = encode_roles_directed(role, n);
    requested_role_filter = logical(use_role_filter);
    effective_role_filter = requested_role_filter;

    pos_total = train_size + test_size;
    need_total_requested = floor(a * pos_total);
    pool_size = negative_pool_size_directed(net, role_code, effective_role_filter);

    if pool_size < need_total_requested && requested_role_filter
        warning('[sample_neg_directed] Pool %d < need %d with role filter. Disabling role filter.', ...
            pool_size, need_total_requested);
        effective_role_filter = false;
        pool_size = negative_pool_size_directed(net, role_code, false);
    end

    if pool_size < need_total_requested
        warning('Not enough negative links available. Reducing the sample size.');
    end

    need_total = min(need_total_requested, pool_size);

    if pool_size == 0 || need_total == 0
        warning('[sample_neg_directed] No negatives available. Returning empties.');
        train_neg = zeros(0, 2);
        test_neg = zeros(0, 2);
        return;
    end

    if evaluate_on_all_unseen
        neg_links = enumerate_negative_links_directed(net, role_code, effective_role_filter);
        pool_size = size(neg_links, 1);

        k_train = min(floor(a * train_size), pool_size);
        idx_train = randperm(pool_size, k_train);
        train_neg = neg_links(idx_train, :);

        keep_test = true(pool_size, 1);
        keep_test(idx_train) = false;
        test_neg = neg_links(keep_test, :);
    else
        [k_train, k_test] = split_negative_counts_directed( ...
            need_total, train_size, test_size, floor(a * train_size), floor(a * test_size));

        neg_links = sample_negative_links_directed( ...
            net, role_code, effective_role_filter, k_train + k_test, pool_size);
        train_neg = neg_links(1:k_train, :);
        test_neg = neg_links(k_train+1:end, :);
    end

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
        pool_size, need_total, a, logical(evaluate_on_all_unseen), effective_role_filter, ...
        size(train_neg,1), size(test_neg,1));

    fprintf('[sample_neg] Final link counts (use_role_filter = %d):\n', effective_role_filter);
    fprintf('    Train Positive: %d\n', size(train_pos, 1));
    fprintf('    Train Negative: %d\n', size(train_neg, 1));
    fprintf('    Test  Positive: %d\n', size(test_pos, 1));
    fprintf('    Test  Negative: %d\n', size(test_neg, 1));
end

function role_code = encode_roles_directed(role, n)
    role_code = zeros(n, 1);
    if isempty(role)
        return;
    end

    role_str = lower(string(role(:)));
    upto = min(n, numel(role_str));
    role_code(1:upto) = double(role_str(1:upto) == "consumer") + ...
        2 * double(role_str(1:upto) == "resource");
end

function pool_size = negative_pool_size_directed(net, role_code, use_role_filter)
    n = size(net, 1);

    if use_role_filter
        num_consumers = sum(role_code == 1);
        num_resources = sum(role_code == 2);
        total_candidates = num_consumers * max(0, num_consumers - 1) + ...
            num_resources * max(0, num_resources - 1);

        [pi, pj] = find(net);
        positive_is_candidate = is_valid_role_pair_directed(pi, pj, role_code);
        pool_size = total_candidates - sum(positive_is_candidate);
    else
        pool_size = n * max(0, n - 1) - nnz(net);
    end

    pool_size = max(0, double(pool_size));
end

function tf = is_valid_role_pair_directed(i, j, role_code)
    src = role_code(i);
    tgt = role_code(j);
    tf = (src == tgt) & (src == 1 | src == 2);
end

function [k_train, k_test] = split_negative_counts_directed(need_total, train_size, test_size, k_train_target, k_test_target)
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

function neg_links = sample_negative_links_directed(net, role_code, use_role_filter, k, pool_size)
    if k <= 0
        neg_links = zeros(0, 2);
        return;
    end

    if k > 0.25 * pool_size
        pool = enumerate_negative_links_directed(net, role_code, use_role_filter);
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
        cand = draw_candidate_pairs_directed(n, role_code, use_role_filter, batch_size);
        if isempty(cand)
            break;
        end

        cand = cand(cand(:,1) ~= cand(:,2), :);
        if isempty(cand)
            continue;
        end

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
        pool = enumerate_negative_links_directed(net, role_code, use_role_filter);
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

function cand = draw_candidate_pairs_directed(n, role_code, use_role_filter, batch_size)
    if ~use_role_filter
        cand = [randi(n, batch_size, 1), randi(n, batch_size, 1)];
        return;
    end

    consumer_nodes = find(role_code == 1);
    resource_nodes = find(role_code == 2);

    consumer_pairs = numel(consumer_nodes) * max(0, numel(consumer_nodes) - 1);
    resource_pairs = numel(resource_nodes) * max(0, numel(resource_nodes) - 1);
    total_pairs = consumer_pairs + resource_pairs;

    if total_pairs == 0
        cand = zeros(0, 2);
        return;
    end

    use_consumer = rand(batch_size, 1) < (consumer_pairs / total_pairs);
    cand = zeros(batch_size, 2);

    n_consumer = sum(use_consumer);
    if n_consumer > 0
        cand(use_consumer, :) = random_pairs_from_group_directed(consumer_nodes, n_consumer);
    end

    n_resource = batch_size - n_consumer;
    if n_resource > 0
        cand(~use_consumer, :) = random_pairs_from_group_directed(resource_nodes, n_resource);
    end
end

function pairs = random_pairs_from_group_directed(nodes, count)
    m = numel(nodes);
    pairs = [nodes(randi(m, count, 1)), nodes(randi(m, count, 1))];
end

function neg_links = enumerate_negative_links_directed(net, role_code, use_role_filter)
    n = size(net, 1);

    if use_role_filter
        groups = {find(role_code == 1), find(role_code == 2)};
        chunks = cell(1, numel(groups));

        for g = 1:numel(groups)
            nodes = groups{g};
            if numel(nodes) < 2
                chunks{g} = zeros(0, 2);
                continue;
            end

            [src, tgt] = ndgrid(nodes, nodes);
            keep = src ~= tgt;
            src = src(keep);
            tgt = tgt(keep);
            lin = sub2ind([n, n], src, tgt);
            keep = net(lin) == 0;
            chunks{g} = [src(keep), tgt(keep)];
        end

        neg_links = vertcat(chunks{:});
    else
        [src, tgt] = ndgrid(1:n, 1:n);
        keep = src ~= tgt;
        src = src(keep);
        tgt = tgt(keep);
        lin = sub2ind([n, n], src, tgt);
        keep = net(lin) == 0;
        neg_links = [src(keep), tgt(keep)];
    end
end
