function [train_pos, train_neg, test_pos, test_neg] = sample_neg_negative(train, test, role, a, portion, evaluate_on_all_unseen, use_role_filter)
    %  Usage: to sample negative links for train and test datasets. When sampling negative train links, assume all testing
    %         links are known and thus sample negative train links only from other unknown links. Set evaluate_on_all_unseen
    %         to true to do link prediction on all links unseen during training.
    %  --Input--
    %       -train: half train positive adjacency matrix
    %       -test: half test positive adjacency matrix
    %       -a: how many times of negative links (w.r.t. pos links) to sample
    %       -portion: if specified, only a portion of the sampled train and test links be returned
    %       -evaluate_on_all_unseen: if true, will not randomly sample negative testing links, but regard all links unseen
    %                          during training as neg testing links; train negative links are sampled in the original way
    %       - consumers: indices of consumer nodes
    %       - resources: indices of resource nodes
    %  --Output--
    %       - train_pos, train_neg: training positive and negative links
    %       - test_pos, test_neg: testing positive and negative links

    % Get all positive links
    [train_i, train_j] = find(train);
    train_pos = [train_i, train_j];
    train_size = length(train_i);

    [test_i, test_j] = find(test);
    test_pos = [test_i, test_j];
    test_size = length(test_i);

    % Combine train and test to find all edges
    if isempty(test)
        net = train;
    else
        net = train + test;
    end

    % Ensure train and test do not overlap
    assert(max(max(net)) == 1);

    % Get all potential negative links (non-existent links)
    neg_net = triu(-(net - 1), 1);
    [neg_i, neg_j] = find(neg_net);
    neg_links = [neg_i, neg_j];

    % Keep only neg links where both endpoints share the same role: consumer-consumer or resource-resource.
    if use_role_filter && ~isempty(role)
        valid_mask = false(size(neg_links, 1), 1);
        for idx = 1:size(neg_links, 1)
            u = neg_links(idx, 1);
            v = neg_links(idx, 2);

            ru = lower(string(role{u}));
            rv = lower(string(role{v}));

            if (ru == "consumer" && rv == "consumer") || (ru == "resource" && rv == "resource")
                valid_mask(idx) = true;
            end
        end
        neg_links_filtered = neg_links(valid_mask, :);
        if ~isempty(neg_links_filtered)
            neg_links = neg_links_filtered;
        else
            warning('[sample_neg_negative] No valid role-filtered negatives. Using unfiltered set.');
        end
    end

    % --- compute needs and cap to pool size ---
    pool_size = size(neg_links, 1);

    % Ensure enough negative links are available
    total_neg_links_needed = a * (train_size + test_size);
    if size(neg_links, 1) < total_neg_links_needed
        warning('Not enough negative links available. Reducing sample size.');
        a = size(neg_links, 1) / (train_size + test_size);
    end

    % Sample negative links for train and test
    perm = randperm(size(neg_links, 1));
    train_neg = neg_links(perm(1:a * train_size), :);
    test_neg = neg_links(perm(a * train_size + 1:a * (train_size + test_size)), :);

    % Sample a portion of links if specified
    if portion < 1
        train_pos = train_pos(1:ceil(size(train_pos, 1) * portion), :);
        train_neg = train_neg(1:ceil(size(train_neg, 1) * portion), :);
        test_pos = test_pos(1:ceil(size(test_pos, 1) * portion), :);
        test_neg = test_neg(1:ceil(size(test_neg, 1) * portion), :);
    elseif portion > 1
        train_pos = train_pos(1:portion, :);
        train_neg = train_neg(1:portion, :);
        test_pos = test_pos(1:portion, :);
        test_neg = test_neg(1:portion, :);
    end

    % --- logging ---
    fprintf('[NegPool] pool=%d need_total=%d a=%d eval_all=%d role_filter=%d | k_train=%d k_test=%d\n', ...
        pool_size, total_neg_links_needed, a, evaluate_on_all_unseen, use_role_filter, size(train_neg,1), size(test_neg,1));

    fprintf('[sample_neg] Final link counts (use_role_filter = %d):\n', use_role_filter);
    fprintf('    Train Positive: %d\n', size(train_pos, 1));
    fprintf('    Train Negative: %d\n', size(train_neg, 1));
    fprintf('    Test  Positive: %d\n', size(test_pos, 1));
    fprintf('    Test  Negative: %d\n', size(test_neg, 1));
end
