function [train_pos, train_neg, test_pos, test_neg] = sample_neg_directed(train, test, role, a, portion, evaluate_on_all_unseen, use_role_filter)
    %  Usage: to sample negative links for train and test datasets.
    %  --Input--
    %       -train: adjacency matrix of training links (directed)
    %       -test: adjacency matrix of testing links (directed)
    %       -a: number of negative links (relative to positive links) to sample
    %       -portion: fraction or number of links to sample
    %       -evaluate_on_all_unseen: if true, considers all unseen links as negative test links
    %  --Output--
    %       -train_pos: positive training links
    %       -train_neg: negative training links
    %       -test_pos: positive testing links
    %       -test_neg: negative testing links
    %%

    n = size(train, 1);
    [i, j] = find(train);
    train_pos = [i, j];
    train_size = length(i);
    [i, j] = find(test);
    test_pos = [i, j];
    test_size = length(i);

    % Combine train and test matrices to get the full network
    if isempty(test)
        net = train;
    else
        net = train + test;
    end

    % Ensure positive train and test links do not overlap
    assert(max(max(net)) == 1, 'Error: Positive train and test links overlap.');

    % Get all negative links (links that don't exist in the network)
    neg_net = -(net - 1);  % Invert the adjacency matrix (1 -> 0, 0 -> 1)
    neg_net(eye(n) == 1) = 0;  % Remove self-loops

    [i, j] = find(neg_net);
    neg_links = [i, j];
    pool_size = size(neg_links, 1);

    % Handle insufficient negative links
    total_neg_links_needed = a * (train_size + test_size);
    available_neg_links = size(neg_links, 1);

    if available_neg_links < total_neg_links_needed
        warning('Not enough negative links available. Reducing the sample size.');
        a = available_neg_links / (train_size + test_size);
    end

    need_total = min(pool_size, floor(a * (train_size + test_size)));

    % Sample negative links
    if evaluate_on_all_unseen
        test_neg = neg_links;  % first let all unknown links be negative test links
    
        % Randomly select train neg from all unknown links
        perm = randperm(size(neg_links, 1));
        train_neg = neg_links(perm(1: a * train_size), :);
        test_neg(perm(1: a * train_size), :) = [];  % remove train negative links from test negative links
    else
        nlinks = size(neg_links, 1);
        ind = randperm(nlinks);
        if a * (train_size + test_size) <= nlinks
            train_ind = ind(1: a * train_size);
            test_ind = ind(a * train_size + 1: a * train_size + a * test_size);
        else
            % Divide proportionally if negative links are insufficient
            ratio = train_size / (train_size + test_size);
            train_ind = ind(1: floor(ratio * nlinks));
            test_ind = ind(floor(ratio * nlinks) + 1: end);
        end
        train_neg = neg_links(train_ind, :);
        test_neg = neg_links(test_ind, :);
    end

    % Sample a portion of the links if specified
    if portion < 1  % only sample a portion of train and test links (for fitting into memory)
        train_pos = train_pos(1:ceil(size(train_pos, 1) * portion), :);
        train_neg = train_neg(1:ceil(size(train_neg, 1) * portion), :);
        test_pos = test_pos(1:ceil(size(test_pos, 1) * portion), :);
        test_neg = test_neg(1:ceil(size(test_neg, 1) * portion), :);
    elseif portion > 1  % portion is an integer, number of selections
        train_pos = train_pos(1:portion, :);
        train_neg = train_neg(1:portion, :);
        test_pos = test_pos(1:portion, :);
        test_neg = test_neg(1:portion, :);
    end        

    % --- logging ---
    fprintf('[NegPool] pool=%d need_total=%d a=%d eval_all=%d role_filter=%d | k_train=%d k_test=%d\n', ...
        pool_size, need_total, a, evaluate_on_all_unseen, use_role_filter, size(train_neg,1), size(test_neg,1));

    fprintf('[sample_neg] Final link counts (use_role_filter = %d):\n', use_role_filter);
    fprintf('    Train Positive: %d\n', size(train_pos, 1));
    fprintf('    Train Negative: %d\n', size(train_neg, 1));
    fprintf('    Test  Positive: %d\n', size(test_pos, 1));
    fprintf('    Test  Negative: %d\n', size(test_neg, 1));
end
