function [train_pos, train_neg, test_pos, test_neg] = sample_neg_original(train, test, a, portion, evaluate_on_all_unseen, use_role_filter)
    %SAMPLE_NEG_ORIGINAL Negative sampling for the "original" WLNM pipeline.
    % Preserves DIR_NEG-style logging only (no role-based filtering here).
    %
    % Compatible call forms:
    %   [tp, tn, vp, vn] = sample_neg_original(train, test, a, portion, eval_all)
    %   [tp, tn, vp, vn] = sample_neg_original(train, test, role, a, portion, eval_all, use_role_filter)
    %
    % Inputs
    %   train, test : upper/half positive adjacency matrices (same size)
    %   role       : (optional) cellstr per node, ignored here (logged only)
    %   a          : multiplier of negatives per positive (default 1)
    %   portion    : <1 keep fraction; >1 keep first N rows (default 1)
    %   eval_all   : if true, test_neg = all unseen; train_neg sampled (default false)
    %   use_role_filter : logged only, not applied
    %
    % Outputs
    %   train_pos, train_neg, test_pos, test_neg : [N×2] index pairs

    if nargin < 3
        a = 1;
    end
    
    if nargin < 4
        portion = 1;
    end
    
    if nargin < 5
        evaluate_on_all_unseen = false;
    end

    [i, j] = find(train);
    train_pos = [i, j];
    train_size = length(i);
    [i, j] = find(test);
    test_pos = [i, j];
    test_size = length(i);

    % Build combined pos network & sanity check
    if isempty(test)
        net = train;
    else
        net = train + test;
    end

    assert(max(max(net)) == 1, 'Train and test must not overlap.');

    % Get all negative links (links that don't exist in the network)
    neg_net = triu(-(net - 1), 1);
    [i, j] = find(neg_net);
    neg_links = [i, j];

    %% Modification to workaround if the negative links are not enough. [START]
    % Check if we have enough negative links
    pool_size = size(neg_links, 1);
    total_neg_links_needed = a * (train_size + test_size);
    available_neg_links = size(neg_links, 1);

    if available_neg_links < total_neg_links_needed
        warning('Not enough negative links available. Reducing the sample size.');
        a = available_neg_links / (train_size + test_size);
    end
    %% Modification to workaround if the negative links are not enough. [END]

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
        else  % if negative links not enough, divide them proportionally
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

    % -------- Logging (preserved style) --------
    % For consistency with sample_neg_dir_neg:
    need_total_logged = min(pool_size, floor(a * (train_size + test_size)));
    fprintf('[NegPool] pool=%d need_total=%d a=%g eval_all=%d role_filter=%d | k_train=%d k_test=%d\n', ...
        pool_size, need_total_logged, a, logical(evaluate_on_all_unseen), logical(use_role_filter), ...
        size(train_neg,1), size(test_neg,1));

    fprintf('[sample_neg] Final link counts (use_role_filter = %d):\n', logical(use_role_filter));
    fprintf('    Train Positive: %d\n', size(train_pos, 1));
    fprintf('    Train Negative: %d\n', size(train_neg, 1));
    fprintf('    Test  Positive: %d\n', size(test_pos, 1));
    fprintf('    Test  Negative: %d\n', size(test_neg, 1));
end
    