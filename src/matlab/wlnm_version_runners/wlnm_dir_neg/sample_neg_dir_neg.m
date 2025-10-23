function [train_pos, train_neg, test_pos, test_neg] = sample_neg_dir_neg(train, test, role, a, portion, evaluate_on_all_unseen, use_role_filter)
    %  Usage: to sample negative links for train and test datasets.
    %         When sampling negative train links, assume all testing
    %         links are known and thus sample negative train links
    %         only from other unknown links. Set evaluate_on_all_unseen
    %         to true to do link prediction on all links unseen during
    %         training.
    %
    %  --Input--
    %  -train: half train positive adjacency matrix
    %  -test: half test positive adjacency matrix
    %  -role: cell array of roles for each node (e.g., {'consumer', 'resource'})
    %  -a: how many times of negative links (w.r.t. pos links) to sample
    %  -portion: if specified, only a portion of the sampled train and test links be returned
    %  -evaluate_on_all_unseen: if true, will not randomly sample
    %                          negative testing links, but regard
    %                          all links unseen during training as
    %                          neg testing links; train negative links
    %                          are sampled in the original way
    %  -use_role_filter: if true, filter negative links based on role constraints
    %
    %  --Output--
    %  -train_pos: a half train positive adjacency matrix
    %  -train_neg: a half train negative adjacency matrix
    %  -test_pos: a half test positive adjacency matrix
    %  -test_neg: a half test positive adjacency matrix
    %
    %  Partly adapted from the codes of
    %  Lu 2011, Link prediction in complex networks: A survey.
    %  Muhan Zhang, Washington University in St. Louis
    %
    % Author: Jorge Eduardo Castro Cruces, Queen Mary University of London

    % === positives ===
    [i, j] = find(train);
    train_pos = [i, j];
    train_size = size(train_pos, 1);

    [i, j] = find(test);
    test_pos = [i, j];
    test_size = size(test_pos, 1);

    % === Build full positive set ===
    net = train + test;
    assert(max(max(net)) <= 1, 'Train and test must not overlap');

    % === Find all possible negative links ===
    neg_net = (net == 0);  % all non-links
    neg_net = neg_net - diag(diag(neg_net));  % remove self-loops
    [i, j] = find(neg_net);
    neg_links_unfiltered = [i, j];  % save original
    neg_links = neg_links_unfiltered;  % default to unfiltered

    % --- Role-based filtering ---
    if use_role_filter
        % === Filter neg_links based on role constraints ===
        valid_mask = false(size(neg_links_unfiltered, 1), 1);

        for idx = 1:size(neg_links_unfiltered, 1)
            src = neg_links_unfiltered(idx, 1);
            tgt = neg_links_unfiltered(idx, 2);

            src_role = lower(string(role{src}));
            tgt_role = lower(string(role{tgt}));

            % if (src_role == "resource" && tgt_role == "consumer")              
            if (src_role == "consumer" && tgt_role == "consumer") || (src_role == "resource" && tgt_role == "resource")
                valid_mask(idx) = true;
            end
        end

        neg_links_filtered = neg_links_unfiltered(valid_mask, :);

        if ~isempty(neg_links_filtered)
            neg_links = neg_links_filtered;
        else
            warning('[sample_neg] No valid role-filtered negatives. Using unfiltered.');
        end
    end

    % --- compute needs and cap to pool size ---
    pool_size   = size(neg_links, 1);
    pos_total   = train_size + test_size;
    need_total  = a * pos_total;

    % fallback: if pool too small AND we used role filter, disable and rebuild once
    if pool_size < need_total && use_role_filter
        warning('[sample_neg] Pool %d < need %d with role filter. Disabling role filter.', pool_size, need_total);
        neg_links = neg_links_unfiltered;
        pool_size = size(neg_links, 1);
    end

    % final cap
    need_total = min(need_total, pool_size);

    if pool_size == 0 || need_total == 0
        warning('[sample_neg] No negatives available. Returning empties.');
        train_neg = zeros(0,2); test_neg = zeros(0,2);
        return;
    end

    % --- sample indices safely ---
    perm = randperm(pool_size);

    if evaluate_on_all_unseen
        % train_neg: up to a*train_size; test_neg: remaining unseen
        k_train = min(a * train_size, pool_size);
        idx_train = perm(1:k_train);
        train_neg = neg_links(idx_train, :);

        mask = true(pool_size, 1);
        mask(idx_train) = false;
        test_neg = neg_links(mask, :);
    else
        % proportional split if need_total < a*(train_size+test_size)
        ratio   = train_size / max(1, (train_size + test_size));
        k_train_target = a * train_size;
        k_test_target  = a * test_size;

        % propose proportional counts then cap by targets and need_total
        k_train = min(k_train_target, floor(need_total * ratio));
        k_test  = min(k_test_target,  need_total - k_train);

        % if rounding left a remainder, assign leftover greedily
        leftover = need_total - (k_train + k_test);
        if leftover > 0
            add_train = min(leftover, max(0, k_train_target - k_train));
            k_train = k_train + add_train;
            k_test  = need_total - k_train;
        end

        % ensure both splits get at least 1 negative when both pos sets are non-empty
        if train_size > 0 && test_size > 0 && need_total >= 2
            if k_train == 0 && k_test > 1
                k_train = 1; k_test = need_total - 1;
            elseif k_test == 0 && k_train > 1
                k_test = 1; k_train = need_total - 1;
            end
        end

        idx_sel  = perm(1:need_total);
        train_neg = neg_links(idx_sel(1:k_train), :);
        test_neg  = neg_links(idx_sel(k_train+1:end), :);
    end

    % === Apply portion filtering (if needed) ===
    if portion < 1
        train_pos = train_pos(1:min(size(train_pos,1), ceil(portion * size(train_pos, 1))), :);
        train_neg = train_neg(1:min(size(train_neg,1), ceil(portion * size(train_neg, 1))), :);
        test_pos  = test_pos(1:min(size(test_pos,1),   ceil(portion * size(test_pos, 1))), :);
        test_neg  = test_neg(1:min(size(test_neg,1),   ceil(portion * size(test_neg, 1))), :);
    elseif portion > 1
        train_pos = train_pos(1:min(size(train_pos,1), portion), :);
        train_neg = train_neg(1:min(size(train_neg,1), portion), :);
        test_pos  = test_pos(1:min(size(test_pos,1),  portion), :);
        test_neg  = test_neg(1:min(size(test_neg,1),  portion), :);
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
