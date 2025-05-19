function [train_pos, train_neg, test_pos, test_neg] = sample_neg(train, test, role, k, portion, evaluate_on_all_unseen)
    %  Usage: to sample negative links for train and test datasets.
    %         When sampling negative train links, assume all testing
    %         links are known and thus sample negative train links
    %         only from other unknown links. Set evaluate_on_all_unseen
    %         to true to do link prediction on all links unseen during
    %         training.
    %  --Input--
    %  -train: half train positive adjacency matrix
    %  -test: half test positive adjacency matrix
    %  -k: how many times of negative links (w.r.t. pos links) to
    %      sample
    %  -portion: if specified, only a portion of the sampled train
    %            and test links be returned
    %  -evaluate_on_all_unseen: if true, will not randomly sample
    %                          negative testing links, but regard
    %                          all links unseen during training as
    %                          neg testing links; train negative links
    %                          are sampled in the original way
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
    %  *author: Jorge Eduardo Castro Cruces, Queen Mary University of London

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

    % === Filter neg_links based on role constraints ===
    valid_mask = false(size(neg_links_unfiltered, 1), 1);

    for idx = 1:size(neg_links_unfiltered, 1)
        src = neg_links_unfiltered(idx, 1);
        tgt = neg_links_unfiltered(idx, 2);

        src_role = lower(string(role(src)));
        tgt_role = lower(string(role(tgt)));

        if (src_role == "consumer" && tgt_role == "consumer") || ...
        (src_role == "resource" && tgt_role == "resource")
            valid_mask(idx) = true;
        end
    end

    neg_links_filtered = neg_links_unfiltered(valid_mask, :);

    if ~isempty(neg_links_filtered)
        neg_links = neg_links_filtered;
    else
        warning('[sample_neg] No valid role-filtered negative links found. Using all negatives.');
    end

    % === Check for enough negatives ===
    total_neg_needed = k * (train_size + test_size);
    if size(neg_links, 1) < total_neg_needed
        warning('Not enough negative links. Reducing k...');
        k = floor(size(neg_links, 1) / (train_size + test_size));
    end

    if k == 0
        warning('[sample_neg] Negative sampling failed — not enough links even after fallback. Using random sampling without role constraint.');
        neg_links = neg_links_unfiltered;
        k = 1;  % fallback to minimal sampling
    end    

    % === Sample negatives ===
    % rng(42);  % reproducibility
    if evaluate_on_all_unseen
        test_neg = neg_links;
        perm = randperm(size(neg_links, 1));
        train_neg = neg_links(perm(1:k * train_size), :);
        test_neg(perm(1:k * train_size), :) = [];  % remove train negs
    else
        perm = randperm(size(neg_links, 1));
        total_needed = k * (train_size + test_size);
        selected = neg_links(perm(1:total_needed), :);
        train_neg = selected(1:k * train_size, :);
        test_neg = selected(k * train_size + 1:end, :);
    end

    % === Apply portion filtering (if needed) ===
    if portion < 1
        train_pos = train_pos(1:ceil(size(train_pos, 1) * portion), :);
        train_neg = train_neg(1:ceil(size(train_neg, 1) * portion), :);
        test_pos  = test_pos(1:ceil(size(test_pos, 1) * portion), :);
        test_neg  = test_neg(1:ceil(size(test_neg, 1) * portion), :);
    elseif portion > 1
        train_pos = train_pos(1:portion, :);
        train_neg = train_neg(1:portion, :);
        test_pos  = test_pos(1:portion, :);
        test_neg  = test_neg(1:portion, :);
    end

    % === Log counts of generated links ===
    fprintf('[sample_neg] Final link counts:\n');
    fprintf('    Train Positive: %d\n', size(train_pos, 1));
    fprintf('    Train Negative: %d\n', size(train_neg, 1));
    fprintf('    Test  Positive: %d\n', size(test_pos, 1));
    fprintf('    Test  Negative: %d\n', size(test_neg, 1));
end
