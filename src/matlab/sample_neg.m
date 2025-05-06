function [train_pos, train_neg, test_pos, test_neg] = sample_neg(train, test, k, portion, evaluate_on_all_unseen)
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
    neg_links = [i, j];

    % === Check for enough negatives ===
    total_neg_needed = k * (train_size + test_size);
    if size(neg_links, 1) < total_neg_needed
        warning('Not enough negative links. Reducing k...');
        k = floor(size(neg_links, 1) / (train_size + test_size));
    end

    % === Sample negatives ===
    rng(42);  % reproducibility
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
end
