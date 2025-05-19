function [train, test, train_nodes, test_nodes] = DivideNet(net, ratioTrain, strategy)
    % Divide a directed network into train/test sets + node degree partitioning.
    %
    % --Input--
    %   net:        n x n binary adjacency matrix
    %   ratioTrain: fraction of links to keep for training
    %   strategy:   degree strategy ('high2low', 'low2high'), otherwise default split
    %
    % --Output--
    %   train, test: train/test adjacency matrices
    %   train_nodes, test_nodes: only returned if degree strategy used, else []
    %
    %  Partly adapted from the codes of
    %  Lu 2011, Link prediction in complex networks: A survey.
    %  Muhan Zhang, Washington University in St. Louis
    %
    %  *author: Jorge Eduardo Castro Cruces, Queen Mary University of London

    % Extract all directed links (i → j)
    [i, j] = find(net);
    linklist = [i, j];

    % Remove self-loops (where source == target)
    self_loops = (i == j);
    linklist(self_loops, :) = [];

    % Count remaining valid links
    total_links = size(linklist, 1);
    num_test = ceil((1 - ratioTrain) * total_links);

    % Reproducible random shuffle
    % rng(42);  % Fixed seed for consistent experiments
    perm = randperm(total_links);
    test_links = linklist(perm(1:num_test), :);
    train_links = linklist(perm(num_test + 1:end), :);

    % Initialize adjacency matrices
    n = size(net, 1);
    train = sparse(n, n);
    test = sparse(n, n);

    % Assign training links
    for k = 1:size(train_links, 1)
        train(train_links(k, 1), train_links(k, 2)) = 1;
    end

    % Assign testing links
    for k = 1:size(test_links, 1)
        test(test_links(k, 1), test_links(k, 2)) = 1;
    end

    % Check for strategy
    valid_strategies = ["high2low", "low2high"];
    if nargin < 3 || ~ismember(lower(strategy), valid_strategies)
        % Use default logic without node partitioning
        train_nodes = [];
        test_nodes = [];
        return;
    end

    % Degree-based node partitioning
    net_dense = full(net);
    total_deg = sum(net_dense, 1)' + sum(net_dense, 2);  % total degree
    [~, sorted_idx] = sort(total_deg, 'descend');
    cutoff = round(0.8 * n);

    switch lower(strategy)
        case 'high2low'
            train_nodes = sorted_idx(1:cutoff);
            test_nodes  = sorted_idx(cutoff+1:end);
        case 'low2high'
            train_nodes = sorted_idx(cutoff+1:end);
            test_nodes  = sorted_idx(1:cutoff);
    end
end
