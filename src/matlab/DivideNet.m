function [train, test] = DivideNet(net, ratioTrain)
    % Divide a directed network into training and testing sets,
    % preserving directionality and removing self-loops.
    %
    % --Input--
    %   net:        n x n binary adjacency matrix (directed)
    %   ratioTrain: fraction of links to retain in training set (e.g., 0.9)
    %
    % --Output--
    %   train:      directed training adjacency matrix
    %   test:       directed testing adjacency matrix
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
    rng(42);  % Fixed seed for consistent experiments
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
end
