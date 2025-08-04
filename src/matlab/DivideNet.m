function [train, test, train_nodes, test_nodes] = DivideNet(net, ratioTrain, strategy, use_original_logic, check_connectivity, adaptive_connectivity, rare_fraction)
    % Divide a directed network into train/test sets + node degree partitioning.
    %
    % --Input--
    %   net: n x n binary adjacency matrix
    %   ratioTrain: fraction of links to keep for training
    %   strategy: degree strategy ('high2low', 'low2high'), otherwise ignored
    %   use_original_logic: true to reproduce the WLNM paper version (undirected)
    %   check_connectivity: true to ensure u still reaches v after removal
    %   adaptive_connectivity: disables check_connectivity when n < 30
    %   rare_fraction: fraction of rare links to include in training
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

    if nargin < 4, use_original_logic = false; end
    if nargin < 5, check_connectivity = false; end
    if nargin < 6, adaptive_connectivity = false; end
    if nargin < 7, rare_fraction = 0.3; end

    n = size(net, 1);
    if adaptive_connectivity && n < 30
        check_connectivity = false;
        fprintf('[DivideNet] Skipping connectivity check (adaptive mode, n = %d).\n', n);
    end

    %% === Mode 1: Original WLNM logic ===
    if use_original_logic
        fprintf('[DivideNet] Using WLNM original logic (undirected upper-triangular).\n');

        net = triu(net) - diag(diag(net));  % upper triangle, no self-loops
        [i, j] = find(net);  % Extract link list
        linklist = [i, j];
        num_test = ceil((1 - ratioTrain) * size(linklist, 1));
        test = sparse(n, n);  % Init test adjacency

        while nnz(test) < num_test && ~isempty(linklist)
            idx = randi(size(linklist, 1));  % Randomly select a candidate edge
            u = linklist(idx, 1);
            v = linklist(idx, 2);
            net(u, v) = 0;  % Remove link from net temporarily

            if ~check_connectivity || hasPath(net, u, v)  % Check connectivity if required
                test(u, v) = 1;
                linklist(idx, :) = [];
            else
                net(u, v) = 1;  % restore
                linklist(idx, :) = [];
            end
        end

        % Return symmetric train/test
        train = net + net';
        test = test + test';
        train_nodes = [];
        test_nodes = [];
        return;
    end

    %% === Mode 2: Modern directed logic ===
    fprintf('[DivideNet] Using directed logic. Connectivity check: %d\n', check_connectivity);

    [i, j] = find(net);  % Extract all directed links (i → j)
    linklist = [i, j];
    linklist(linklist(:,1) == linklist(:,2), :) = [];  % Remove self-loops
    total_links = size(linklist, 1);  % Count remaining valid links
    num_test = ceil((1 - ratioTrain) * total_links);

    if strcmpi(strategy, 'rarelinks')
        [train, test, train_nodes, test_nodes] = splitRareLinks(net, linklist, ratioTrain, rare_fraction, n);
        return;
    end

    perm = randperm(total_links);
    test = sparse(n, n);
    train = net;
    accepted = 0; attempts = 0;

    for idx = perm
        if accepted >= num_test
            break;
        end
        u = linklist(idx, 1);
        v = linklist(idx, 2);
        train(u, v) = 0;  % tentative removal
        attempts = attempts + 1;

        if ~check_connectivity || hasPath(train, u, v)
            test(u, v) = 1;
            accepted = accepted + 1;
        else
            train(u, v) = 1;  % restore
        end
    end

    if accepted == 0
        warning('[DivideNet] No test links were accepted. Consider disabling connectivity check or using adaptive mode.');
    end
    fprintf('[DivideNet] Test links accepted: %d / %d (%.1f%%)\n', accepted, num_test, 100 * accepted / num_test);
    fprintf('[DivideNet] Attempts made: %d | Failed attempts: %d\n', attempts, attempts - accepted);

    valid_strategies = ["high2low", "low2high", "rarelinks"];  % Degree strategy (optional)
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

% Helper function: check if there's still a path from u to v
function reachable = hasPath(adj, u, v)
    reachable = false;
    visited = false(size(adj, 1), 1);
    queue = u;

    while ~isempty(queue)
        current = queue(1);
        queue(1) = [];
        if current == v
            reachable = true;
            return;
        end
        visited(current) = true;
        neighbors = find(adj(current, :) > 0);
        queue = [queue; neighbors(~visited(neighbors))'];
    end
end


% Helper function: rare link splitting
function [train, test, train_nodes, test_nodes] = splitRareLinks(net, linklist, ratioTrain, rare_fraction, n)

    fprintf('[DivideNet] Using rarelinks strategy with fraction %.2f for training selection.\n', rare_fraction);

    % Calculate out-degree and in-degree for each node
    out_deg = sum(net, 2);
    in_deg = sum(net, 1)';
    total_deg = out_deg + in_deg;

    % Sort nodes by rarity score (sum of out-degree and in-degree)
    % [~, sorted_idx] = sort(total_deg, 'ascend');
    rarity_score = total_deg(linklist(:,1)) + total_deg(linklist(:,2));
    [~, rare_idx] = sort(rarity_score, 'ascend');

    % Select training links based on the sorted rarity index
    total_links = size(linklist, 1);
    num_train_links = ceil(ratioTrain * total_links);
    num_rare_train = min(num_train_links, ceil(num_train_links * rare_fraction));

    % Ensure we have enough rare links
    rare_train_links = linklist(rare_idx(1:num_rare_train), :);
    remaining_idx = rare_idx(num_rare_train+1:end);
    remaining_needed = num_train_links - num_rare_train;
    other_train_links = linklist(remaining_idx(1:remaining_needed), :);

    % Combine rare and other training links
    train_links = [rare_train_links; other_train_links];
    test_links = setdiff(linklist, train_links, 'rows');

    % Create sparse matrices for train and test sets
    train = sparse(n, n);
    test = sparse(n, n);
    for k = 1:size(train_links, 1)
        train(train_links(k, 1), train_links(k, 2)) = 1;
    end
    for k = 1:size(test_links, 1)
        test(test_links(k, 1), test_links(k, 2)) = 1;
    end

    % Ensure symmetry for undirected networks
    train_nodes = unique(train_links(:));
    test_nodes = unique(test_links(:));

end