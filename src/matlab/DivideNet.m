function [train, test, train_nodes, test_nodes] = DivideNet(net, ratioTrain, strategy, use_original_logic, check_connectivity, adaptive_connectivity)
    % Divide a directed network into train/test sets + node degree partitioning.
    %
    % --Input--
    %   net: n x n binary adjacency matrix
    %   ratioTrain: fraction of links to keep for training
    %   strategy: degree strategy ('high2low', 'low2high'), otherwise ignored
    %   use_original_logic: true to reproduce the WLNM paper version (undirected)
    %   check_connectivity: true to ensure u still reaches v after removal
    %   adaptive_connectivity: disables check_connectivity when n < 30
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

    n = size(net, 1);
    if adaptive_connectivity && n < 30
        check_connectivity = false;
        fprintf('[DivideNet] Skipping connectivity check (adaptive mode, n = %d).\n', n);
    end

    % === Mode 1: Original WLNM logic ===
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

    % === Mode 2: Modern directed logic ===
    fprintf('[DivideNet] Using directed logic. Connectivity check: %d\n', check_connectivity);

    [i, j] = find(net);  % Extract all directed links (i → j)
    linklist = [i, j];
    linklist(linklist(:,1) == linklist(:,2), :) = [];  % Remove self-loops
    total_links = size(linklist, 1);  % Count remaining valid links
    num_test = ceil((1 - ratioTrain) * total_links);

    % Reproducible random shuffle
    % rng(42);
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

    valid_strategies = ["high2low", "low2high"];  % Degree strategy (optional)
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
