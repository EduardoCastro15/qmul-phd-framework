function [train, test, train_nodes, test_nodes] = DivideNet(net, ratioTrain, nodeSelection, use_original_logic, check_connectivity, adaptive_connectivity, rare_fraction, varargin)
    % Divide a directed network into train/test sets + optional degree partitioning.
    %
    % --Input--
    %   net: n x n binary adjacency matrix
    %   ratioTrain: fraction of links to keep for training
    %   nodeSelection: degree nodeSelection ('high2low', 'low2high'), otherwise ignored
    %   use_original_logic: true to reproduce the WLNM paper version (undirected)
    %   check_connectivity: true to ensure u still reaches v after removal
    %   adaptive_connectivity: disables check_connectivity when n < 30
    %   rare_fraction: fraction of rare links to include in training
    %
    % --Output--
    %   train, test: train/test adjacency matrices
    %   train_nodes, test_nodes: only returned if degree nodeSelection used, else []
    %
    %  Partly adapted from the codes of
    %  Lu 2011, Link prediction in complex networks: A survey.
    %  Muhan Zhang, Washington University in St. Louis
    %
    %  *author: Jorge Eduardo Castro Cruces, Queen Mary University of London

    if nargin < 4, use_original_logic = false; end
    if nargin < 5, check_connectivity = false; end
    if nargin < 6, adaptive_connectivity = false; end
    if nargin < 7, rare_fraction = 0.0; end           % <- default OFF

    % clamp
    ratioTrain    = max(0, min(1, ratioTrain));
    rare_fraction = max(0, min(1, rare_fraction));

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
            else
                net(u, v) = 1;  % restore
            end
            linklist(idx, :) = [];
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

    % ---- Rare mode (toggle) ----
    if rare_fraction > 0 || strcmpi(nodeSelection, 'rarelinks')  % keep 'rarelinks' for backward compat
        [train, test, train_nodes, test_nodes] = ...
            splitRareLinks(net, linklist, ratioTrain, rare_fraction, n, ...
                           'check_connectivity', check_connectivity, ...
                           varargin{:});   % <- forward rare knobs from Main
        return;
    end

    % ---- Standard random removal with optional connectivity ----
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
    fprintf('[DivideNet] Test links accepted: %d / %d (%.1f%%)\n', accepted, num_test, 100 * accepted / max(1,num_test));
    fprintf('[DivideNet] Attempts made: %d | Failed attempts: %d\n', attempts, attempts - accepted);

    % ---- Optional degree-based node partitioning (orthogonal to rare mode) ----
    valid_strategies = ["high2low", "low2high"];   % 'rarelinks' no longer needed here
    if nargin < 3 || ~ismember(lower(string(nodeSelection)), valid_strategies)
        train_nodes = []; test_nodes = [];
        return;
    end

    % Degree-based node partitioning
    net_dense = full(net);
    total_deg = sum(net_dense, 1)' + sum(net_dense, 2);  % total degree
    [~, sorted_idx] = sort(total_deg, 'descend');
    cutoff = round(0.8 * n);

    switch lower(string(nodeSelection))
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
        if ~visited(current)
            visited(current) = true;
            neighbors = find(adj(current, :) > 0);
            queue = [queue; neighbors(~visited(neighbors))'];
        end
    end
end

% -- rare splitting with unified rounding --
function [train, test, train_nodes, test_nodes] = splitRareLinks(net, linklist, ratioTrain, rare_fraction, n, varargin)
    % Directed rare-links split with enforced TRAIN rare-quota and optional connectivity check.
    %
    % Name-Value params:
    %   'check_connectivity' (default: false)
    %   'rare_policy'        (default: 'hm')  % 'hm' (harmonic mean), 'sum', 'product', 'alpha_beta'
    %   'alpha'              (default: 1)     % for 'alpha_beta' policy
    %   'beta'               (default: 1)     % for 'alpha_beta' policy
    %   'quota_basis'        (default: 'train')  % 'train' or 'total'
    %   'min_rare_in_test'   (default: 0)     % reserve some rare edges in TEST if desired (0 = off)

    p = inputParser;
    addParameter(p, 'check_connectivity', false);
    addParameter(p, 'rare_policy', 'hm');       % 'hm','sum','product','alpha_beta'
    addParameter(p, 'alpha', 1);
    addParameter(p, 'beta', 1);
    addParameter(p, 'quota_basis', 'train');    % 'train' or 'total'
    addParameter(p, 'min_rare_in_test', 0);     % reserve fraction (0 = off)
    parse(p, varargin{:});
    opts = p.Results;

    % Filter self-loops (should be none already)
    linklist = linklist(linklist(:,1) ~= linklist(:,2), :);

    total_links = size(linklist, 1);
    num_test    = ceil((1 - ratioTrain) * total_links);     % <- unified with DivideNet
    num_train   = total_links - num_test;

    % ---- Edge rarity scores (lower = rarer) ----
    scores = computeEdgeRarity(net, linklist, opts.rare_policy, opts.alpha, opts.beta);
    % Tiny jitter to break ties deterministically under rng
    scores = scores + 1e-12 * randn(size(scores));

    [~, order] = sort(scores, 'ascend');

    % ---- Rare quota ----
    if strcmpi(opts.quota_basis, 'total')
        Krare = min(num_train, ceil(rare_fraction * total_links));
    else
        Krare = ceil(rare_fraction * num_train);
    end

    % NEW: guarantee that we can still fill TEST from non-rare edges
    max_Krare = max(0, total_links - num_test);   % leave >= num_test non-rare edges
    if Krare > max_Krare
        fprintf('[splitRareLinks] Reducing Krare from %d to %d to satisfy TEST size %d.\n', ...
                Krare, max_Krare, num_test);
        Krare = max_Krare;
    end

    % Optionally keep at least some rare edges for TEST
    Ktest_rare_min = min(num_test, floor(opts.min_rare_in_test * num_test));
    if Ktest_rare_min > 0
        Krare = max(0, Krare - Ktest_rare_min);
    end

    rare_rank_mask = false(total_links, 1);
    if Krare > 0
        rare_rank_mask(order(1:Krare)) = true;
    end

    % ---- Build TRAIN/TEST by removing only non-rare edges to TEST first ----
    train = sparse(net);         % start from full net
    test  = sparse(n, n);

    % Candidate edges for TEST (avoid touching rare-in-train set)
    candidates = find(~rare_rank_mask);
    candidates = candidates(randperm(numel(candidates)));

    accepted = 0;
    attempts = 0;

    for idx = candidates
        if accepted >= num_test, break; end
        u = linklist(idx, 1); v = linklist(idx, 2);
        if train(u, v) == 0, continue; end  % already removed
        attempts = attempts + 1;

        train(u, v) = 0;  % tentative removal
        if ~opts.check_connectivity || hasPath(train, u, v)
            test(u, v) = 1;
            accepted = accepted + 1;
        else
            train(u, v) = 1;  % restore
        end
    end

    % If we couldn't fill TEST, relax connectivity and/or dip into rare edges as last resort
    if accepted < num_test
        fprintf('[splitRareLinks] Connectivity blocked %d removals; relaxing check.\n', num_test - accepted);
        % still avoid rare edges, but ignore connectivity
        for idx = candidates
            if accepted >= num_test, break; end
            u = linklist(idx, 1); v = linklist(idx, 2);
            if train(u, v) == 0, continue; end
            train(u, v) = 0;
            test(u, v)  = 1;
            accepted    = accepted + 1;
        end
        % Absolute last resort: allow rare edges to move to TEST (still keeps most of the quota)
        if accepted < num_test
            rare_candidates = find(rare_rank_mask);
            rare_candidates = rare_candidates(randperm(numel(rare_candidates)));
            for idx = rare_candidates
                if accepted >= num_test, break; end
                u = linklist(idx, 1); v = linklist(idx, 2);
                if train(u, v) == 0, continue; end
                train(u, v) = 0;
                test(u, v)  = 1;
                accepted    = accepted + 1;
            end
        end
    end

    fprintf('[splitRareLinks] TEST accepted: %d / %d (quota basis: %s, rare_fraction: %.2f)\n', ...
            accepted, num_test, opts.quota_basis, rare_fraction);

    % Node bookkeeping (optional)
    [ti, tj] = find(train); train_nodes = unique([ti; tj]);
    [si, sj] = find(test);  test_nodes  = unique([si; sj]);
end

function scores = computeEdgeRarity(net, linklist, policy, alpha, beta)
    if nargin < 3 || isempty(policy), policy = 'hm'; end
    if nargin < 4, alpha = 1; end
    if nargin < 5, beta  = 1; end

    net = sparse(net);
    dout = full(sum(net, 2));     % column vector (n x 1)
    din  = full(sum(net, 1))';    % column vector (n x 1)

    u = linklist(:,1); v = linklist(:,2);
    switch lower(policy)
        case 'hm'      % harmonic mean of (dout(u)+1, din(v)+1)
            a = dout(u) + 1; b = din(v) + 1;
            scores = 2 ./ (1./a + 1./b);
        case 'sum'     % total-degree sum of endpoints
            tot = (dout + din);
            scores = tot(u) + tot(v);
        case 'product' % product of (dout(u)+1)*(din(v)+1)
            scores = (dout(u)+1) .* (din(v)+1);
        case 'alpha_beta' % weighted product
            scores = (dout(u)+1).^alpha .* (din(v)+1).^beta;
        otherwise
            error('Unknown rare policy: %s', policy);
    end
end
