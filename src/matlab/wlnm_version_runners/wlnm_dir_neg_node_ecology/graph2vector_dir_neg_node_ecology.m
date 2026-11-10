function [data, label, canonical_node_ids, node_counts] = ...
        graph2vector_dir_neg_node_ecology(pos, neg, A, K, useParallel, dataname)
%GRAPH2VECTOR_DIR_NEG_NODE_ECOLOGY Encode topology and expose canonical IDs.
% A(i,j)=1 means resource i -> consumer j. The candidate direction is
% removed from both the binary and weighted matrices before vectorization.
% canonical_node_ids uses zero padding because MATLAB node IDs start at 1.

    if nargin < 6
        dataname = '';
    end
    if ~isscalar(K) || ~isfinite(K) || K ~= 10
        error('WLNMDirNegNodeEcology:SubgraphSize', ...
            'The node-ecology protocol is frozen at K=10.');
    end

    A = sparse(A);
    if size(A,1) ~= size(A,2)
        error('WLNMDirNegNodeEcology:AdjacencyShape', ...
            'The adjacency matrix must be square.');
    end

    pairs = [pos; neg];
    pos_size = size(pos,1);
    all_size = size(pairs,1);
    label = [ones(pos_size,1); zeros(size(neg,1),1)];
    data = zeros(all_size, K * (K - 1));
    canonical_node_ids = zeros(all_size, K, 'int32');
    node_counts = zeros(all_size, 1);

    fprintf('Encoding %d node-ecology subgraphs for %s (K = %d)...\n', ...
        all_size, char(string(dataname)), K);
    timer = tic;

    if useParallel && isempty(gcp('nocreate'))
        parpool('local');
    end

    if useParallel
        parfor row = 1:all_size
            [data(row,:), canonical_node_ids(row,:), node_counts(row)] = ...
                subgraph2vector_node_ecology(pairs(row,:), A, K);
        end
    else
        step = max(1, floor(all_size / 10));
        for row = 1:all_size
            [data(row,:), canonical_node_ids(row,:), node_counts(row)] = ...
                subgraph2vector_node_ecology(pairs(row,:), A, K);
            if mod(row, step) == 0 || row == all_size
                fprintf('Progress: %d%% - Elapsed: %.1fs\n', ...
                    round(100 * row / max(1, all_size)), toc(timer));
            end
        end
    end

    fprintf('Done. Total time: %.1fs\n', toc(timer));
end

function [sample, padded_ids, node_count] = ...
        subgraph2vector_node_ecology(ind, A, K)
    D = K * (K - 1);
    max_depth = 2;
    n = size(A,1);

    if numel(ind) ~= 2 || any(ind < 1) || any(ind > n) || ...
            any(ind ~= floor(ind)) || ind(1) == ind(2)
        error('WLNMDirNegNodeEcology:InvalidCandidate', ...
            'Candidates must contain two distinct integer node IDs.');
    end

    nodes = unique([ind(1); ind(2)], 'stable');
    links = ind;
    links_dist = 0;
    dist = 0;
    blocked_idx = sub2ind([n,n], ind(1), ind(2));

    while true
        dist = dist + 1;
        fringe = neighbors_vectorized_node_ecology(links, A, blocked_idx);
        if isempty(fringe) || dist > max_depth
            break;
        end

        new_nodes = setdiff(fringe(:), nodes, 'stable');
        nodes = [nodes; new_nodes]; %#ok<AGROW>
        links = [links; fringe]; %#ok<AGROW>
        links_dist = [links_dist; dist * ones(size(fringe,1),1)]; %#ok<AGROW>

        if numel(nodes) >= K
            nodes = nodes(1:K);
            break;
        end
    end

    nodes = unique(nodes, 'stable');
    node_count = numel(nodes);
    padded_ids = zeros(1, K, 'int32');

    if node_count < 2
        sample = zeros(1,D);
        if node_count == 1
            padded_ids(1) = int32(nodes(1));
        end
        return;
    end

    binary_subgraph = A(nodes,nodes);
    binary_subgraph(1,2) = 0;

    weighted_subgraph = build_weighted_subgraph_node_ecology( ...
        A, nodes, links, links_dist, dist);
    weighted_subgraph(1,2) = 0;

    order = g_label_node_ecology(binary_subgraph);
    ordered_weighted = weighted_subgraph(order,order);
    ordered_ids = nodes(order);
    padded_ids(1:node_count) = int32(ordered_ids(:)');

    offdiag = ~eye(size(ordered_weighted,1));
    sample = ordered_weighted(offdiag);
    if ~isempty(sample)
        sample(1) = eps;
    end
    if numel(sample) < D
        sample(end+1:D) = 0;
    end
    sample = reshape(full(sample), 1, D);
end

function N = neighbors_vectorized_node_ecology(links, A, blocked_idx)
    n = size(A,1);
    N = zeros(0,2);
    visited_idx = blocked_idx(:);

    for row = 1:size(links,1)
        u = links(row,1);
        v = links(row,2);
        successors = find(A(u,:));
        predecessors = find(A(:,v))';

        for successor = successors
            link_idx = sub2ind([n,n], u, successor);
            if ~ismember(link_idx, visited_idx)
                N(end+1,:) = [u, successor]; %#ok<AGROW>
                visited_idx(end+1,1) = link_idx; %#ok<AGROW>
            end
        end
        for predecessor = predecessors
            link_idx = sub2ind([n,n], predecessor, v);
            if ~ismember(link_idx, visited_idx)
                N(end+1,:) = [predecessor, v]; %#ok<AGROW>
                visited_idx(end+1,1) = link_idx; %#ok<AGROW>
            end
        end
    end

    if ~isempty(N)
        N = unique(N, 'rows');
    end
end

function weighted = build_weighted_subgraph_node_ecology( ...
        A, nodes, links, links_dist, dist)
    weighted = A(nodes,nodes) / (dist + 1);
    keep = links_dist > 0;
    if ~any(keep)
        return;
    end

    node_position = zeros(size(A,1),1);
    node_position(nodes) = 1:numel(nodes);
    local_i = node_position(links(keep,1));
    local_j = node_position(links(keep,2));
    local_dist = links_dist(keep);
    inside = local_i > 0 & local_j > 0;
    if ~any(inside)
        return;
    end
    local_idx = sub2ind(size(weighted), local_i(inside), local_j(inside));
    weighted(local_idx) = 1 ./ local_dist(inside);
end

function order = g_label_node_ecology(subgraph)
    Klocal = size(subgraph,1);
    G = graph(subgraph, "upper");
    d1 = distances(G,1);
    d2 = distances(G,2);
    d1(isinf(d1)) = 2 * Klocal;
    d2(isinf(d2)) = 2 * Klocal;
    average_distance = sqrt(d1 .* d2);
    [~,~,colors] = unique(average_distance);
    classes = palette_wl(subgraph, colors);
    order = canon(full(subgraph), classes);
end
