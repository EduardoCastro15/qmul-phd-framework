function [data, label] = graph2vector(pos, neg, A, K, useParallel)
    % Convert enclosing subgraphs of links (positive and negative) into vector representations

    all = [pos; neg];
    pos_size = size(pos, 1);
    neg_size = size(neg, 1);
    all_size = pos_size + neg_size;

    label = [ones(pos_size, 1); zeros(neg_size, 1)];

    d = K * (K - 1) / 2;
    data = zeros(all_size, d);

    fprintf('Encoding %d subgraphs (K = %d)...\n', all_size, K);
    tic;

    if useParallel && isempty(gcp('nocreate'))
        parpool('local');
    end

    if useParallel
        parfor i = 1:all_size
            ind = all(i, :);
            data(i, :) = subgraph2vector(ind, A, K);
        end
    else
        for i = 1:all_size
            ind = all(i, :);
            data(i, :) = subgraph2vector(ind, A, K);
            if mod(i, floor(all_size / 10)) == 0
                fprintf('Progress: %d%% – Elapsed: %.1fs\n', round(100 * i / all_size), toc);
                fprintf("Encoding link %d of %d: (%d,%d)\n", i, all_size, ind(1), ind(2));
            end
        end
    end

    fprintf('Done. Total time: %.1fs\n', toc);
end



function sample = subgraph2vector(ind, A, K)
    D = K * (K - 1) / 2;
    max_depth = 2;

    nodes = unique([ind(1); ind(2)], 'stable');
    links = ind;
    links_dist = 0;
    dist = 0;
    visited = false(size(A));
    visited(ind(1), ind(2)) = true;

    while true
        dist = dist + 1;
        fringe = neighbors_vectorized(links, A, visited);

        if isempty(fringe) || dist > max_depth
            break;
        end

        new_nodes = setdiff(fringe(:), nodes, 'stable');
        nodes = [nodes; new_nodes];

        if numel(nodes) > 3*K
            fprintf('[WARN] Link (%d,%d) triggered large expansion: %d nodes\n', ind(1), ind(2), numel(nodes));
        end
        
        links = [links; fringe];
        links_dist = [links_dist; dist * ones(size(fringe, 1), 1)];

        if numel(nodes) >= K
            nodes = nodes(1:K);  % Hard cut to avoid very large subgraphs
            break;
        end        
    end

    nodes = unique(nodes, 'stable');

    % === Safeguard against subgraphs too small ===
    if numel(nodes) < 2
        sample = zeros(1, D);
        return;
    end

    subgraph = A(nodes, nodes);

    % avoid encoding the true link if present
    if size(subgraph,1) >= 2
        subgraph(1, 2) = 0;
        subgraph(2, 1) = 0;
    end

    links_ind = sub2ind(size(A), links(:,1), links(:,2));
    A_copy = A / (dist + 1);
    A_copy(links_ind) = 1 ./ links_dist;
    A_copy_u = max(triu(A_copy, 1), tril(A_copy, -1)');
    A_copy = A_copy_u + A_copy_u';
    lweight_subgraph = A_copy(nodes, nodes);

    order = g_label(subgraph);

    if length(order) > K
        order(K+1:end) = [];
        subgraph = subgraph(order, order);
        lweight_subgraph = lweight_subgraph(order, order);
        order = g_label(subgraph);
    end

    plweight_subgraph = lweight_subgraph(order, order);
    sample = plweight_subgraph(triu(true(size(subgraph)), 1));
    sample(1) = eps;

    if numel(sample) < D
        sample(end+1:D) = 0;
    end
end




function N = neighbors_vectorized(links, A, visited)
    N = [];

    for i = 1:size(links, 1)
        u = links(i, 1);
        v = links(i, 2);

        succ = find(A(u, :));
        pred = find(A(:, v))';

        % Build new pairs and filter if not visited
        for s = succ
            if ~visited(u, s)
                N(end+1, :) = [u, s];
                visited(u, s) = true;
            end
        end
        for p = pred
            if ~visited(p, v)
                N(end+1, :) = [p, v];
                visited(p, v) = true;
            end
        end
    end

    if ~isempty(N)
        N = unique(N, 'rows');
    end
end



function order = g_label(subgraph)
    %  Usage: impose a vertex order for a enclosing subgraph using graph labeling

    K = size(subgraph, 1);
    G = graph(subgraph, "upper");  % Create a graph object from the adjacency matrix
    d1 = distances(G, 1);  % Compute shortest paths from node 1
    d2 = distances(G, 2);  % Compute shortest paths from node 2

    d1(isinf(d1)) = 2 * K;  % replace inf nodes (unreachable from 1 or 2) by an upperbound dist
    d2(isinf(d2)) = 2 * K;
    avg = sqrt(d1 .* d2);  % use geometric mean as the average distance to the link
    [~, ~, colors] = unique(avg);  % f mapping to initial colors

    % palette_wl with initial colors, break ties by nauty
    classes = palette_wl(subgraph, colors);
    %classes = palette_wl(subgraph);  % no initial colors
    order = canon(full(subgraph), classes)';
end
