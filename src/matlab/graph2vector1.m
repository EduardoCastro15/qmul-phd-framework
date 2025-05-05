function [data, label] = graph2vector(pos, neg, A, K, useParallel)
    all = [pos; neg];
    pos_size = size(pos, 1);
    neg_size = size(neg, 1);
    all_size = pos_size + neg_size;

    label = [ones(pos_size, 1); zeros(neg_size, 1)];
    d = K * (K - 1) / 2;
    data = zeros(all_size, d);

    fprintf('Subgraph Pattern Encoding Begins for %d samples...\n', all_size);
    tic
    pool_created = false;

    if useParallel && isempty(gcp('nocreate'))
        parpool(feature('numcores'));
        pool_created = true;
    end

    if useParallel
        parfor i = 1:all_size
            data(i, :) = subgraph2vector(all(i, :), A, K);
            if mod(i, floor(all_size / 10)) == 0
                fprintf('Progress: %d%% — Elapsed: %.1f s\n', round(100*i/all_size), toc);
            end
        end
    else
        for i = 1:all_size
            data(i, :) = subgraph2vector(all(i, :), A, K);
            if mod(i, floor(all_size / 10)) == 0
                fprintf('Progress: %d%% — Elapsed: %.1f s\n', round(100*i/all_size), toc);
            end
        end
    end

    if pool_created
        delete(gcp('nocreate'));
    end
end

function sample = subgraph2vector(ind, A, K)
    D = K * (K - 1) / 2;
    max_depth = 5;
    links = ind;
    links_dist = 0;
    dist = 0;
    fringe = ind;
    nodes = [ind(1); ind(2)];
    nodes_dist = [0; 0];
    visited_set = containers.Map('KeyType','char','ValueType','logical');
    visited_set(sprintf('%d-%d', ind(1), ind(2))) = true;

    while true
        dist = dist + 1;
        fringe = neighbors_vectorized(fringe, A);
        if isempty(fringe), fringe = zeros(0,2); end

        % Fast deduplication
        new_fringe = zeros(0, 2);
        for row = 1:size(fringe,1)
            key = sprintf('%d-%d', fringe(row,1), fringe(row,2));
            if ~isKey(visited_set, key)
                visited_set(key) = true;
                new_fringe(end+1, :) = fringe(row, :);
            end
        end
        fringe = new_fringe;

        if isempty(fringe) || dist > max_depth
            subgraph = A(nodes, nodes);
            subgraph(1,2) = 0; subgraph(2,1) = 0;
            break
        end

        new_nodes = setdiff(fringe(:), nodes, 'rows');
        nodes = [nodes; new_nodes];
        nodes_dist = [nodes_dist; ones(length(new_nodes),1)*dist];
        links = [links; fringe];
        links_dist = [links_dist; ones(size(fringe,1),1)*dist];

        if numel(nodes) >= K
            subgraph = A(nodes, nodes);
            subgraph(1,2) = 0; subgraph(2,1) = 0;
            break
        end
    end

    links_ind = sub2ind(size(A), links(:,1), links(:,2));
    A_copy = A / (dist + 1);
    A_copy(links_ind) = 1 ./ links_dist;
    A_copy = max(triu(A_copy,1), tril(A_copy,-1)');
    A_copy = A_copy + A_copy';
    lweight_subgraph = A_copy(nodes, nodes);

    order = g_label(subgraph);
    if length(order) > K
        order(K+1:end) = [];
        subgraph = subgraph(order, order);
        lweight_subgraph = lweight_subgraph(order, order);
        order = g_label(subgraph);
    end

    plweight_subgraph = lweight_subgraph(order, order);
    sample = plweight_subgraph(triu(true(K),1));
    sample(1) = eps;

    if numel(sample) < D
        sample(end+1:D) = 0;
    end
end

function N = neighbors_vectorized(fringe, A)
    N = [];
    for i = 1:size(fringe, 1)
        u = fringe(i,1); v = fringe(i,2);
        succ = find(A(u,:)); pred = find(A(:,v));
        if ~isempty(succ)
            N = [N; [repmat(u, numel(succ), 1), succ']];
        end
        if ~isempty(pred)
            N = [N; [pred, repmat(v, numel(pred), 1)]];
        end
    end
    N = unique(N, 'rows');
end

function order = g_label(subgraph)
    K = size(subgraph, 1);
    G = graph(subgraph, 'upper');
    d1 = distances(G, 1); d2 = distances(G, 2);
    d1(isinf(d1)) = 2*K; d2(isinf(d2)) = 2*K;
    avg = sqrt(d1 .* d2);
    [~, ~, colors] = unique(avg);
    classes = palette_wl(subgraph, colors);
    order = canon(full(subgraph), classes)';
end
