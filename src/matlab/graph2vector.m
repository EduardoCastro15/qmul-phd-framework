function [data, label] = graph2vector(pos, neg, A, K, useParallel, dataname, use_original_wlnm)
    % Usage: Encode subgraphs of links in a graph into vectors
    %  --Input--
    %  -pos: indices of positive links
    %  -neg: indices of negative links
    %  -A: adjacency matrix
    %  -K: number of nodes per subgraph
    %  -useParallel: whether to use parallel computing
    %  -dataname: dataset name for building block storage
    %  -use_original_wlnm: if true, use the original WLNM subgraph extraction logic
    %
    %  --Output--
    %  -data: the constructed training data, each row is a link's vector representation
    %  -label: a column vector of links' labels
    %
    %  Partly adapted from the codes of
    %  Lu 2011, Link prediction in complex networks: A survey.
    %  Muhan Zhang, Washington University in St. Louis
    %
    % Author: Jorge Eduardo Castro Cruces, Queen Mary University of London

    if nargin < 7, use_original_wlnm = false; end
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
            is_positive = i <= pos_size;
            if use_original_wlnm
                data(i, :) = subgraph2vector_original(ind, A, K);
            else
                data(i, :) = subgraph2vector(ind, A, K, dataname, is_positive, i);
            end
        end
    else
        for i = 1:all_size
            ind = all(i, :);
            is_positive = i <= pos_size;
            if use_original_wlnm
                data(i, :) = subgraph2vector_original(ind, A, K);
            else
                data(i, :) = subgraph2vector(ind, A, K, dataname, is_positive, i);
            end
            if mod(i, floor(all_size / 10)) == 0
                fprintf('Progress: %d%% – Elapsed: %.1fs\n', round(100 * i / all_size), toc);
                % fprintf("Encoding link %d of %d: (%d,%d)\n", i, all_size, ind(1), ind(2));
            end
        end
    end

    fprintf('Done. Total time: %.1fs\n', toc);
end

function sample = subgraph2vector_original(ind, A, K)
    D = K * (K - 1) / 2;
    max_nodes = 3 * K;

    nodes = unique(ind(:), 'stable');
    visited = nodes(:);
    fringe = ind;

    while numel(nodes) < K
        new_nodes = [];
        for i = 1:size(fringe, 1)
            u = fringe(i, 1);
            v = fringe(i, 2);
            neighbors = unique([find(A(u, :)), find(A(:, v)')]);
            new_nodes = [new_nodes, neighbors];
        end
        new_nodes = setdiff(unique(new_nodes(:)), nodes, 'stable');
        if isempty(new_nodes)
            break;
        end
        nodes = [nodes; new_nodes];
        if numel(nodes) > max_nodes
            break;
        end
        fringe = nchoosek(nodes, 2);
    end

    nodes = nodes(1:min(end, K));
    subgraph = A(nodes, nodes);

    if size(subgraph,1) >= 2
        subgraph(1, 2) = 0;  % remove the link
        subgraph(2, 1) = 0;  % remove the link
    end

    order = g_label(subgraph);
    subgraph = subgraph(order, order);

    plweight_subgraph = subgraph;
    sample = plweight_subgraph(triu(true(size(subgraph)), 1));
    sample(1) = eps;

    if numel(sample) < D
        sample(end+1:D) = 0;
    end
end

function sample = subgraph2vector(ind, A, K, dataname, is_positive, idx)
    % Save enclosing subgraph snapshot (optional for building block extraction)
    save_building_blocks = true;

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

    subgraph = A(nodes, nodes);                 % Adjacency before any canonical relabeling
    adj_before = subgraph;                      % Save before editing

    % avoid encoding the true link if present
    if size(subgraph,1) >= 2
        subgraph(1, 2) = 0;  % remove the link
        subgraph(2, 1) = 0;  % remove the link
    end

    links_ind = sub2ind(size(A), links(:,1), links(:,2));
    A_copy = A / (dist + 1);
    A_copy(links_ind) = 1 ./ links_dist;
    A_copy_u = max(triu(A_copy, 1), tril(A_copy, -1)');
    A_copy = A_copy_u + A_copy_u';
    lweight_subgraph = A_copy(nodes, nodes);

    % === Canonical labeling ===
    [order, classes] = g_label(subgraph);

    % Optional truncation
    if length(order) > K
        order(K+1:end) = [];
        subgraph = subgraph(order, order);
        lweight_subgraph = lweight_subgraph(order, order);
        order = g_label(subgraph);
    end

    subgraph_ordered = subgraph(order, order);         % Binary
    plweight_subgraph = lweight_subgraph(order, order);% Weighted

    % Vector encoding
    sample = plweight_subgraph(triu(true(size(subgraph)), 1));
    sample(1) = eps;

    if numel(sample) < D
        sample(end+1:D) = 0;
    end

    % === Save visualization info ===
    if save_building_blocks
        block_dir = fullfile('data/result/building_blocks', dataname);
        if ~exist(block_dir, 'dir')
            try
                mkdir(block_dir);
            catch
                % Avoid race conditions in parallel mkdir
            end
        end

        % Ensure types are double and vectors are column vectors
        building_block.adj_before     = full(double(adj_before));
        building_block.adj_after      = full(double(subgraph_ordered));
        building_block.ordered_adj    = full(double(plweight_subgraph));
        building_block.nodes          = double(nodes(:));          % original order
        building_block.nodes_ordered  = double(nodes(order(:)));   % relabeled
        building_block.link           = double(ind(:));
        building_block.order          = double(order(:));
        building_block.label          = double(is_positive);
        building_block.classes        = double(classes(:));

        filename = sprintf('link_%d_%d_K_%d_idx_%d_forviz.mat', ind(1), ind(2), K, idx);
        outpath = fullfile(block_dir, filename);
        save(outpath, '-struct', 'building_block', '-v7');
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

function [order, classes] = g_label(subgraph)
    % Algorithm 1: Weisfeiler-lehman Graph Labeling
    % Usage: impose a vertex order for a enclosing subgraph using graph labeling

    K = size(subgraph, 1);
    G = graph(subgraph, "upper");  % Create a graph object from the adjacency matrix
    d1 = distances(G, 1);  % Compute shortest paths from node 1
    d2 = distances(G, 2);  % Compute shortest paths from node 2

    d1(isinf(d1)) = 2 * K;  % replace inf nodes (unreachable from 1 or 2) by an upperbound dist
    d2(isinf(d2)) = 2 * K;

    avg = sqrt(d1 .* d2);  % use geometric mean as the average distance to the link
    [~, ~, colors] = unique(avg);  % f mapping to initial colors

    classes_init = palette_wl(subgraph, colors);  % palette_wl with initial colors, break ties by nauty
    [order, classes] = canon(full(subgraph), classes_init);  % canon returns both
end
