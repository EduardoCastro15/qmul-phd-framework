function [data, label] = graph2vector_directed(pos, neg, A, K, useParallel, dataname, use_original_wlnm)
    %  Usage: to convert links' enclosing subgraphs (both pos and neg) into real vectors
    %  --Input--
    %       -pos: indices of positive links
    %       -neg: indices of negative links
    %       -A: the observed graph's adjacency matrix from which to
    %           to extract subgraph features
    %       -K: the number of nodes in each link's subgraph
    %  --Output--
    %       -data: the constructed training data, each row is a
    %              link's vector representation
    %       -label: a column vector of links' labels
    %
    %  *author: Muhan Zhang, Washington University in St. Louis

    all = [pos; neg];
    pos_size = size(pos, 1);
    neg_size = size(neg, 1);
    all_size = pos_size + neg_size;

    % Generate labels
    label = [ones(pos_size, 1); zeros(neg_size, 1)];

    % Generate vector data
    d = K * (K - 1) / 2;  % dim of data vectors
    data = zeros(all_size, d);

    fprintf('Encoding %d subgraphs (K = %d)...\n', all_size, K);
    t0   = tic;
    step = max(1, floor(all_size/10));

    for i = 1: all_size
        ind = all(i, :);
        is_positive = i <= pos_size;
        sample = subgraph2vector(ind, A, K, dataname, is_positive, i);
        data(i, :) = sample;

        if mod(i, step) == 0 || i == all_size
            fprintf('  Progress: %3d%%  Elapsed: %.1fs\n', round(100*i/all_size), toc(t0));
        end
    end
end


function sample = subgraph2vector(ind, A, K, dataname, is_positive, idx)
    %  Usage: 1) to extract the enclosing subgraph for a link
    %            Aij (i = ind(1), j = ind(2))
    %         2) to impose a vertex ordering for the vertices
    %            of the enclosing subgraph using graph labeling
    %         3) to construct an adjacency matrix and output
    %            the reshaped vector
    %
    %  *author: Muhan Zhang, Washington University in St. Louis
    save_building_blocks = false;

    D = K * (K - 1) / 2;  % the length of output vector

    % Extract a subgraph of K nodes
    links = [ind];
    links_dist = [0];  % the graph distance to the initial link
    dist = 0;
    fringe = [ind];
    nodes = [ind(1); ind(2)];
    nodes_dist = [0; 0];
    while 1
        dist = dist + 1;
        fringe = neighbors(fringe, A);
        fringe = setdiff(fringe, links, 'rows');
        if isempty(fringe)  % no more new neighbors, add dummy nodes
            subgraph = A(nodes, nodes);
            adj_before = subgraph;
            subgraph(1, 2) = 0;  % ensure subgraph patterns do not contain information about link existence
            subgraph(2, 1) = 0;
            break
        end
        new_nodes = setdiff(fringe(:), nodes, 'rows');
        nodes = [nodes; new_nodes];
        nodes_dist = [nodes_dist; ones(length(new_nodes), 1) * dist];
        links = [links; fringe];
        links_dist = [links_dist; ones(size(fringe, 1), 1) * dist];
        if size(nodes, 1) >= K  % nodes enough, extract subgraph
            subgraph = A(nodes, nodes);  % the unweighted subgraph
            adj_before = subgraph;
            subgraph(1, 2) = 0;  % ensure subgraph patterns do not contain information about link existence
            subgraph(2, 1) = 0;
            break
        end
    end

    % Calculate the link-weighted subgraph without symmetrization
    links_ind = sub2ind(size(A), links(:, 1), links(:, 2));
    A_copy = A / (dist + 1);
    A_copy(links_ind) = 1 ./ links_dist;

    % Extract the link-weighted subgraph without symmetrizing
    lweight_subgraph = A_copy(nodes, nodes);

    % Generate enclosing subgraph's vector representation
    [order, classes] = g_label(subgraph);
    if length(order) > K  % if size > K, keep only the top-K vertices and reorder
        order(K + 1: end) = [];
        subgraph = subgraph(order, order);
        lweight_subgraph = lweight_subgraph(order, order);
        [order, classes] = g_label(subgraph);
    end

    subgraph_ordered  = subgraph(order, order);        % binary (directed)

    % Generate enclosing subgraph's vector representation
    ng2v = 2;  % method for transforming a g_labeled subgraph to vector
    switch ng2v
    case 1  % the simplest way -- one dimensional vector by ravelling adjacency matrix
        psubgraph = subgraph(order, order);  % g_labeled subgraph
        sample = psubgraph(triu(logical(ones(size(subgraph))), 1));
        sample(1) = eps;
    case 2  % use link distance-weighted adjcency matrix, performanc is better
        plweight_subgraph = lweight_subgraph(order, order);  % g_labeled link-weighted subgraph
        sample = plweight_subgraph(triu(logical(ones(size(subgraph))), 1));
        sample(1) = eps;  % avoid inf, and more important, avoid empty vector in libsvm format (directly deleting sample(1) results in libsvm format error)
    end
    if length(sample) < D  % add dummy nodes if not enough nodes extracted in subgraph
        sample = [sample; zeros(D - length(sample), 1)];
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


function N = neighbors(fringe, A)
    %  Usage: find the neighbor links of all links in fringe from A

    N = [];
    for no = 1: size(fringe, 1)
        ind = fringe(no, :);
        i = ind(1);
        j = ind(2);
        [~, ij] = find(A(i, :));
        [ji, ~] = find(A(:, j));
        N = [N; [i * ones(length(ij), 1), ij']; [ji, j * ones(length(ji), 1)]];
        N = unique(N, 'rows', 'stable');  % eliminate repeated ones and keep in order
    end
end


function [order, classes] = g_label(subgraph)
    %  Usage: impose a vertex order for a enclosing subgraph using graph labeling

    K = size(subgraph, 1);  % local variable

    % Graph Representation and Distance Calculation
    G = digraph(subgraph);  % Create a directed graph object
    dist_to_1 = distances(G, 1);  % Compute shortest paths from node 1
    dist_to_2 = distances(G, 2);  % Compute shortest paths from node 2

    % Handling Unreachable Nodes
    dist_to_1(isinf(dist_to_1)) = 2 * K;  % replace inf nodes (unreachable from 1 or 2) by an upperbound dist
    dist_to_2(isinf(dist_to_2)) = 2 * K;

    % Initial Vertex Coloring
    avg_dist = sqrt(dist_to_1 .* dist_to_2);  % use geometric mean as the average distance to the link
    [~, ~, avg_dist_colors] = unique(avg_dist);  % f mapping to initial colors

    p_mo = 7;
    % switch different graph labeling methods
    switch p_mo
    case 1
        % use classical wl, no initial colors
        classes = wl_string_lexico(subgraph);
        order = canon(full(subgraph), classes)';
    case 2
        % use wl_hashing, no initial colors
        classes = wl_hashing(subgraph);
        order = canon(full(subgraph), classes)';
    case 3
        % use classical wl, with initial colors
        classes = wl_string_lexico(subgraph, avg_dist_colors);
        order = canon(full(subgraph), classes)';
    case 4
        % use wl_hashing, with initial colors
        classes = wl_hashing(subgraph, avg_dist_colors);
        order = canon(full(subgraph), classes)';
    case 5
        % directly use nauty for canonical labeling
        order = canon(full(subgraph), ones(K, 1))';
    case 6
        % no graph labeling, directly use the predefined order
        order = [1: 1: K];
    case 7
        % palette_wl with initial colors, break ties by nauty
        classes_init = palette_wl(subgraph, avg_dist_colors);
        %classes = palette_wl(subgraph);  % no initial colors
        [order, classes] = canon(full(subgraph), classes_init);
    case 8
        % random labeling
        order = randperm(K);
    end
end
