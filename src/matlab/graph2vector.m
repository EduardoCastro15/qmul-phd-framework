function [data, label] = graph2vector(pos, neg, A, K, useParallel)
    %  Usage: to convert links' enclosing subgraphs (both pos
    %         and neg) into real vectors
    %  --Input--
    %       -pos: indices of positive links
    %       -neg: indices of negative links
    %       -A: the observed graph's adjacency matrix from which to
    %           to extract subgraph features
    %       -K: the number of nodes in each link's subgraph
    %       -useParallel: flag to enable or disable parallel pool
    %  --Output--
    %       -data: the constructed training data, each row is a
    %         link's vector representation
    %       -label: a column vector of links' labels
    %
    %  Partly adapted from the codes of
    %  Lu 2011, Link prediction in complex networks: A survey.
    %  Muhan Zhang, Washington University in St. Louis
    %
    %  *author: Jorge Eduardo Castro Cruces, Queen Mary University of London

    all = [pos; neg];
    pos_size = size(pos, 1);
    neg_size = size(neg, 1);
    all_size = pos_size + neg_size;

    % Labels
    label = [ones(pos_size, 1); zeros(neg_size, 1)];

    % Preallocate data
    d = K * (K - 1) / 2;
    data = zeros(all_size, d);

    fprintf('Subgraph Pattern Encoding Begins for %d samples...\n', all_size);
    tic

    pool_created = false;  % track if we open a pool ourselves

    % --- Parallel or serial processing depending on flag ---
    if useParallel && isempty(gcp('nocreate'))
        parpool(feature('numcores'));
        pool_created = true;
        % parpool('local', str2double(getenv('NSLOTS')));
    end
    
    if useParallel
        parfor i = 1:all_size
            ind = all(i, :);
            data(i, :) = subgraph2vector(ind, A, K);
            if mod(i, floor(all_size / 10)) == 0
                fprintf('Progress: %d%% — Elapsed: %.1f seconds\n', round(100*i/all_size), toc);
            end
        end
    else
        for i = 1:all_size
            ind = all(i, :);
            data(i, :) = subgraph2vector(ind, A, K);
            if mod(i, floor(all_size / 10)) == 0
                fprintf('Progress: %d%% — Elapsed: %.1f seconds\n', round(100*i/all_size), toc);
            end
        end
    end

    % --- Close pool if we opened it ---
    if pool_created
        delete(gcp('nocreate'));
    end
end


function sample = subgraph2vector(ind, A, K)
    %  Usage: to extract a subgraph of K nodes from a given adjacency matrix A
    %  --Input--
    %       -ind: indices of the links to be extracted
    %       -A: the observed graph's adjacency matrix from which to
    %           to extract subgraph features
    %       -K: the number of nodes in each link's subgraph
    %  --Output--
    %       -sample: the constructed subgraph vector representation
    %
    %  Partly adapted from the codes of
    %  Lu 2011, Link prediction in complex networks: A survey.
    %  Muhan Zhang, Washington University in St. Louis
    %
    %  *author: Jorge Eduardo Castro Cruces, Queen Mary University of London

    D = K * (K - 1) / 2;    % the length of output vector
    max_depth = 5;          % New: limit maximum fringe expansion depth

    % Extract a subgraph of K nodes
    links = ind;
    links_dist = 0;         % the graph distance to the initial link
    dist = 0;
    fringe = ind;
    nodes = [ind(1); ind(2)];
    nodes_dist = [0; 0];
    visited_set = containers.Map('KeyType','char','ValueType','logical');
    visited_set(sprintf('%d-%d', ind(1), ind(2))) = true;  % Add initial link to the visited set

    while 1
        dist = dist + 1;
        fringe = neighbors_vectorized(fringe, A);

        if isempty(fringe)
            fringe = zeros(0, 2);
        end

        % Use fast deduplication via a visited set
        new_fringe = zeros(0,2);
        for row = 1:size(fringe,1)
            key = sprintf('%d-%d', fringe(row,1), fringe(row,2));
            if ~isKey(visited_set, key)
                visited_set(key) = true;
                new_fringe(end+1, :) = fringe(row, :);
            end
        end
        fringe = new_fringe;

        if isempty(fringe) || dist > max_depth  % no more neighbors or reached max depth, break
            subgraph = A(nodes, nodes);
            subgraph(1, 2) = 0;                 % ensure subgraph patterns do not contain information about link existence
            subgraph(2, 1) = 0;
            break
        end

        new_nodes = setdiff(fringe(:), nodes, 'rows');
        nodes = [nodes; new_nodes];
        nodes_dist = [nodes_dist; ones(length(new_nodes), 1) * dist];
        links = [links; fringe];
        links_dist = [links_dist; ones(size(fringe, 1), 1) * dist];

        if numel(nodes) >= K              % nodes enough, extract subgraph
            subgraph = A(nodes, nodes);     % the unweighted subgraph
            subgraph(1, 2) = 0;             % ensure subgraph patterns do not contain information about link existence
            subgraph(2, 1) = 0;
            break
        end
    end

    % Calculate the link-weighted subgraph, each entry in the adjacency matrix is weighted by the inverse of its distance to the target link
    links_ind = sub2ind(size(A), links(:, 1), links(:, 2));
    A_copy = A / (dist + 1);                                % if a link between two existing nodes < dist+1, it must be in 'links'. The only links not in 'links' are the dist+1 links between some farthest nodes in 'nodes', so here we weight them by dist+1
    A_copy(links_ind) = 1 ./ links_dist;
    A_copy_u = max(triu(A_copy, 1), tril(A_copy, -1)');     % for links (i, j) and (j, i), keep the smallest dist
    A_copy = A_copy_u + A_copy_u';
    lweight_subgraph = A_copy(nodes, nodes);

    % Calculate the graph labeling of the subgraph
    order = g_label(subgraph);
    if length(order) > K  % if size > K, keep only the top-K vertices and reorder
        order(K + 1: end) = [];
        subgraph = subgraph(order, order);
        lweight_subgraph = lweight_subgraph(order, order);
        order = g_label(subgraph);
    end

    % Generate enclosing subgraph's vector representation
    % method for transforming a g_labeled subgraph to vector
    % use link distance-weighted adjcency matrix, performanc is better
    plweight_subgraph = lweight_subgraph(order, order);                     % g_labeled link-weighted subgraph
    sample = plweight_subgraph(triu(logical(ones(size(subgraph))), 1));
    sample(1) = eps;                                                        % avoid inf, and more important, avoid empty vector in libsvm format (directly deleting sample(1) results in libsvm format error)

    if numel(sample) < D                                                           % add dummy nodes if not enough nodes extracted in subgraph
        sample(end+1:D) = 0;
    end
end


function N = neighbors_vectorized(fringe, A)
    % Find neighbor links of all links in the fringe from A

    N = [];

    % For each link in the fringe
    for i = 1:size(fringe, 1)
        u = fringe(i, 1);
        v = fringe(i, 2);

        % Find successors of node i (i -> ?)
        succ = find(A(u, :));
        if ~isempty(succ)
            N = [N; [repmat(u, numel(succ), 1), succ']];
        end

        % Find predecessors of node j (? -> j)
        pred = find(A(:, v));
        if ~isempty(pred)
            N = [N; [pred, repmat(v, numel(pred), 1)]];
        end
    end

    % Remove duplicates
    N = unique(N, 'rows');
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
