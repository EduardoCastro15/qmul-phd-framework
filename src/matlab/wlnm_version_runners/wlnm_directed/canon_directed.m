function [order, final_classes] = canon_directed(subgraph, classes)
    % [OPTIMIZATION]
    K = size(subgraph,1);
    if nargin<2, classes = ones(K,1); end
    [sorted_classes, order] = sort(classes);
    subgraph1 = subgraph(order,order);

    sorted_classes = [sorted_classes; sorted_classes(end)+1];
    colors_nauty   = 1 - diff(sorted_classes);
    num_edges      = nnz(subgraph1);
    degrees        = sum(subgraph1,2);

    assert(~isempty(which(['canonical.' mexext])), ...
        'Nauty MEX missing. Call ensure_nauty_mex() before loops.');

    clabels = canonical(subgraph1, num_edges, degrees, colors_nauty) + 1;
    order   = order(clabels);
    final_classes = sorted_classes(clabels);
end
