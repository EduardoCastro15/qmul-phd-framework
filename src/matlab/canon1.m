function [order, final_classes] = canon(subgraph, classes)
    % Canonical labeling using NAUTY with initial coloring (classes)
    K = size(subgraph, 1);
    if nargin < 2
        classes = ones(K, 1);
    end

    % Sort nodes by class to prepare for NAUTY
    [sorted_classes, order] = sort(classes);  % initial ordering by class
    subgraph1 = subgraph(order, order);

    % Prepare NAUTY color breakpoints
    sorted_classes = [sorted_classes; sorted_classes(end) + 1];
    colors_nauty = 1 - diff(sorted_classes);
    num_edges = nnz(subgraph1);
    degrees = sum(subgraph1, 2);

    % Compile canonical.c if needed
    if ~exist(['canonical.' mexext], 'file')
        disp('Compiling canonical.c...')
        !rm canonical.mex*;
        cd software/nauty26r7;
        !cp ../../canonical.c .;
        mex canonical.c nauty.c nautil.c naugraph.c schreier.c naurng.c nausparse.c;
        !cp canonical.mex* ../../;
        cd ../..;
    end

    % Run NAUTY
    clabels = canonical(subgraph1, num_edges, degrees, colors_nauty) + 1;
    order = order(clabels);
    final_classes = sorted_classes(clabels);  % output the class vector as well
end
