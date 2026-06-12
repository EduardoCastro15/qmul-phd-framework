function cmp = compare_empirical_pseudo_webs(empirical_full, pseudo_full)
    %COMPARE_EMPIRICAL_PSEUDO_WEBS Compare a pseudo web against an empirical web.
    %
    % Inputs
    %   empirical_full : n x n binary directed adjacency (ground truth)
    %   pseudo_full    : n x n binary directed adjacency (predicted/reconstructed)
    %
    % Assumes:
    %   - same node ordering in both matrices
    %   - directed graph
    %   - self-loops excluded from evaluation
    %
    % Outputs
    %   TP, FP, FN, TN
    %   TPR, TNR, FPR, FNR
    %   Precision, Recall, F1Score
    %   MCC
    %   TSS
    %   JaccardLinks
    %   EmpiricalLinks, PseudoLinks

    if isempty(empirical_full) || isempty(pseudo_full)
        error('compare_empirical_pseudo_webs:EmptyInput', ...
              'Both empirical_full and pseudo_full must be non-empty.');
    end

    if ~isequal(size(empirical_full), size(pseudo_full))
        error('compare_empirical_pseudo_webs:SizeMismatch', ...
              'empirical_full and pseudo_full must have the same size.');
    end

    empirical_full = spones(sparse(empirical_full));
    pseudo_full    = spones(sparse(pseudo_full));

    n = size(empirical_full, 1);

    % Remove self-loops defensively
    empirical_full = empirical_full - spdiags(diag(empirical_full), 0, n, n);
    pseudo_full    = pseudo_full    - spdiags(diag(pseudo_full),    0, n, n);

    % Evaluate on all off-diagonal directed pairs
    mask = ~eye(n);

    E = logical(full(empirical_full(mask)));
    P = logical(full(pseudo_full(mask)));

    TP = sum(P & E);
    FP = sum(P & ~E);
    FN = sum(~P & E);
    TN = sum(~P & ~E);

    TPR = TP / max(TP + FN, eps);   % recall / sensitivity
    TNR = TN / max(TN + FP, eps);   % specificity
    FPR = FP / max(FP + TN, eps);
    FNR = FN / max(FN + TP, eps);

    precision = TP / max(TP + FP, eps);
    recall    = TPR;
    f1_score  = 2 * (precision * recall) / max(precision + recall, eps);
    mcc_den = sqrt(double(TP + FP) * double(TP + FN) * ...
                   double(TN + FP) * double(TN + FN));
    mcc = (double(TP) * double(TN) - double(FN) * double(FP)) / max(mcc_den, eps);

    TSS = TPR + TNR - 1;

    empirical_links = nnz(empirical_full);
    pseudo_links    = nnz(pseudo_full);

    union_links = TP + FP + FN;
    if union_links > 0
        jaccard_links = TP / union_links;
    else
        jaccard_links = 0;
    end

    cmp = struct();
    cmp.TP              = TP;
    cmp.FP              = FP;
    cmp.FN              = FN;
    cmp.TN              = TN;

    cmp.TPR             = TPR;
    cmp.TNR             = TNR;
    cmp.FPR             = FPR;
    cmp.FNR             = FNR;

    cmp.Precision       = precision;
    cmp.Recall          = recall;
    cmp.F1Score         = f1_score;
    cmp.MCC             = mcc;

    cmp.TSS             = TSS;
    cmp.JaccardLinks    = jaccard_links;

    cmp.EmpiricalLinks  = empirical_links;
    cmp.PseudoLinks     = pseudo_links;
    cmp.LinkDelta       = pseudo_links - empirical_links;
end
