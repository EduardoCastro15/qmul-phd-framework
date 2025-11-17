function [B, thr, stats] = backbone_regime(net, p_values_mat, varargin)
    %BACKBONE_REGIME  Build a backbone mask from PF p-values with adaptive thresholding.
    %
    % Inputs
    %   net           : n×n binary, directed adjacency (original edges)
    %   p_values_mat  : n×n sparse of PF p-values on edges, 0 elsewhere
    %
    % Name-Value options (all optional)
    %   'q'           : initial BH FDR level (default 0.05)
    %   'max_q'       : max BH FDR level to try if no discoveries (default 0.25)
    %   'q_ladder'    : multiplicative step when increasing q (default 2.0)
    %   'min_keep'    : minimum number of edges we’d like to keep if possible (default  max(5, ceil(0.01*m)))
    %   'alpha_fallback' : fallback quantile for p-values if BH finds none.
    %                      If empty [], we compute α adaptively from size/connectance (default [])
    %
    % Outputs
    %   B     : n×n sparse logical mask (1 = backbone edge)
    %   thr   : scalar p-value threshold used (NaN if quantile fallback)
    %   stats : struct with counts and method used

    opts = inputParser;
    addParameter(opts, 'q', 0.05);
    addParameter(opts, 'max_q', 0.25);
    addParameter(opts, 'q_ladder', 2.0);
    addParameter(opts, 'min_keep', []);
    addParameter(opts, 'alpha_fallback', []);
    parse(opts, varargin{:});
    q0      = opts.Results.q;
    qmax    = opts.Results.max_q;
    qstep   = opts.Results.q_ladder;
    alphaFB = opts.Results.alpha_fallback;

    % Edge list (directed)
    [i, j] = find(net);
    m = numel(i);
    if isempty(opts.Results.min_keep)
        min_keep = max(5, ceil(0.01*m));  % keep at least ~1% or >=5 edges if possible
    else
        min_keep = opts.Results.min_keep;
    end

    % Gather p-values on existing edges
    idx  = sub2ind(size(net), i, j);
    pvec = full(p_values_mat(idx));

    % Safety: remove NaNs/negatives/out-of-range (clamp)
    pvec(~isfinite(pvec)) = 1;
    pvec = max(0, min(1, pvec));

    % === 1) BH (adaptive q ladder) ===
    thr = NaN; sel = false(size(pvec));
    method = 'bh';

    q = q0;
    while q <= qmax
        [selBH, thrBH] = bh_select(pvec, q);
        if nnz(selBH) >= min_keep
            sel = selBH; thr = thrBH; break;
        end
        % If not enough, still accept if nonzero and min_keep is too strict
        if nnz(selBH) > 0 && nnz(selBH) >= max(1, floor(0.002*m))  % >=0.2% of edges
            sel = selBH; thr = thrBH; break;
        end
        q = q * qstep;
    end

    % === 2) Fallback to quantile if BH gave nothing ===
    if ~any(sel)
        method = 'quantile';
        if isempty(alphaFB)
            % Adaptive α: increases gently with connectance, but capped
            n  = size(net,1);
            C  = m / max(1, n*(n-1));      % connectance
            % Example adaptive rule: alpha ≈ base + slope*sqrt(C) with caps
            alpha = min(0.25, max(0.05, 0.10 + 0.35*sqrt(C)));  % 5%–25%
        else
            alpha = alphaFB;
        end
        thr = quantile(pvec, alpha);
        sel = pvec <= thr;
        % ensure we keep at least a few edges
        if nnz(sel) == 0 && m > 0
            [~, ord] = sort(pvec, 'ascend');
            k = min_keep;
            sel(ord(1:min(k,m))) = true;
            thr = pvec(ord(min(k,m)));
        end
    end

    % Build mask
    B = sparse(i(sel), j(sel), true, size(net,1), size(net,2));

    % Pack stats
    stats = struct( ...
        'm_edges', m, ...
        'kept', nnz(sel), ...
        'fraction', nnz(sel)/max(1,m), ...
        'method', method, ...
        'q_start', q0, ...
        'q_used', min(q, qmax), ...
        'thr', thr ...
    );
end

function [sel, thr] = bh_select(p, q)
    % Standard Benjamini–Hochberg (one-sided) on vector p; returns selection mask and threshold.
    [ps, ord] = sort(p(:), 'ascend');
    m = numel(ps);
    k = find(ps <= ( (1:m)'/m ) * q, 1, 'last');
    if isempty(k)
        sel = false(size(p));
        thr = NaN;
    else
        thr = ps(k);
        cut = false(size(p));
        cut(ord(1:k)) = true;
        sel = cut;
    end
end
