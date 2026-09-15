function [levels, stats] = compute_networkx_trophic_levels_v2(A, highPrecision)
% Validated unweighted trophic levels. A(resource, consumer)=1.
% No regularization, artificial basal links, upper cap, or partial LCC mean.
    if nargin < 2, highPrecision = 'auto'; end
    highPrecision = validatestring(highPrecision, {'auto','off','required'});
    hasSymbolic = exist('sym', 'file') == 2 && license('test', 'Symbolic_Toolbox');
    if strcmp(highPrecision, 'required') && ~hasSymbolic
        error('WLNM:TrophicPrecisionUnavailable', 'Symbolic Math Toolbox is required.');
    end
    validateattributes(A, {'numeric','logical'}, {'2d','square','real','finite','nonnegative'});
    n = size(A,1);
    A = spones(sparse(A));
    if n > 0, A = A - spdiags(diag(A),0,n,n); end
    levels = NaN(n,1);
    stats = struct('NumSpeciesFull',n,'NumSpeciesLargest',0, ...
        'NumSpeciesWithLevel',0,'LargestFraction',NaN,'StatusCode',1, ...
        'NumBasal',0,'NumUnreachable',0,'ReciprocalCondition',NaN, ...
        'ScaledResidual',NaN,'ResidualNorm',NaN,'CandidateMin',NaN, ...
        'CandidateMax',NaN,'AboveLegacyCap',false,'PrecisionDigits',0, ...
        'PrecisionRelativeDifference',NaN,'SolverCode',0,'FailureReason','empty_graph');
    if n == 0, return; end
    bins = conncomp(graph(spones(A | A')));
    [~, largest] = max(accumarray(bins(:),1)); % Same first-component tie rule as v1.
    idx = find(bins(:) == largest);
    B = A(idx,idx); m = numel(idx);
    stats.NumSpeciesLargest = m; stats.LargestFraction = m/n;
    degree = full(sum(B,1))';
    basal = degree == 0;
    stats.NumBasal = nnz(basal);
    reached = basal;
    while true
        next = reached | (B' * double(reached) > 0);
        if isequal(next,reached), break; end
        reached = next;
    end
    stats.NumUnreachable = nnz(~reached);
    if ~any(basal)
        stats.StatusCode = 6; stats.FailureReason = 'no_basal_in_lcc'; return;
    elseif ~all(reached)
        stats.StatusCode = 7; stats.FailureReason = 'basal_unreachable_in_lcc'; return;
    end
    M = speye(m) - spdiags(1./max(degree,1),0,m,m)*B';
    b = ones(m,1);
    stats.ReciprocalCondition = rcond(full(M));
    stats.PrecisionDigits = 16; stats.SolverCode = 1;
    [oldMsg,oldID] = lastwarn;
    restoreWarning = onCleanup(@() lastwarn(oldMsg,oldID)); %#ok<NASGU>
    lastwarn('');
    try
        candidate = M\b;
        [solveWarning,~] = lastwarn;
    catch
        candidate = NaN(m,1); solveWarning = 'solve_failed';
    end
    stats = candidate_stats(stats,M,b,candidate,m);
    good = acceptable(candidate,stats.ScaledResidual);
    needsPrecision = ~good || ~isempty(solveWarning) || ...
        ~isfinite(stats.ReciprocalCondition) || stats.ReciprocalCondition <= 1e-10;
    if needsPrecision
        stats.StatusCode = 2; stats.FailureReason = 'numerical_unresolved';
        if strcmp(highPrecision,'off') || ~hasSymbolic, return; end
        try
            % Construct exact rational coefficients BEFORE division.
            exactM = sym(eye(m)) - diag(1./sym(max(degree,1)))*sym(full(B'));
            exactB = sym(ones(m,1));
            oldDigits = digits; restoreDigits = onCleanup(@() digits(oldDigits)); %#ok<NASGU>
            digits(50); t50 = vpa(exactM,50)\vpa(exactB,50);
            digits(100); t100 = vpa(exactM,100)\vpa(exactB,100);
            rel = abs(vpa(t50,100)-t100)./max(abs(t100),sym(1));
            stats.PrecisionRelativeDifference = double(max(rel));
            hpResidual = double(norm(vpa(exactM,100)*t100-exactB,1) / ...
                (norm(vpa(exactM,100),1)*norm(t100,1)+norm(exactB,1)));
            candidate = double(t100);
            stats.PrecisionDigits = 100; stats.SolverCode = 2;
            stats = candidate_stats(stats,M,b,candidate,m);
            good = acceptable(candidate,stats.ScaledResidual) && ...
                isfinite(hpResidual) && hpResidual <= 1e-12 && ...
                isfinite(stats.PrecisionRelativeDifference) && stats.PrecisionRelativeDifference <= 1e-10;
            if ~good, return; end
        catch
            stats.FailureReason = 'high_precision_failed'; return;
        end
    end
    levels(idx) = candidate;
    stats.NumSpeciesWithLevel = m;
    stats.StatusCode = 0; stats.FailureReason = 'ok';
end

function tf = acceptable(t,residual)
    tf = all(isfinite(t)) && all(t >= 1-1e-8) && isfinite(residual) && residual <= 1e-12;
end

function s = candidate_stats(s,M,b,t,m)
    s.CandidateMin = min(t); s.CandidateMax = max(t);
    s.AboveLegacyCap = any(t > max(20,m));
    s.ResidualNorm = norm(M*t-b,1);
    denominator = norm(M,1)*norm(t,1)+norm(b,1);
    s.ScaledResidual = s.ResidualNorm/denominator;
    if ~isfinite(denominator), s.ScaledResidual = NaN; end
end
