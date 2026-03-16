function metrics = compute_foodweb_metrics(A)
    %COMPUTE_FOODWEB_METRICS Compute web-level and node-level food-web metrics.
    %
    % Assumes adjacency orientation:
    %   A(i,j) = 1  means  prey/resource i  ->  predator/consumer j
    %
    % Therefore:
    %   - column sum = number of prey/resources consumed by species j
    %                = consumer-side diet breadth
    %   - row sum    = number of predators/consumers feeding on species i
    %                = resource-side vulnerability
    %
    % Returned fields include both summary metrics and node-level vectors.
    %
    % Web-level outputs
    %   NumSpecies
    %   NumLinks
    %   Connectance
    %   MeanTrophicLevel
    %   MeanDegree
    %   MeanGenerality
    %   MeanVulnerability
    %   NumBasal / NumIntermediate / NumTop / NumIsolate
    %   PropBasal / PropIntermediate / PropTop / PropIsolate
    %
    % Node-level outputs
    %   InDegree        : number of prey consumed by each species
    %   OutDegree       : number of predators consuming each species
    %   TotalDegree
    %   Generality      : same as InDegree under this adjacency convention
    %   Vulnerability   : same as OutDegree under this adjacency convention
    %   TrophicLevel
    %   BasalMask / IntermediateMask / TopMask / IsolateMask

    % --- Validate and binarize ---
    if isempty(A)
        error('compute_foodweb_metrics:EmptyInput', 'Input adjacency matrix A is empty.');
    end
    if size(A,1) ~= size(A,2)
        error('compute_foodweb_metrics:NonSquare', 'Adjacency matrix A must be square.');
    end

    A = spones(sparse(A));  % binary sparse
    n = size(A,1);

    % Remove self-loops defensively
    A = A - spdiags(diag(A), 0, n, n);
    A = spones(A);

    % --- Basic counts ---
    L = nnz(A);

    if n > 1
        C = L / (n * (n - 1));
    else
        C = 0;
    end

    % Under A(prey, predator)=1
    in_degree  = full(sum(A, 1))';  % prey/resources per consumer
    out_degree = full(sum(A, 2));   % predators/consumers per resource
    total_degree = in_degree + out_degree;

    % Food-web role classes
    basal_mask        = (in_degree == 0) & (out_degree > 0);
    top_mask          = (in_degree > 0)  & (out_degree == 0);
    intermediate_mask = (in_degree > 0)  & (out_degree > 0);
    isolate_mask      = (in_degree == 0) & (out_degree == 0);

    num_basal        = sum(basal_mask);
    num_top          = sum(top_mask);
    num_intermediate = sum(intermediate_mask);
    num_isolate      = sum(isolate_mask);

    prop_basal        = num_basal / max(n, 1);
    prop_top          = num_top / max(n, 1);
    prop_intermediate = num_intermediate / max(n, 1);
    prop_isolate      = num_isolate / max(n, 1);

    % Standard food-web node properties
    generality    = in_degree;   % prey/resources per consumer
    vulnerability = out_degree;  % consumers/predators per resource

    if any(generality > 0)
        mean_generality = mean(generality(generality > 0));
    else
        mean_generality = 0;
    end

    if any(vulnerability > 0)
        mean_vulnerability = mean(vulnerability(vulnerability > 0));
    else
        mean_vulnerability = 0;
    end

    mean_degree = mean(total_degree);

    % ------------------------------------------------------------
    % Prey-averaged trophic levels (stable implementation)
    %
    % For consumer/predator j:
    %   TL_j = 1 + mean( TL of prey of j )
    %
    % With A(prey, predator)=1:
    %   A' has rows = predators, cols = prey
    % so row-normalized A' gives prey contributions to each predator diet.
    %
    % We first try the linear solve:
    %   (I - P) TL = 1
    % If that system is ill-conditioned or gives implausible values,
    % we fall back to a fixed-point iteration. If that also fails,
    % trophic levels are returned as NaN rather than huge unstable values.
    % ------------------------------------------------------------
    prey_count_per_consumer = in_degree;     % number of prey for each consumer
    basal_or_isolate_mask = (prey_count_per_consumer == 0);

    % Build P so that (P * TL)_j = average trophic level of prey of j
    denom = max(prey_count_per_consumer, 1);
    P = spdiags(1 ./ denom, 0, n, n) * A';   % row-normalized by consumer

    I = speye(n);
    M = I - P;

    trophic_level = NaN(n,1);
    solve_ok = false;

    % ---- Attempt 1: direct solve if matrix is reasonably conditioned ----
    try
        r = rcond(full(M));
        if isfinite(r) && r > 1e-10
            tl_try = M \ ones(n,1);

            % enforce basal/isolate species at level 1
            tl_try(basal_or_isolate_mask) = 1;

            % accept only finite and biologically plausible values
            if all(isfinite(tl_try)) && all(tl_try >= 1 - 1e-8) && all(tl_try <= max(20, n))
                trophic_level = tl_try;
                solve_ok = true;
            end
        end
    catch
        solve_ok = false;
    end

    % ---- Attempt 2: fixed-point iteration fallback ----
    if ~solve_ok
        tl_old = ones(n,1);
        tl_old(basal_or_isolate_mask) = 1;

        max_iter = 5000;
        tol = 1e-9;
        converged = false;

        for iter = 1:max_iter
            tl_new = 1 + P * tl_old;
            tl_new(basal_or_isolate_mask) = 1;

            if any(~isfinite(tl_new)) || any(abs(tl_new) > 1e6)
                converged = false;
                break;
            end

            if max(abs(tl_new - tl_old)) < tol
                converged = true;
                break;
            end

            tl_old = tl_new;
        end

        if converged && all(tl_new >= 1 - 1e-8) && all(tl_new <= max(20, n))
            trophic_level = tl_new;
            solve_ok = true;
        else
            warning('compute_foodweb_metrics:TrophicLevelUnstable', ...
                ['Trophic level calculation was unstable for this network. ' ...
                 'Returning NaN trophic levels.']);
            trophic_level = NaN(n,1);
            trophic_level(basal_or_isolate_mask) = 1;
        end
    end

    % Final cleanup
    trophic_level(~isfinite(trophic_level)) = NaN;

    if any(isfinite(trophic_level))
        mean_trophic_level = mean(trophic_level(isfinite(trophic_level)));
    else
        mean_trophic_level = NaN;
    end

    % --- Pack output ---
    metrics = struct();

    % Web-level
    metrics.NumSpecies           = n;
    metrics.NumLinks             = L;
    metrics.Connectance          = C;
    metrics.MeanTrophicLevel     = mean_trophic_level;
    metrics.MeanDegree           = mean_degree;
    metrics.MeanGenerality       = mean_generality;
    metrics.MeanVulnerability    = mean_vulnerability;

    metrics.NumBasal             = num_basal;
    metrics.NumIntermediate      = num_intermediate;
    metrics.NumTop               = num_top;
    metrics.NumIsolate           = num_isolate;

    metrics.PropBasal            = prop_basal;
    metrics.PropIntermediate     = prop_intermediate;
    metrics.PropTop              = prop_top;
    metrics.PropIsolate          = prop_isolate;

    % Node-level
    metrics.InDegree             = in_degree;
    metrics.OutDegree            = out_degree;
    metrics.TotalDegree          = total_degree;
    metrics.Generality           = generality;
    metrics.Vulnerability        = vulnerability;
    metrics.TrophicLevel         = trophic_level;

    metrics.BasalMask            = basal_mask;
    metrics.IntermediateMask     = intermediate_mask;
    metrics.TopMask              = top_mask;
    metrics.IsolateMask          = isolate_mask;
end
