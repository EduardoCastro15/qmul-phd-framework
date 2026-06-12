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
    %   DegreeStd / DegreeCV / DegreeGini
    %   GeneralityStd / GeneralityCV / GeneralityGini
    %   VulnerabilityStd / VulnerabilityCV / VulnerabilityGini
    %   TrophicLevelStd / TrophicLevelCV / TrophicLevelRange
    %   MeanLocalClustering / Transitivity / TriangleDensity
    %   MeanDietOverlap
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
    degree_std  = finite_std(total_degree);
    degree_cv   = finite_cv(total_degree);
    degree_gini = gini_coefficient(total_degree);

    generality_std  = finite_std(generality);
    generality_cv   = finite_cv(generality);
    generality_gini = gini_coefficient(generality);

    vulnerability_std  = finite_std(vulnerability);
    vulnerability_cv   = finite_cv(vulnerability);
    vulnerability_gini = gini_coefficient(vulnerability);

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
        finite_trophic_level = trophic_level(isfinite(trophic_level));
        mean_trophic_level = mean(finite_trophic_level);
        trophic_level_std = finite_std(finite_trophic_level);
        trophic_level_cv = finite_cv(finite_trophic_level);
        trophic_level_range = max(finite_trophic_level) - min(finite_trophic_level);
    else
        mean_trophic_level = NaN;
        trophic_level_std = NaN;
        trophic_level_cv = NaN;
        trophic_level_range = NaN;
    end

    closure_metrics = compute_closure_metrics(A);
    mean_diet_overlap = compute_mean_diet_overlap(A);

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

    metrics.DegreeStd            = degree_std;
    metrics.DegreeCV             = degree_cv;
    metrics.DegreeGini           = degree_gini;
    metrics.GeneralityStd        = generality_std;
    metrics.GeneralityCV         = generality_cv;
    metrics.GeneralityGini       = generality_gini;
    metrics.VulnerabilityStd     = vulnerability_std;
    metrics.VulnerabilityCV      = vulnerability_cv;
    metrics.VulnerabilityGini    = vulnerability_gini;

    metrics.TrophicLevelStd      = trophic_level_std;
    metrics.TrophicLevelCV       = trophic_level_cv;
    metrics.TrophicLevelRange    = trophic_level_range;

    metrics.MeanLocalClustering  = closure_metrics.MeanLocalClustering;
    metrics.Transitivity         = closure_metrics.Transitivity;
    metrics.NumTriangles         = closure_metrics.NumTriangles;
    metrics.TriangleDensity      = closure_metrics.TriangleDensity;
    metrics.MeanDietOverlap      = mean_diet_overlap;

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

function val = finite_std(x)
    x = x(:);
    x = x(isfinite(x));
    if numel(x) <= 1
        val = 0;
    else
        val = std(double(x), 0);
    end
end

function val = finite_cv(x)
    x = x(:);
    x = x(isfinite(x));
    if isempty(x)
        val = NaN;
        return;
    end

    mu = mean(double(x));
    if abs(mu) <= eps
        val = 0;
    else
        val = finite_std(x) / abs(mu);
    end
end

function g = gini_coefficient(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        g = NaN;
        return;
    end

    x = sort(max(x, 0));
    n = numel(x);
    sx = sum(x);
    if sx <= eps
        g = 0;
    else
        g = (2 * sum((1:n)' .* x) / (n * sx)) - ((n + 1) / n);
    end
end

function closure = compute_closure_metrics(A)
    n = size(A, 1);
    B = spones(A | A');
    B = B - spdiags(diag(B), 0, n, n);
    B = spones(B);

    undirected_degree = full(sum(B, 2));
    connected_triples = sum(undirected_degree .* max(undirected_degree - 1, 0) / 2);

    if n >= 3 && nnz(B) > 0
        BB = B * B;
        num_triangles = full(sum(sum((BB .* B))) / 6);
    else
        num_triangles = 0;
    end

    local_clustering = NaN(n, 1);
    for i = 1:n
        neighbors = find(B(:, i));
        k = numel(neighbors);
        if k >= 2
            subgraph = B(neighbors, neighbors);
            local_edges = nnz(triu(subgraph, 1));
            local_clustering(i) = (2 * local_edges) / (k * (k - 1));
        end
    end

    finite_clustering = local_clustering(isfinite(local_clustering));
    if isempty(finite_clustering)
        mean_local_clustering = 0;
    else
        mean_local_clustering = mean(finite_clustering);
    end

    if connected_triples > 0
        transitivity = (3 * num_triangles) / connected_triples;
    else
        transitivity = 0;
    end

    if n >= 3
        triangle_density = num_triangles / nchoosek(n, 3);
    else
        triangle_density = 0;
    end

    closure = struct( ...
        'MeanLocalClustering', mean_local_clustering, ...
        'Transitivity', transitivity, ...
        'NumTriangles', num_triangles, ...
        'TriangleDensity', triangle_density);
end

function mean_overlap = compute_mean_diet_overlap(A)
    prey_per_consumer = full(sum(A, 1))';
    consumer_idx = find(prey_per_consumer > 0);
    num_consumers = numel(consumer_idx);

    if num_consumers < 2
        mean_overlap = 0;
        return;
    end

    diets = spones(A(:, consumer_idx));
    shared_prey = full(diets' * diets);
    diet_size = full(sum(diets, 1))';
    union_prey = diet_size + diet_size' - shared_prey;

    upper_mask = triu(true(num_consumers), 1) & union_prey > 0;
    overlaps = shared_prey(upper_mask) ./ union_prey(upper_mask);

    if isempty(overlaps)
        mean_overlap = 0;
    else
        mean_overlap = mean(overlaps);
    end
end
