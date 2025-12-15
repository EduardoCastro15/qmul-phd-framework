function [train, test, split_stats] = DivideNet_dir_neg(net, ratioTrain, check_connectivity, adaptive_connectivity, varargin)
    % Divide a directed network into train/test sets with optional backbone regime (subset of TRAIN).
    %
    % Inputs
    %   net                  : n x n binary, directed adjacency
    %   ratioTrain           : overall fraction of links for TRAIN
    %   check_connectivity   : used only in STANDARD (non-backbone) mode
    %   adaptive_connectivity: if true and n<30, disables connectivity check in STANDARD mode
    %
    % Name-Value pairs (optional)
    %   'use_backbone'       : false | true
    %   'inverse_backbone'   : false | true
    %                          false  → prioritize backbone edges in TRAIN (STANDARD)
    %                          true   → prioritize NON-backbone edges in TRAIN (INVERSE)
    %   'ratioBackbone'      : STANDARD mode (inverse=false):
    %                            fraction of BACKBONE (primary) edges to put in TRAIN
    %                          INVERSE mode (inverse=true):
    %                            fraction of NON-backbone (primary) edges to put in TRAIN
    %   'backbone_mask'      : []   logical n x n (if provided, used directly)
    %   'p_values_mat'       : []   sparse/double n x n (PF p-values; requires backbone_regime.m)
    %   'backbone_q'         : 0.05 (BH start)
    %   'backbone_max_q'     : 0.25 (BH ceiling)
    %   'backbone_q_ladder'  : 2.0  (BH step multiplier)
    %   'alpha_fallback'     : []   (quantile fallback if BH yields none)
    %
    % Outputs
    %   train, test          : adjacency matrices (positives only; no negatives here)
    %   split_stats          : struct with detailed counts of links in each category

    % ---- parse options ----
    p = inputParser;
    addParameter(p, 'use_backbone', false);
    addParameter(p, 'inverse_backbone', false);
    addParameter(p, 'ratioBackbone', 0.2);
    addParameter(p, 'backbone_mask', []);
    addParameter(p, 'p_values_mat', []);
    addParameter(p, 'backbone_q', 0.05);
    addParameter(p, 'backbone_max_q', 0.25);
    addParameter(p, 'backbone_q_ladder', 2.0);
    addParameter(p, 'alpha_fallback', []);
    parse(p, varargin{:});
    opt = p.Results;

    % Clamp fractions
    ratioTrain    = max(0, min(1, ratioTrain));
    ratioBackbone = max(0, min(1, opt.ratioBackbone));

    % Normalize adjacency (remove self-loops defensively)
    net = sparse(net);
    n   = size(net,1);
    net = net - spdiags(diag(net), 0, n, n);

    % Optional adaptive connectivity for STANDARD mode
    if ~opt.use_backbone
        if adaptive_connectivity && n < 30
            check_connectivity = false;
            fprintf('[DivideNet] Skipping connectivity check (adaptive mode, n = %d).\n', n);
        end
        fprintf('[DivideNet] STANDARD split. Connectivity check: %d\n', check_connectivity);
    else
        fprintf('[DivideNet] BACKBONE mode ON. ratioTrain=%.3f, ratioBackbone=%.3f\n', ratioTrain, ratioBackbone);
    end

    % ===== BACKBONE MODE (subset of TRAIN) =====
    if opt.use_backbone
        % 1) Build/obtain backbone mask B (true = backbone edge)
        if ~isempty(opt.backbone_mask)
            B = logical(sparse(opt.backbone_mask));
        elseif ~isempty(opt.p_values_mat)
            assert(exist('backbone_regime','file')==2, ...
                   'backbone_regime.m must be on path to threshold p_values_mat.');
            [B, thr, st] = backbone_regime(net, opt.p_values_mat, ...
                               'q', opt.backbone_q, ...
                               'max_q', opt.backbone_max_q, ...
                               'q_ladder', opt.backbone_q_ladder, ...
                               'alpha_fallback', opt.alpha_fallback);
            fprintf('[Backbone] kept %d/%d edges (%.1f%%), method=%s, thr=%g\n', ...
                    st.kept, st.m_edges, 100*st.fraction, st.method, st.thr);
        else
            error('Backbone mode requires either ''backbone_mask'' or ''p_values_mat''.');
        end
        B = B & (net > 0);  % ensure mask only where edges exist

        % 2) Counts and masks
        [i_all, j_all] = find(net);
        m  = numel(i_all);                     % total true links

        if opt.inverse_backbone
            fprintf('[DivideNet] BACKBONE mode (INVERSE). ratioTrain=%.3f, ratioBackbone=%.3f (primary = NON-backbone)\n', ...
                    ratioTrain, ratioBackbone);
        else
            fprintf('[DivideNet] BACKBONE mode (STANDARD). ratioTrain=%.3f, ratioBackbone=%.3f (interpreted as backboneTrainFrac)\n', ...
                    ratioTrain, ratioBackbone);
        end

        % Decide which set is "primary" for TRAIN
        if opt.inverse_backbone
            primary_mask   = net & ~B;         % NON-backbone edges are prioritized
            secondary_mask = B;                % backbone edges used as filler
            primary_label  = 'non-backbone';
            secondary_label= 'backbone';
        else
            primary_mask   = B;                % backbone edges are prioritized
            secondary_mask = net & ~B;         % non-backbone edges used as filler
            primary_label  = 'backbone';
            secondary_label= 'non-backbone';
        end

        [ip, jp]   = find(primary_mask);
        [is, js]   = find(secondary_mask);
        mPrimary   = numel(ip);
        mSecondary = numel(is);

        num_test  = ceil((1 - ratioTrain) * m);
        num_train = m - num_test;

        % --- Determine target_primary based on mode (SYMMETRIC behavior) ---
        if ~opt.inverse_backbone
            % STANDARD backbone mode:
            %   ratioBackbone ≡ backboneTrainFrac = fraction of BACKBONE
            %   edges (primary set) to place in TRAIN, but we keep at
            %   least a small fraction for TEST.

            backboneTrainFrac = ratioBackbone;              % in [0,1]
            backboneTrainFrac = max(0, min(1, backboneTrainFrac));

            % Minimum number of primary edges we want to keep for TEST
            if mPrimary >= 2
                min_primary_test = max(1, ceil(0.10 * mPrimary));   % at least 10% or >=1
            else
                min_primary_test = 0;                               % too few to enforce
            end
            max_primary_train_allowed = max(0, mPrimary - min_primary_test);

            raw_target_primary = round(backboneTrainFrac * mPrimary);
            target_primary     = min([raw_target_primary, max_primary_train_allowed, num_train]);

            % If requested frac > 0 but we ended with 0 and have capacity, ensure at least one
            if target_primary == 0 && backboneTrainFrac > 0 && mPrimary > 0 && num_train > 0
                target_primary = min(1, max_primary_train_allowed);
            end

            fprintf('[DivideNet] STANDARD backbone: backboneTrainFrac=%.2f → target %s TRAIN edges = %d of %d\n', ...
                    backboneTrainFrac, primary_label, target_primary, mPrimary);
        else
            % INVERSE backbone mode (SYMMETRIC):
            %   ratioBackbone ≡ nonbackboneTrainFrac = fraction of NON-backbone
            %   edges (primary set) to place in TRAIN, while keeping at least
            %   a small fraction for TEST (mirrors STANDARD behavior).
            nonbackboneTrainFrac = ratioBackbone;              % in [0,1]
            nonbackboneTrainFrac = max(0, min(1, nonbackboneTrainFrac));

            % Minimum number of primary edges (non-backbone) to keep for TEST
            if mPrimary >= 2
                min_primary_test = max(1, ceil(0.10 * mPrimary));   % at least 10% or >=1
            else
                min_primary_test = 0;                               % too few to enforce
            end
            max_primary_train_allowed = max(0, mPrimary - min_primary_test);

            raw_target_primary = round(nonbackboneTrainFrac * mPrimary);
            target_primary     = min([raw_target_primary, max_primary_train_allowed, num_train]);

            % If requested frac > 0 but we ended with 0 and have capacity, ensure at least one
            if target_primary == 0 && nonbackboneTrainFrac > 0 && mPrimary > 0 && num_train > 0
                target_primary = min(1, max_primary_train_allowed);
            end

            fprintf('[DivideNet] INVERSE backbone: nonbackboneTrainFrac=%.2f → target %s TRAIN edges = %d of %d\n', ...
                    nonbackboneTrainFrac, primary_label, target_primary, mPrimary);
        end

        % 3) Select primary portion for TRAIN (up to target_primary)
        perm_p  = randperm(mPrimary);
        sel_p   = perm_p(1:min(target_primary, mPrimary));
        i_pr_tr = ip(sel_p); j_pr_tr = jp(sel_p);

        % 4) Fill the remaining TRAIN slots with SECONDARY edges first
        remaining_needed = num_train - numel(sel_p);
        i_fill = []; j_fill = [];

        if remaining_needed < 0
            % Rare case: target_primary > num_train
            sel_p   = sel_p(1:num_train);
            i_pr_tr = ip(sel_p); j_pr_tr = jp(sel_p);
            remaining_needed = 0;
        end

        if remaining_needed > 0
            take_sec = min(remaining_needed, mSecondary);
            if take_sec > 0
                perm_s = randperm(mSecondary);
                sel_s  = perm_s(1:take_sec);
                i_fill = is(sel_s); j_fill = js(sel_s);
            end

            remaining_needed = num_train - (numel(i_pr_tr) + numel(i_fill));
            if remaining_needed > 0
                % Not enough secondary edges → gracefully top up with extra PRIMARY
                leftover_primary_idx = setdiff(1:mPrimary, sel_p, 'stable');
                if ~isempty(leftover_primary_idx)
                    add_p = min(remaining_needed, numel(leftover_primary_idx));
                    extra = leftover_primary_idx(randperm(numel(leftover_primary_idx), add_p));
                    i_pr_tr = [i_pr_tr; ip(extra)];
                    j_pr_tr = [j_pr_tr; jp(extra)];
                    remaining_needed = num_train - (numel(i_pr_tr) + numel(i_fill));
                    if remaining_needed > 0
                        warning('[DivideNet] TRAIN still short by %d edges even after topping up with %s edges (check ratios).', ...
                                remaining_needed, primary_label);
                    end
                else
                    warning('[DivideNet] No leftover %s edges to top up; TRAIN short by %d edges.', ...
                            primary_label, remaining_needed);
                end
            end
        end

        % 5) Assemble TRAIN and TEST
        train = sparse([i_pr_tr; i_fill], [j_pr_tr; j_fill], 1, n, n);
        test  = net - train;  % train ⊆ net

        % 6) Reporting
        n_train = nnz(train);
        n_test  = nnz(test);

        % Backbone / non-backbone masks
        backbone_mask    = B;
        nonbackbone_mask = net & ~B;

        n_bb_total  = nnz(backbone_mask);
        n_nb_total  = nnz(nonbackbone_mask);

        n_bb_train  = nnz(train & backbone_mask);
        n_nb_train  = nnz(train & nonbackbone_mask);
        n_bb_test   = nnz(test  & backbone_mask);
        n_nb_test   = nnz(test  & nonbackbone_mask);

        split_stats = struct( ...
            'TotalLinks',           m, ...
            'TrainLinks',           n_train, ...
            'TestLinks',            n_test, ...
            'BackboneTotal',        n_bb_total, ...
            'NonBackboneTotal',     n_nb_total, ...
            'BackboneTrainLinks',   n_bb_train, ...
            'NonBackboneTrainLinks',n_nb_train, ...
            'BackboneTestLinks',    n_bb_test, ...
            'NonBackboneTestLinks', n_nb_test ...
        );

        n_primary_in_train   = nnz(train & primary_mask);
        n_secondary_in_train = nnz(train & secondary_mask);

        fprintf('[DivideNet] TRAIN total: %d (requested %d)\n', n_train, num_train);
        fprintf('[DivideNet]  ├─ %s in TRAIN: %d (target %d of %d)\n', ...
                primary_label, n_primary_in_train, target_primary, mPrimary);
        fprintf('[DivideNet]  └─ %s in TRAIN: %d of %d\n', ...
                secondary_label, n_secondary_in_train, mSecondary);
        fprintf('[DivideNet] TEST total: %d\n', n_test);
        return;
    end

    % ===== STANDARD RANDOM REMOVAL (original logic) =====
    [i, j] = find(net);               % all directed links (i → j)
    linklist = [i, j];
    m = size(linklist, 1);
    num_test  = ceil((1 - ratioTrain) * m);

    perm = randperm(m);
    test = sparse(n, n);
    train = net;
    accepted = 0; attempts = 0;

    for idx = perm
        if accepted >= num_test, break; end
        u = linklist(idx, 1); v = linklist(idx, 2);
        if train(u, v) == 0, continue; end
        train(u, v) = 0;  % tentative removal
        attempts = attempts + 1;

        if ~check_connectivity || hasPath(train, u, v)
            test(u, v) = 1;
            accepted = accepted + 1;
        else
            train(u, v) = 1;  % restore
        end
    end

    if accepted == 0
        warning('[DivideNet] No test links were accepted. Consider disabling connectivity check or using adaptive mode.');
    end
    fprintf('[DivideNet] Test links accepted: %d / %d (%.1f%%)\n', accepted, num_test, 100 * accepted / max(1,num_test));
    fprintf('[DivideNet] Attempts made: %d | Failed attempts: %d\n', attempts, attempts - accepted);

    n_train = nnz(train);
    n_test  = nnz(test);
    m       = nnz(net);

    split_stats = struct( ...
        'TotalLinks',            m, ...
        'TrainLinks',            n_train, ...
        'TestLinks',             n_test, ...
        'BackboneTotal',         0, ...
        'NonBackboneTotal',      m, ...
        'BackboneTrainLinks',    0, ...
        'NonBackboneTrainLinks', n_train, ...
        'BackboneTestLinks',     0, ...
        'NonBackboneTestLinks',  n_test ...
    );
end

% Helper: reachability check (STANDARD mode only)
function reachable = hasPath(adj, u, v)
    reachable = false;
    visited = false(size(adj, 1), 1);
    queue = u;

    while ~isempty(queue)
        current = queue(1); queue(1) = [];
        if current == v, reachable = true; return; end
        if ~visited(current)
            visited(current) = true;
            neighbors = find(adj(current, :) > 0);
            queue = [queue; neighbors(~visited(neighbors))'];
        end
    end
end
