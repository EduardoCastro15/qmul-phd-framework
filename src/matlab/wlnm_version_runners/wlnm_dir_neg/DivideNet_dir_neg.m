function [train, test] = DivideNet_dir_neg(net, ratioTrain, check_connectivity, adaptive_connectivity, varargin)
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
    %   'ratioBackbone'      : 0.2  (fraction of TOTAL links to force into TRAIN from backbone)
    %   'backbone_mask'      : []   logical n x n (if provided, used directly)
    %   'p_values_mat'       : []   sparse/double n x n (PF p-values; requires backbone_regime.m)
    %   'backbone_q'         : 0.05 (BH start)
    %   'backbone_max_q'     : 0.25 (BH ceiling)
    %   'backbone_q_ladder'  : 2.0  (BH step multiplier)
    %   'alpha_fallback'     : []   (quantile fallback if BH yields none)
    %
    % Outputs
    %   train, test          : adjacency matrices (positives only; no negatives here)

    % ---- parse options ----
    p = inputParser;
    addParameter(p, 'use_backbone', false);
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
        fprintf('[DivideNet] BACKBONE mode ON. ratioTrain=%.3f, ratioBackbone=%.3f (subset of TRAIN)\n', ratioTrain, ratioBackbone);
    end

    % ===== BACKBONE MODE (subset of TRAIN) =====
    if opt.use_backbone
        % 1) Build/obtain backbone mask B
        if ~isempty(opt.backbone_mask)
            B = logical(sparse(opt.backbone_mask));
        elseif ~isempty(opt.p_values_mat)
            assert(exist('backbone_regime','file')==2, 'backbone_regime.m must be on path to threshold p_values_mat.');
            [B, thr, st] = backbone_regime(net, opt.p_values_mat, ...
                               'q', opt.backbone_q, ...
                               'max_q', opt.backbone_max_q, ...
                               'q_ladder', opt.backbone_q_ladder, ...
                               'alpha_fallback', opt.alpha_fallback);
            fprintf('[Backbone] kept %d/%d edges (%.1f%%), method=%s, thr=%g\n', st.kept, st.m_edges, 100*st.fraction, st.method, st.thr);
        else
            error('Backbone mode requires either ''backbone_mask'' or ''p_values_mat''.');
        end
        B = B & (net > 0);  % ensure mask only where edges exist

        % 2) Counts and targets
        [i_all, j_all] = find(net);
        m  = numel(i_all);                             % total true links
        mB = nnz(B);                                   % total backbone links
        num_test  = ceil((1 - ratioTrain) * m);
        num_train = m - num_test;                      % exact TRAIN size

        % --- Enforce: BackboneRatio ≤ TrainRatio (as fractions of total links) ---
        % This guarantees that the "forced backbone" part cannot exceed the train split.
        ratioBackbone_eff = min(ratioBackbone, ratioTrain);
        if ratioBackbone_eff < ratioBackbone
            fprintf(['[DivideNet] Clamping BackboneRatio from %.3f to %.3f ', ...
                    'so backbone TRAIN fraction does not exceed TrainRatio=%.3f.\n'], ...
                    ratioBackbone, ratioBackbone_eff, ratioTrain);
        end

        % Desired number of backbone edges in TRAIN (w.r.t total links),
        % also never exceeding the TRAIN size itself.
        raw_target_bb = round(ratioBackbone_eff * m);
        target_bb     = min([mB, raw_target_bb, num_train]);

        % 3) Split candidates
        [ib, jb] = find(B);
        [inb, jnb] = find(net & ~B);
        mNB = numel(inb);                              % total non-backbone links

        % 4) Select backbone portion for TRAIN (up to target_bb)
        perm_b  = randperm(mB);
        sel_bb  = perm_b(1:min(target_bb, mB));        % if mB<target_bb, take all
        i_bb_tr = ib(sel_bb); j_bb_tr = jb(sel_bb);

        % 5) Fill the remaining TRAIN slots with non-backbone first
        remaining_needed = num_train - numel(sel_bb);
        if remaining_needed < 0
            % Rare case: target_bb > num_train (e.g., huge ratioBackbone with small ratioTrain)
            % Trim backbone selection down to num_train
            sel_bb  = sel_bb(1:num_train);
            i_bb_tr = ib(sel_bb); j_bb_tr = jb(sel_bb);
            remaining_needed = 0;
        end

        i_fill = []; j_fill = [];
        if remaining_needed > 0
            take_nb = min(remaining_needed, mNB);
            if take_nb > 0
                perm_nb = randperm(mNB);
                sel_nb  = perm_nb(1:take_nb);
                i_fill  = inb(sel_nb); j_fill = jnb(sel_nb);
            end

            remaining_needed = num_train - (numel(sel_bb) + numel(i_fill));
            if remaining_needed > 0
                % Not enough non-backbone to reach num_train → gracefully top up with extra backbone
                % (beyond target_bb) to satisfy TRAIN size.
                leftover_bb_idx = setdiff(1:mB, sel_bb, 'stable');
                if ~isempty(leftover_bb_idx)
                    add_bb = min(remaining_needed, numel(leftover_bb_idx));
                    extra  = leftover_bb_idx(randperm(numel(leftover_bb_idx), add_bb));
                    i_bb_tr = [i_bb_tr; ib(extra)];
                    j_bb_tr = [j_bb_tr; jb(extra)];
                    remaining_needed = num_train - (numel(i_bb_tr) + numel(i_fill));
                    if remaining_needed > 0
                        warning('[DivideNet] Even after topping up with backbone, TRAIN short by %d edges (check ratios).', remaining_needed);
                    end
                else
                    warning('[DivideNet] No leftover backbone edges to top up; TRAIN short by %d edges.', remaining_needed);
                end
            end
        end

        % 6) Assemble TRAIN and TEST
        train = sparse([i_bb_tr; i_fill], [j_bb_tr; j_fill], 1, n, n);

        % TEST = all edges not chosen into TRAIN
        test = net - train;  % safe since train ⊆ net

        % 7) Reporting
        n_train = nnz(train);
        n_test  = nnz(test);
        n_bb_in_train = nnz(train & B);
        n_nb_in_train = n_train - n_bb_in_train;

        fprintf('[DivideNet] TRAIN total: %d (requested %d)\n', n_train, num_train);
        fprintf('[DivideNet]  ├─ backbone in TRAIN: %d (target %d of %d)\n', n_bb_in_train, target_bb, mB);
        fprintf('[DivideNet]  └─ non-backbone in TRAIN: %d of %d\n', n_nb_in_train, mNB);
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
