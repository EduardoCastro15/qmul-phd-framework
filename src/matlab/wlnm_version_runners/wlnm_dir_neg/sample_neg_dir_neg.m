function [train_pos, train_neg, test_pos, test_neg] = sample_neg_dir_neg( ...
        train, test, role, a, portion, evaluate_on_all_unseen, use_role_filter, ...
        mass, use_mass_preference, mass_preference_threshold)
    %SAMPLE_NEG_DIR_NEG Sample directed negative links for WLNM_dir_neg.
    %
    % The common path samples only the requested number of negatives by
    % rejection, avoiding materializing the full n-by-n complement. If the
    % role-constrained pool is too small, constrained links are used first and
    % the remaining negatives are sampled from random directed non-links.
    % When evaluate_on_all_unseen is true we still enumerate the pool because
    % the caller explicitly asks for every unseen candidate in the test set.

    if nargin < 5 || isempty(portion), portion = 1; end
    if nargin < 6 || isempty(evaluate_on_all_unseen), evaluate_on_all_unseen = false; end
    if nargin < 7 || isempty(use_role_filter), use_role_filter = true; end
    if nargin < 8, mass = []; end
    if nargin < 9 || isempty(use_mass_preference), use_mass_preference = false; end
    if nargin < 10 || isempty(mass_preference_threshold), mass_preference_threshold = 1.0; end

    train = sparse(train);
    test = sparse(test);
    n = size(train, 1);
    mass = normalize_mass_vector(mass, n);
    mass_preference_threshold = double(mass_preference_threshold);
    if ~isfinite(mass_preference_threshold) || mass_preference_threshold <= 0
        warning('[sample_neg] Invalid mass preference threshold %.4g. Falling back to 1.0.', ...
            mass_preference_threshold);
        mass_preference_threshold = 1.0;
    end
    mass_preference_active = logical(use_mass_preference) && any(isfinite(mass) & mass > 0);
    if logical(use_mass_preference) && ~mass_preference_active
        warning('[sample_neg] Mass preference requested, but no positive finite masses are available. Using role/top-up sampling only.');
    end

    % === positives ===
    [i, j] = find(train);
    train_pos = [i, j];
    train_size = size(train_pos, 1);

    [i, j] = find(test);
    test_pos = [i, j];
    test_size = size(test_pos, 1);

    % === Build full positive set ===
    if nnz(train & test) ~= 0
        error('sample_neg_dir_neg:Overlap', 'Train and test must not overlap');
    end
    net = spones(train + test);
    net = net - spdiags(diag(net), 0, n, n);
    net = spones(net);

    role_code = encode_roles(role, n);
    requested_role_filter = logical(use_role_filter);
    effective_role_filter = requested_role_filter;

    pos_total = train_size + test_size;
    need_total_requested = floor(a * pos_total);
    constrained_pool_size = negative_pool_size(net, role_code, true);
    full_pool_size = negative_pool_size(net, role_code, false);

    if requested_role_filter
        if constrained_pool_size < need_total_requested
            sampling_mode = 'hybrid';
            pool_size = full_pool_size;
            warning(['[sample_neg] Role-constrained pool %d < need %d. ' ...
                'Using constrained negatives first, then random non-link top-up.'], ...
                constrained_pool_size, need_total_requested);
        else
            sampling_mode = 'role';
            pool_size = constrained_pool_size;
        end
    else
        sampling_mode = 'all_nonlinks';
        pool_size = full_pool_size;
    end

    need_total = min(need_total_requested, pool_size);
    constrained_neg_count = 0;
    random_topup_count = 0;
    mass_preferred_count = 0;
    role_mass_preferred_count = 0;
    role_other_count = 0;
    nonrole_mass_preferred_count = 0;
    nonrole_other_count = 0;

    if pool_size == 0 || need_total == 0
        warning('[sample_neg] No negatives available. Returning empties.');
        train_neg = zeros(0, 2);
        test_neg = zeros(0, 2);
        return;
    end

    if evaluate_on_all_unseen
        neg_links = enumerate_negative_links(net, role_code, effective_role_filter);
        pool_size = size(neg_links, 1);

        k_train = min(floor(a * train_size), pool_size);
        idx_train = select_indices_by_mass_preference( ...
            neg_links, mass, mass_preference_threshold, k_train, mass_preference_active);
        train_neg = neg_links(idx_train, :);

        mask = true(pool_size, 1);
        mask(idx_train) = false;
        test_neg = neg_links(mask, :);
        mass_preferred_count = sum(is_mass_preferred_pair(train_neg, mass, mass_preference_threshold));
    else
        [k_train, k_test] = split_negative_counts( ...
            need_total, train_size, test_size, floor(a * train_size), floor(a * test_size));

        if mass_preference_active
            [neg_links, mass_counts] = sample_negative_links_with_mass_preference( ...
                net, role_code, requested_role_filter, mass, mass_preference_threshold, k_train + k_test);
            constrained_neg_count = mass_counts.role_mass + mass_counts.role_other;
            random_topup_count = mass_counts.nonrole_mass + mass_counts.nonrole_other;
            mass_preferred_count = mass_counts.role_mass + mass_counts.nonrole_mass;
            role_mass_preferred_count = mass_counts.role_mass;
            role_other_count = mass_counts.role_other;
            nonrole_mass_preferred_count = mass_counts.nonrole_mass;
            nonrole_other_count = mass_counts.nonrole_other;

            if ~isempty(neg_links)
                neg_links = neg_links(randperm(size(neg_links, 1)), :);
            end
        else
            [neg_links, constrained_neg_count, random_topup_count] = sample_negative_links_with_topup( ...
                net, role_code, requested_role_filter, k_train + k_test, ...
                constrained_pool_size, full_pool_size);
        end
        train_neg = neg_links(1:k_train, :);
        test_neg = neg_links(k_train+1:end, :);
    end

    % === Apply portion filtering (if needed) ===
    if portion < 1
        train_pos = train_pos(1:min(size(train_pos,1), ceil(portion * size(train_pos, 1))), :);
        train_neg = train_neg(1:min(size(train_neg,1), ceil(portion * size(train_neg, 1))), :);
        test_pos  = test_pos(1:min(size(test_pos,1),   ceil(portion * size(test_pos, 1))), :);
        test_neg  = test_neg(1:min(size(test_neg,1),   ceil(portion * size(test_neg, 1))), :);
    elseif portion > 1
        train_pos = train_pos(1:min(size(train_pos,1), portion), :);
        train_neg = train_neg(1:min(size(train_neg,1), portion), :);
        test_pos  = test_pos(1:min(size(test_pos,1),  portion), :);
        test_neg  = test_neg(1:min(size(test_neg,1),  portion), :);
    end

    % --- logging ---
    fprintf(['[NegPool] mode=%s role_pool=%d full_pool=%d need_total=%d a=%d ' ...
        'eval_all=%d role_filter=%d | constrained_neg=%d random_topup=%d | k_train=%d k_test=%d\n'], ...
        sampling_mode, constrained_pool_size, full_pool_size, need_total, a, ...
        evaluate_on_all_unseen, effective_role_filter, constrained_neg_count, random_topup_count, ...
        size(train_neg,1), size(test_neg,1));
    fprintf(['[NegMassPref] enabled=%d active=%d threshold=%.4g valid_mass_nodes=%d/%d ' ...
        '| selected_mass_pref=%d role_mass=%d role_other=%d nonrole_mass=%d nonrole_other=%d\n'], ...
        logical(use_mass_preference), mass_preference_active, mass_preference_threshold, ...
        sum(isfinite(mass) & mass > 0), n, mass_preferred_count, role_mass_preferred_count, ...
        role_other_count, nonrole_mass_preferred_count, nonrole_other_count);

    fprintf('[sample_neg] Final link counts (mode = %s, use_role_filter = %d):\n', ...
        sampling_mode, effective_role_filter);
    fprintf('    Train Positive: %d\n', size(train_pos, 1));
    fprintf('    Train Negative: %d\n', size(train_neg, 1));
    fprintf('    Test  Positive: %d\n', size(test_pos, 1));
    fprintf('    Test  Negative: %d\n', size(test_neg, 1));
end

function role_code = encode_roles(role, n)
    role_code = zeros(n, 1);
    if isempty(role)
        return;
    end

    role_str = lower(string(role(:)));
    upto = min(n, numel(role_str));
    role_code(1:upto) = double(role_str(1:upto) == "consumer") + ...
        2 * double(role_str(1:upto) == "resource") + ...
        3 * double(role_str(1:upto) == "consumer-resource");
end

function pool_size = negative_pool_size(net, role_code, use_role_filter)
    n = size(net, 1);

    if use_role_filter
        valid_codes = role_code(role_code >= 1 & role_code <= 3);
        role_counts = accumarray(valid_codes(:), 1, [3, 1], @sum, 0);
        pairs = allowed_role_pairs();
        total_candidates = 0;

        for p = 1:size(pairs, 1)
            src_code = pairs(p, 1);
            tgt_code = pairs(p, 2);

            if src_code == tgt_code
                total_candidates = total_candidates + ...
                    role_counts(src_code) * max(0, role_counts(tgt_code) - 1);
            else
                total_candidates = total_candidates + ...
                    role_counts(src_code) * role_counts(tgt_code);
            end
        end

        [pi, pj] = find(net);
        positive_is_candidate = is_valid_role_pair(pi, pj, role_code);
        pool_size = total_candidates - sum(positive_is_candidate);
    else
        pool_size = n * max(0, n - 1) - nnz(net);
    end

    pool_size = max(0, double(pool_size));
end

function tf = is_valid_role_pair(i, j, role_code)
    src = role_code(i);
    tgt = role_code(j);
    pairs = allowed_role_pairs();
    tf = false(size(src));

    for p = 1:size(pairs, 1)
        tf = tf | (src == pairs(p, 1) & tgt == pairs(p, 2));
    end
end

function pairs = allowed_role_pairs()
    % Role codes: consumer=1, resource=2, consumer-resource=3.
    pairs = [
        1 1  % consumer -> consumer
        2 2  % resource -> resource
        1 2  % consumer -> resource
        1 3  % consumer -> consumer-resource
        3 2  % consumer-resource -> resource
    ];
end

function [k_train, k_test] = split_negative_counts(need_total, train_size, test_size, k_train_target, k_test_target)
    ratio = train_size / max(1, train_size + test_size);

    k_train = min(k_train_target, floor(need_total * ratio));
    k_test = min(k_test_target, need_total - k_train);

    leftover = need_total - (k_train + k_test);
    if leftover > 0
        add_train = min(leftover, max(0, k_train_target - k_train));
        k_train = k_train + add_train;
        k_test = need_total - k_train;
    end

    if train_size > 0 && test_size > 0 && need_total >= 2
        if k_train == 0 && k_test > 1
            k_train = 1;
            k_test = need_total - 1;
        elseif k_test == 0 && k_train > 1
            k_test = 1;
            k_train = need_total - 1;
        end
    end
end

function [neg_links, counts] = sample_negative_links_with_mass_preference( ...
        net, role_code, use_role_filter, mass, mass_preference_threshold, k)

    counts = empty_mass_preference_counts();
    if k <= 0
        neg_links = zeros(0, 2);
        return;
    end

    pool = enumerate_negative_links(net, role_code, false);
    if size(pool, 1) < k
        error('sample_neg_dir_neg:InsufficientNegativePool', ...
            'Negative pool has %d links but %d are required.', size(pool, 1), k);
    end

    is_mass = is_mass_preferred_pair(pool, mass, mass_preference_threshold);
    if use_role_filter
        is_role = is_valid_role_pair(pool(:,1), pool(:,2), role_code);
        priority_groups = { ...
            find(is_role & is_mass), ...
            find(is_role & ~is_mass), ...
            find(~is_role & is_mass), ...
            find(~is_role & ~is_mass) ...
        };
    else
        is_role = false(size(pool, 1), 1);
        priority_groups = {find(is_mass), find(~is_mass)};
    end

    selected = zeros(0, 1);
    needed = k;
    for g = 1:numel(priority_groups)
        [take_idx, needed] = take_random_candidates(priority_groups{g}, needed);
        selected = [selected; take_idx]; %#ok<AGROW>
        if needed <= 0
            break;
        end
    end

    if numel(selected) < k
        error('sample_neg_dir_neg:InsufficientPreferredNegativePool', ...
            'Priority pools selected %d links but %d are required.', numel(selected), k);
    end

    selected = selected(1:k);
    neg_links = pool(selected, :);

    selected_role = is_role(selected);
    selected_mass = is_mass(selected);
    counts.role_mass = sum(selected_role & selected_mass);
    counts.role_other = sum(selected_role & ~selected_mass);
    counts.nonrole_mass = sum(~selected_role & selected_mass);
    counts.nonrole_other = sum(~selected_role & ~selected_mass);
end

function counts = empty_mass_preference_counts()
    counts = struct( ...
        'role_mass', 0, ...
        'role_other', 0, ...
        'nonrole_mass', 0, ...
        'nonrole_other', 0 ...
    );
end

function [selected, remaining_needed] = take_random_candidates(candidates, needed)
    candidates = candidates(:);
    if needed <= 0 || isempty(candidates)
        selected = zeros(0, 1);
        remaining_needed = needed;
        return;
    end

    n_take = min(needed, numel(candidates));
    selected = candidates(randperm(numel(candidates), n_take));
    remaining_needed = needed - n_take;
end

function idx = select_indices_by_mass_preference(links, mass, mass_preference_threshold, k, mass_preference_active)
    n = size(links, 1);
    k = min(k, n);
    if k <= 0
        idx = zeros(0, 1);
        return;
    end

    if ~mass_preference_active
        idx = randperm(n, k)';
        return;
    end

    is_mass = is_mass_preferred_pair(links, mass, mass_preference_threshold);
    preferred = find(is_mass);
    fallback = find(~is_mass);

    [idx_pref, remaining] = take_random_candidates(preferred, k);
    [idx_fallback, ~] = take_random_candidates(fallback, remaining);
    idx = [idx_pref; idx_fallback];
end

function tf = is_mass_preferred_pair(links, mass, mass_preference_threshold)
    if isempty(links)
        tf = false(0, 1);
        return;
    end

    src = links(:,1);
    tgt = links(:,2);
    valid = isfinite(mass(src)) & isfinite(mass(tgt)) & mass(src) > 0 & mass(tgt) > 0;
    tf = valid & mass(tgt) < mass_preference_threshold .* mass(src);
end

function mass = normalize_mass_vector(mass, n)
    normalized = NaN(n, 1);
    if nargin < 1 || isempty(mass) || ~isnumeric(mass)
        mass = normalized;
        return;
    end

    values = double(mass(:));
    upto = min(n, numel(values));
    normalized(1:upto) = values(1:upto);
    mass = normalized;
end

function [neg_links, constrained_count, random_topup_count] = sample_negative_links_with_topup( ...
        net, role_code, use_role_filter, k, constrained_pool_size, full_pool_size)

    constrained_count = 0;
    random_topup_count = 0;

    if k <= 0
        neg_links = zeros(0, 2);
        return;
    end

    if ~use_role_filter
        neg_links = sample_negative_links(net, role_code, false, k, full_pool_size);
        random_topup_count = size(neg_links, 1);
        return;
    end

    if constrained_pool_size >= k
        neg_links = sample_negative_links(net, role_code, true, k, constrained_pool_size);
        constrained_count = size(neg_links, 1);
        return;
    end

    constrained_links = sample_negative_links(net, role_code, true, constrained_pool_size, constrained_pool_size);
    constrained_count = size(constrained_links, 1);

    topup_needed = k - constrained_count;
    topup_links = sample_random_topup_links(net, role_code, constrained_links, topup_needed, full_pool_size);
    random_topup_count = size(topup_links, 1);

    neg_links = [constrained_links; topup_links];
end

function topup_links = sample_random_topup_links(net, role_code, excluded_links, k, full_pool_size)
    if k <= 0
        topup_links = zeros(0, 2);
        return;
    end

    pool = enumerate_negative_links(net, role_code, false);

    if ~isempty(excluded_links)
        n = size(net, 1);
        pool_lin = sub2ind([n, n], pool(:,1), pool(:,2));
        excluded_lin = sub2ind([n, n], excluded_links(:,1), excluded_links(:,2));
        pool = pool(~ismember(pool_lin, excluded_lin), :);
    end

    if size(pool, 1) < k
        error('sample_neg_dir_neg:InsufficientTopupPool', ...
            'Random top-up pool has %d links but %d are required (full_pool=%d).', ...
            size(pool, 1), k, full_pool_size);
    end

    idx = randperm(size(pool, 1), k);
    topup_links = pool(idx, :);
end

function neg_links = sample_negative_links(net, role_code, use_role_filter, k, pool_size)
    if k <= 0
        neg_links = zeros(0, 2);
        return;
    end

    % When the request is a large fraction of the pool, enumerating once is
    % faster and more predictable than many rejection retries.
    if k > 0.25 * pool_size
        pool = enumerate_negative_links(net, role_code, use_role_filter);
        idx = randperm(size(pool, 1), k);
        neg_links = pool(idx, :);
        return;
    end

    n = size(net, 1);
    lin = zeros(0, 1);
    max_rounds = 25;

    for round_id = 1:max_rounds
        remaining = k - numel(lin);
        if remaining <= 0
            break;
        end

        batch_size = max(1000, ceil(remaining * 2.5));
        cand = draw_candidate_pairs(n, role_code, use_role_filter, batch_size);
        if isempty(cand)
            break;
        end

        cand = cand(cand(:,1) ~= cand(:,2), :);
        if isempty(cand)
            continue;
        end

        cand_lin = sub2ind([n, n], cand(:,1), cand(:,2));
        cand_lin = cand_lin(net(cand_lin) == 0);
        if isempty(cand_lin)
            continue;
        end

        lin = unique([lin; cand_lin(:)], 'stable');

        if round_id > 5 && numel(lin) < 0.5 * k
            break;
        end
    end

    if numel(lin) < k
        pool = enumerate_negative_links(net, role_code, use_role_filter);
        pool_lin = sub2ind([n, n], pool(:,1), pool(:,2));
        already = ismember(pool_lin, lin);
        pool = pool(~already, :);

        extra = pool(randperm(size(pool, 1), k - numel(lin)), :);
        [i, j] = ind2sub([n, n], lin);
        neg_links = [[i(:), j(:)]; extra];
    else
        lin = lin(1:k);
        [i, j] = ind2sub([n, n], lin);
        neg_links = [i(:), j(:)];
    end
end

function cand = draw_candidate_pairs(n, role_code, use_role_filter, batch_size)
    if ~use_role_filter
        cand = [randi(n, batch_size, 1), randi(n, batch_size, 1)];
        return;
    end

    pairs = allowed_role_pairs();
    pair_counts = zeros(size(pairs, 1), 1);

    for p = 1:size(pairs, 1)
        src_nodes = find(role_code == pairs(p, 1));
        tgt_nodes = find(role_code == pairs(p, 2));

        if pairs(p, 1) == pairs(p, 2)
            pair_counts(p) = numel(src_nodes) * max(0, numel(tgt_nodes) - 1);
        else
            pair_counts(p) = numel(src_nodes) * numel(tgt_nodes);
        end
    end

    total_pairs = sum(pair_counts);

    if total_pairs == 0
        cand = zeros(0, 2);
        return;
    end

    edges = cumsum(pair_counts) / total_pairs;
    draws = rand(batch_size, 1);
    cand = zeros(batch_size, 2);

    lower = 0;
    for p = 1:size(pairs, 1)
        if pair_counts(p) == 0
            continue;
        end

        selected = draws > lower & draws <= edges(p);
        count = sum(selected);
        if count > 0
            src_nodes = find(role_code == pairs(p, 1));
            tgt_nodes = find(role_code == pairs(p, 2));
            cand(selected, :) = random_pairs_between_groups(src_nodes, tgt_nodes, count, pairs(p, 1) == pairs(p, 2));
        end
        lower = edges(p);
    end
end

function pairs = random_pairs_between_groups(src_nodes, tgt_nodes, count, same_role)
    if count <= 0
        pairs = zeros(0, 2);
        return;
    end

    if ~same_role
        pairs = [src_nodes(randi(numel(src_nodes), count, 1)), ...
                 tgt_nodes(randi(numel(tgt_nodes), count, 1))];
        return;
    end

    m = numel(src_nodes);
    if m < 2
        pairs = zeros(0, 2);
        return;
    end

    src_idx = randi(m, count, 1);
    offset = randi(m - 1, count, 1);
    tgt_idx = src_idx + offset;
    tgt_idx(tgt_idx > m) = tgt_idx(tgt_idx > m) - m;

    pairs = [src_nodes(src_idx), src_nodes(tgt_idx)];
end

function neg_links = enumerate_negative_links(net, role_code, use_role_filter)
    n = size(net, 1);

    if use_role_filter
        pairs = allowed_role_pairs();
        chunks = cell(1, size(pairs, 1));

        for p = 1:size(pairs, 1)
            src_nodes = find(role_code == pairs(p, 1));
            tgt_nodes = find(role_code == pairs(p, 2));

            if isempty(src_nodes) || isempty(tgt_nodes)
                chunks{p} = zeros(0, 2);
                continue;
            end

            [src, tgt] = ndgrid(src_nodes, tgt_nodes);
            if pairs(p, 1) == pairs(p, 2)
                keep = src ~= tgt;
                src = src(keep);
                tgt = tgt(keep);
            end
            lin = sub2ind([n, n], src, tgt);
            keep = net(lin) == 0;
            src = src(keep);
            tgt = tgt(keep);
            chunks{p} = [src(:), tgt(:)];
        end

        neg_links = vertcat(chunks{:});
    else
        [src, tgt] = ndgrid(1:n, 1:n);
        keep = src ~= tgt;
        src = src(keep);
        tgt = tgt(keep);
        lin = sub2ind([n, n], src, tgt);
        keep = net(lin) == 0;
        neg_links = [src(keep), tgt(keep)];
    end
end
