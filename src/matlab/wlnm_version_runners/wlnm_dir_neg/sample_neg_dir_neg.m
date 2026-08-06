function [train_pos, train_neg, test_pos, test_neg, diagnostics] = sample_neg_dir_neg( ...
        train, test, role, a, portion, evaluate_on_all_unseen, use_role_filter, ...
        mass, use_mass_constraint, mass_constraint_threshold, topup_policy, sampling_strategy)
    %SAMPLE_NEG_DIR_NEG Sample directed negative links for WLNM_dir_neg.
    %
    % The eligibility pool can be role-only, role-or-mass, mass-only, or all
    % non-links. Candidates are sampled uniformly without replacement. If
    % the selected eligibility pool is too small and top-up is enabled, all
    % eligible links are retained and the deficit is sampled uniformly from
    % the remaining non-links.
    % When evaluate_on_all_unseen is true we still enumerate the pool because
    % the caller explicitly asks for every unseen candidate in the test set.

    if nargin < 5 || isempty(portion), portion = 1; end
    if nargin < 6 || isempty(evaluate_on_all_unseen), evaluate_on_all_unseen = false; end
    if nargin < 7 || isempty(use_role_filter), use_role_filter = true; end
    if nargin < 8, mass = []; end
    if nargin < 9 || isempty(use_mass_constraint), use_mass_constraint = false; end
    if nargin < 10 || isempty(mass_constraint_threshold), mass_constraint_threshold = 1.0; end
    if nargin < 11 || isempty(topup_policy), topup_policy = 'uniform_remaining_nonlinks'; end
    if nargin < 12 || isempty(sampling_strategy), sampling_strategy = 'uniform_without_replacement'; end

    protocol = resolve_negative_sampling_protocol('', a, sampling_strategy, ...
        topup_policy, use_role_filter, use_mass_constraint);
    a = protocol.negative_positive_ratio;
    topup_policy = protocol.topup_policy;
    sampling_strategy = protocol.sampling_strategy;
    use_role_filter = protocol.use_role_filter;
    use_mass_constraint = protocol.use_mass_constraint;

    train = sparse(train);
    test = sparse(test);
    n = size(train, 1);
    mass = normalize_mass_vector(mass, n);
    mass_constraint_threshold = double(mass_constraint_threshold);
    if ~isfinite(mass_constraint_threshold) || mass_constraint_threshold <= 0
        warning('[sample_neg] Invalid mass constraint threshold %.4g. Falling back to 1.0.', ...
            mass_constraint_threshold);
        mass_constraint_threshold = 1.0;
    end
    valid_mass_nodes = isfinite(mass) & mass > 0;
    mass_constraint_active = logical(use_mass_constraint) && sum(valid_mass_nodes) >= 2;
    if logical(use_mass_constraint) && ~mass_constraint_active
        warning(['[sample_neg] Mass eligibility requested, but fewer than two positive finite ' ...
            'masses are available. Using role/top-up sampling only.']);
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

    [full_pool, role_mask, mass_mask, eligible_mask, eligibility_mode] = ...
        build_negative_candidate_pool(net, role_code, requested_role_filter, ...
            mass, mass_constraint_threshold, mass_constraint_active);

    pos_total = train_size + test_size;
    need_total_requested = floor(a * pos_total);
    role_pool_size = sum(role_mask);
    mass_pool_size = sum(mass_mask);
    eligible_pool_size = sum(eligible_mask);
    full_pool_size = size(full_pool, 1);
    eligibility_filter_active = requested_role_filter || mass_constraint_active;

    if ~eligibility_filter_active
        sampling_mode = 'all_nonlinks';
    elseif eligible_pool_size < need_total_requested
        sampling_mode = 'hybrid';
        if strcmp(topup_policy, 'error')
            error('sample_neg_dir_neg:EligiblePoolShortfall', ...
                ['Eligible negative pool (%s) has %d links but %d are required, ' ...
                'and negativeTopupPolicy=error.'], ...
                eligibility_mode, eligible_pool_size, need_total_requested);
        end
        warning(['[sample_neg] Eligible negative pool (%s) %d < need %d. ' ...
            'Using all eligible negatives, then uniform random non-link top-up.'], ...
            eligibility_mode, eligible_pool_size, need_total_requested);
    else
        sampling_mode = 'ecological';
    end

    need_total = min(need_total_requested, full_pool_size);
    full_pool_shortfall = max(0, need_total_requested - full_pool_size);

    if full_pool_size == 0 || need_total == 0
        warning('[sample_neg] No negatives available. Returning empties.');
        train_neg = zeros(0, 2);
        test_neg = zeros(0, 2);
        diagnostics = make_negative_sampling_diagnostics( ...
            eligibility_mode, a, sampling_strategy, topup_policy, ...
            role_pool_size, mass_pool_size, eligible_pool_size, full_pool_size, ...
            need_total_requested, 0, 0, max(0, need_total_requested), ...
            full_pool_shortfall, 0, 0, 0);
        return;
    end

    if evaluate_on_all_unseen
        neg_links = full_pool(eligible_mask, :);
        pool_size = size(neg_links, 1);

        k_train = min(floor(a * train_size), pool_size);
        idx_train = randperm(pool_size, k_train)';
        train_neg = neg_links(idx_train, :);

        mask = true(pool_size, 1);
        mask(idx_train) = false;
        test_neg = neg_links(mask, :);
        mass_counts = summarize_negative_link_counts( ...
            neg_links, role_code, mass, mass_constraint_threshold, mass_constraint_active);
        eligible_neg_count = size(neg_links, 1);
        eligible_shortfall = 0;
        random_topup_count = 0;
        mass_preferred_count = mass_counts.role_mass + mass_counts.nonrole_mass;
        role_mass_preferred_count = mass_counts.role_mass;
        role_other_count = mass_counts.role_other;
        nonrole_mass_preferred_count = mass_counts.nonrole_mass;
        nonrole_other_count = mass_counts.nonrole_other;
    else
        [k_train, k_test] = split_negative_counts( ...
            need_total, train_size, test_size, floor(a * train_size), floor(a * test_size));

        [neg_links, mass_counts, eligible_shortfall, eligible_neg_count, random_topup_count] = ...
            sample_negative_links_random_eligible_pool( ...
                full_pool, eligible_mask, role_code, mass, mass_constraint_threshold, ...
                mass_constraint_active, k_train + k_test, topup_policy);

        mass_preferred_count = mass_counts.role_mass + mass_counts.nonrole_mass;
        role_mass_preferred_count = mass_counts.role_mass;
        role_other_count = mass_counts.role_other;
        nonrole_mass_preferred_count = mass_counts.nonrole_mass;
        nonrole_other_count = mass_counts.nonrole_other;

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

    selected_negative_count = size(train_neg, 1) + size(test_neg, 1);
    diagnostics = make_negative_sampling_diagnostics( ...
        eligibility_mode, a, sampling_strategy, topup_policy, ...
        role_pool_size, mass_pool_size, eligible_pool_size, full_pool_size, ...
        need_total_requested, selected_negative_count, eligible_neg_count, ...
        eligible_shortfall, full_pool_shortfall, random_topup_count, ...
        size(train_neg, 1), size(test_neg, 1));

    % --- logging ---
    fprintf(['[NegPool] mode=%s strategy=%s legacy_strategy=random_eligible_pool topup_policy=%s ' ...
        'role_pool=%d full_pool=%d need_total=%d ' ...
        'requested_need=%d eligible_shortfall=%d full_pool_shortfall=%d a=%g ' ...
        'eval_all=%d role_filter=%d mass_filter=%d eligibility=%s mass_pool=%d eligible_pool=%d ' ...
        '| eligible_neg=%d random_topup=%d | k_train=%d k_test=%d\n'], ...
        sampling_mode, sampling_strategy, topup_policy, role_pool_size, full_pool_size, ...
        need_total, need_total_requested, ...
        eligible_shortfall, full_pool_shortfall, a, evaluate_on_all_unseen, requested_role_filter, ...
        mass_constraint_active, eligibility_mode, mass_pool_size, eligible_pool_size, ...
        eligible_neg_count, random_topup_count, size(train_neg,1), size(test_neg,1));
    fprintf(['[NegMassPref] enabled=%d active=%d eligibility_sampling=1 priority_sampling=0 ' ...
        'threshold=%.4g valid_mass_nodes=%d/%d ' ...
        '| selected_mass_pref=%d role_mass=%d role_other=%d nonrole_mass=%d nonrole_other=%d\n'], ...
        logical(use_mass_constraint), mass_constraint_active, mass_constraint_threshold, ...
        sum(valid_mass_nodes), n, mass_preferred_count, role_mass_preferred_count, ...
        role_other_count, nonrole_mass_preferred_count, nonrole_other_count);

    fprintf(['[sample_neg] Final link counts (mode = %s, eligibility = %s, ' ...
        'use_role_filter = %d, use_mass_constraint = %d):\n'], ...
        sampling_mode, eligibility_mode, requested_role_filter, mass_constraint_active);
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

function [full_pool, role_mask, mass_mask, eligible_mask, eligibility_mode] = ...
        build_negative_candidate_pool( ...
            net, role_code, use_role_filter, mass, mass_constraint_threshold, ...
            mass_constraint_active)

    full_pool = enumerate_negative_links(net, role_code, false);
    role_mask = false(size(full_pool, 1), 1);
    mass_mask = false(size(full_pool, 1), 1);

    if use_role_filter && ~isempty(full_pool)
        role_mask = is_valid_role_pair(full_pool(:,1), full_pool(:,2), role_code);
    end

    if mass_constraint_active && ~isempty(full_pool)
        mass_mask = is_mass_preferred_pair( ...
            full_pool, mass, mass_constraint_threshold);
    end

    if use_role_filter && mass_constraint_active
        eligible_mask = role_mask | mass_mask;
        eligibility_mode = 'role_or_mass';
    elseif use_role_filter
        eligible_mask = role_mask;
        eligibility_mode = 'role_only';
    elseif mass_constraint_active
        eligible_mask = mass_mask;
        eligibility_mode = 'mass_only';
    else
        eligible_mask = true(size(full_pool, 1), 1);
        eligibility_mode = 'all_nonlinks';
    end
end

function [neg_links, counts, eligible_shortfall, eligible_selected_count, random_topup_count] = ...
        sample_negative_links_random_eligible_pool( ...
            full_pool, eligible_mask, role_code, mass, mass_constraint_threshold, ...
            mass_constraint_active, k, topup_policy)

    counts = empty_mass_preference_counts();
    eligible_shortfall = 0;
    eligible_selected_count = 0;
    random_topup_count = 0;

    if k <= 0
        neg_links = zeros(0, 2);
        return;
    end

    if size(full_pool, 1) < k
        error('sample_neg_dir_neg:InsufficientNegativePool', ...
            'Full negative pool has %d links but %d are required.', size(full_pool, 1), k);
    end

    eligible_indices = find(eligible_mask);
    if numel(eligible_indices) >= k
        idx = eligible_indices(randperm(numel(eligible_indices), k));
        neg_links = full_pool(idx, :);
        eligible_selected_count = k;
    else
        eligible_selected_count = numel(eligible_indices);
        eligible_shortfall = k - eligible_selected_count;
        if strcmp(topup_policy, 'error')
            error('sample_neg_dir_neg:EligiblePoolShortfall', ...
                ['Eligible negative pool has %d links but %d are required, ' ...
                'and negativeTopupPolicy=error.'], eligible_selected_count, k);
        end
        fallback_indices = find(~eligible_mask);

        if numel(fallback_indices) < eligible_shortfall
            error('sample_neg_dir_neg:InsufficientTopupPool', ...
                'Relaxed top-up pool has %d links but %d are required.', ...
                numel(fallback_indices), eligible_shortfall);
        end

        topup_idx = fallback_indices(randperm(numel(fallback_indices), eligible_shortfall));
        neg_links = [full_pool(eligible_indices, :); full_pool(topup_idx, :)];
        random_topup_count = eligible_shortfall;
    end

    if ~isempty(neg_links)
        neg_links = neg_links(randperm(size(neg_links, 1)), :);
    end

    counts = summarize_negative_link_counts( ...
        neg_links, role_code, mass, mass_constraint_threshold, mass_constraint_active);
end

function diagnostics = make_negative_sampling_diagnostics( ...
        eligibility_mode, ratio, sampling_strategy, topup_policy, ...
        role_pool_size, mass_pool_size, eligible_pool_size, full_pool_size, ...
        requested_count, selected_count, eligible_selected_count, ...
        eligible_shortfall, full_pool_shortfall, random_topup_count, ...
        train_negative_count, test_negative_count)

    diagnostics = struct();
    diagnostics.EligibilityMode = char(string(eligibility_mode));
    diagnostics.NegativePositiveRatio = double(ratio);
    diagnostics.SamplingStrategy = char(string(sampling_strategy));
    diagnostics.TopupPolicy = char(string(topup_policy));
    diagnostics.RolePoolSize = double(role_pool_size);
    diagnostics.MassPoolSize = double(mass_pool_size);
    diagnostics.EligiblePoolSize = double(eligible_pool_size);
    diagnostics.FullNegativePoolSize = double(full_pool_size);
    diagnostics.RequestedNegativeCount = double(requested_count);
    diagnostics.SelectedNegativeCount = double(selected_count);
    diagnostics.EligibleNegativeCount = double(eligible_selected_count);
    diagnostics.EligibleShortfall = double(eligible_shortfall);
    diagnostics.FullPoolShortfall = double(full_pool_shortfall);
    diagnostics.RandomTopupCount = double(random_topup_count);
    diagnostics.TopupProportion = double(random_topup_count) / max(1, double(selected_count));
    diagnostics.TrainNegativeCount = double(train_negative_count);
    diagnostics.TestNegativeCount = double(test_negative_count);
end

function counts = summarize_negative_link_counts( ...
        links, role_code, mass, mass_constraint_threshold, mass_constraint_active)

    counts = empty_mass_preference_counts();
    if isempty(links)
        return;
    end

    is_role = is_valid_role_pair(links(:,1), links(:,2), role_code);
    if mass_constraint_active
        is_mass = is_mass_preferred_pair(links, mass, mass_constraint_threshold);
    else
        is_mass = false(size(links, 1), 1);
    end

    counts.role_mass = sum(is_role & is_mass);
    counts.role_other = sum(is_role & ~is_mass);
    counts.nonrole_mass = sum(~is_role & is_mass);
    counts.nonrole_other = sum(~is_role & ~is_mass);
end

function counts = empty_mass_preference_counts()
    counts = struct( ...
        'role_mass', 0, ...
        'role_other', 0, ...
        'nonrole_mass', 0, ...
        'nonrole_other', 0 ...
    );
end

function tf = is_mass_preferred_pair(links, mass, mass_constraint_threshold)
    if isempty(links)
        tf = false(0, 1);
        return;
    end

    src = links(:,1);
    tgt = links(:,2);
    valid = isfinite(mass(src)) & isfinite(mass(tgt)) & mass(src) > 0 & mass(tgt) > 0;
    tf = valid & mass(tgt) < mass_constraint_threshold .* mass(src);
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
