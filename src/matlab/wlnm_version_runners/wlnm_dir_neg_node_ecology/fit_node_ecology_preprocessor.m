function state = fit_node_ecology_preprocessor(mass, train_roles, canonical_node_ids)
%FIT_NODE_ECOLOGY_PREPROCESSOR Fit mass scaling on canonical train nodes.
% Mass is transformed with ln, missing values are median-imputed, and mean
% plus population standard deviation are calculated after imputation.

    roles = string(train_roles(:));
    n = numel(roles);
    role_categories = ["resource", "consumer-resource", "consumer", "isolate"];
    if any(~ismember(roles, role_categories))
        error('WLNMDirNegNodeEcology:UnknownRole', ...
            'Training roles contain values outside the frozen role vocabulary.');
    end

    ids = double(canonical_node_ids(:));
    ids = unique(ids(ids > 0));
    if isempty(ids) || any(ids ~= floor(ids)) || any(ids > n)
        error('WLNMDirNegNodeEcology:TrainingNodeIDs', ...
            'Canonical training IDs must identify at least one valid node.');
    end

    values = NaN(n,1);
    if isnumeric(mass) && ~isempty(mass)
        raw = double(mass(:));
        upto = min(n,numel(raw));
        values(1:upto) = raw(1:upto);
    end
    valid = isfinite(values) & values > 0;
    log_mass = NaN(n,1);
    log_mass(valid) = log(values(valid));

    train_values = log_mass(ids);
    finite_train = isfinite(train_values);
    if any(finite_train)
        mass_median = median(train_values(finite_train));
    else
        mass_median = 0;
    end
    filled = train_values;
    filled(~finite_train) = mass_median;
    mass_center = mean(filled);
    mass_scale = std(filled,1);
    if ~isfinite(mass_center)
        mass_center = 0;
    end
    if ~isfinite(mass_scale) || mass_scale <= 1e-12
        mass_scale = 1;
    end

    state = struct( ...
        'training_node_ids', ids(:)', ...
        'mass_median', double(mass_median), ...
        'mass_center', double(mass_center), ...
        'mass_scale', double(mass_scale), ...
        'role_categories', role_categories, ...
        'feature_dim_per_node', 7, ...
        'protocol', 'ln_train_subgraph_median_mean_std_v1', ...
        'role_protocol', 'train_adjacency_fixed_four_roles_v1' ...
    );
end
