function features = transform_node_ecology_features(mass, train_roles, state)
%TRANSFORM_NODE_ECOLOGY_FEATURES Build the frozen seven-channel node block.
% Order: [mass_z, mass_missing, resource, consumer-resource, consumer,
% isolate, real_node].

    roles = string(train_roles(:));
    n = numel(roles);
    categories = string(state.role_categories(:))';
    expected = ["resource", "consumer-resource", "consumer", "isolate"];
    if ~isequal(categories, expected)
        error('WLNMDirNegNodeEcology:RoleVocabulary', ...
            'The role vocabulary does not match the frozen channel order.');
    end
    if any(~ismember(roles, expected))
        error('WLNMDirNegNodeEcology:UnknownRole', ...
            'Node roles contain values outside the frozen vocabulary.');
    end

    values = NaN(n,1);
    if isnumeric(mass) && ~isempty(mass)
        raw = double(mass(:));
        upto = min(n,numel(raw));
        values(1:upto) = raw(1:upto);
    end
    valid = isfinite(values) & values > 0;
    missing = ~valid;
    log_mass = NaN(n,1);
    log_mass(valid) = log(values(valid));
    log_mass(missing) = state.mass_median;
    mass_z = (log_mass - state.mass_center) ./ state.mass_scale;

    onehot = zeros(n,numel(expected));
    for category = 1:numel(expected)
        onehot(:,category) = double(roles == expected(category));
    end
    features = [mass_z, double(missing), onehot, ones(n,1)];
    if size(features,2) ~= 7 || any(~isfinite(features(:)))
        error('WLNMDirNegNodeEcology:InvalidNodeFeatures', ...
            'Node feature construction produced invalid values or dimensions.');
    end
end
