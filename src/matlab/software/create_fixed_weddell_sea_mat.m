function output_path = create_fixed_weddell_sea_mat()
%CREATE_FIXED_WEDDELL_SEA_MAT Create the agreed Weddell Sea analysis input.
% Removes self-links and the reciprocal Amphidinium-Gyrodinium links so
% both algae are basal, while retaining every species and all other links.

    matlab_dir = fileparts(fileparts(mfilename('fullpath')));
    data_dir = fullfile(matlab_dir, 'data', 'foodwebs_mat');
    input_path = fullfile(data_dir, 'Weddell Sea_tax_mass.mat');
    output_path = fullfile(data_dir, 'Weddell Sea_tax_mass_fixed.mat');

    S = load(input_path);
    required = {'net', 'taxonomy', 'mass', 'role'};
    assert(all(isfield(S, required)), 'Weddell Sea input is missing required variables.');

    names = strtrim(string(S.taxonomy(:)));
    amphidinium_idx = find(names == "Amphidinium hadai");
    gyrodinium_idx = find(names == "Gyrodinium lachryama");
    assert(isscalar(amphidinium_idx) && isscalar(gyrodinium_idx), ...
        'Expected each target taxon exactly once.');

    net = spones(sparse(S.net));
    original_links = nnz(net);
    original_self_links = nnz(diag(net));

    assert(net(amphidinium_idx, gyrodinium_idx) == 1, ...
        'Missing Amphidinium -> Gyrodinium link.');
    assert(net(gyrodinium_idx, amphidinium_idx) == 1, ...
        'Missing Gyrodinium -> Amphidinium link.');

    net = net - spdiags(diag(net), 0, size(net, 1), size(net, 2));
    net(amphidinium_idx, gyrodinium_idx) = 0;
    net(gyrodinium_idx, amphidinium_idx) = 0;
    net = spones(net);

    in_degree = full(sum(net, 1))';
    out_degree = full(sum(net, 2));
    role = repmat({'consumer-resource'}, size(net, 1), 1);
    role(in_degree == 0 & out_degree > 0) = {'resource'};
    role(in_degree > 0 & out_degree == 0) = {'consumer'};
    role(in_degree == 0 & out_degree == 0) = {'isolate'};

    assert(in_degree(amphidinium_idx) == 0 && in_degree(gyrodinium_idx) == 0, ...
        'The corrected algae are not basal.');

    S.net = net;
    S.role = role;
    save(output_path, '-struct', 'S');

    fprintf('Created: %s\n', output_path);
    fprintf('Species: %d\n', size(net, 1));
    fprintf('Links: %d -> %d (%d self-links and 2 reciprocal links removed)\n', ...
        original_links, nnz(net), original_self_links);
end
