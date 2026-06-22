function audit = fix_duplicate_taxa_in_foodweb_mats()
%FIX_DUPLICATE_TAXA_IN_FOODWEB_MATS Merge taxa duplicated by outer whitespace.
%
% This targeted data repair updates the three known affected food webs in
% data/foodwebs_mat. It unions their links, removes self-links, verifies
% duplicate masses, recalculates node roles, and updates the food-web index.

    matlab_dir = fileparts(fileparts(mfilename('fullpath')));
    data_dir = fullfile(matlab_dir, 'data', 'foodwebs_mat');
    index_file = fullfile(data_dir, 'foodweb_metrics_ecosystem.csv');

    foodwebs = [
        "Blackrock Stream_tax_mass"
        "Sutton Stream_tax_mass"
        "Gearagh_tax_mass"
    ];

    audit = table('Size', [numel(foodwebs), 8], ...
        'VariableTypes', ["string", repmat("double", 1, 7)], ...
        'VariableNames', { ...
            'FoodWeb', 'OriginalNodes', 'FixedNodes', 'OriginalLinks', ...
            'FixedLinks', 'DuplicateNodesMerged', 'SelfLinksRemoved', ...
            'MassConflicts'});

    for i = 1:numel(foodwebs)
        foodweb = foodwebs(i);
        mat_file = fullfile(data_dir, foodweb + ".mat");
        input = load(mat_file, 'net', 'taxonomy', 'mass', 'role');

        original_net = spones(sparse(input.net));
        original_nodes = size(original_net, 1);
        original_links = nnz(original_net);
        original_self_links = nnz(diag(original_net));

        normalized_taxonomy = strtrim(string(input.taxonomy(:)));
        if any(ismissing(normalized_taxonomy) | strlength(normalized_taxonomy) == 0)
            error('fix_duplicate_taxa:EmptyTaxonomy', ...
                'Empty taxonomy encountered in %s.', foodweb);
        end

        [fixed_taxonomy, ~, group] = unique(normalized_taxonomy, 'stable');
        fixed_nodes = numel(fixed_taxonomy);
        membership = sparse(1:original_nodes, group, 1, original_nodes, fixed_nodes);

        fixed_net = spones(membership' * original_net * membership);
        fixed_net = fixed_net - spdiags(diag(fixed_net), 0, fixed_nodes, fixed_nodes);
        fixed_net = spones(fixed_net);

        original_mass = double(input.mass(:));
        fixed_mass = NaN(fixed_nodes, 1);
        mass_conflicts = 0;
        for group_id = 1:fixed_nodes
            values = original_mass(group == group_id);
            reference = values(1);
            tolerance = max(1e-12, 1e-10 * max(1, abs(reference)));
            if any(abs(values - reference) > tolerance)
                mass_conflicts = mass_conflicts + 1;
            else
                fixed_mass(group_id) = reference;
            end
        end

        if mass_conflicts > 0
            error('fix_duplicate_taxa:MassConflict', ...
                '%s contains %d duplicate taxon group(s) with conflicting masses.', ...
                foodweb, mass_conflicts);
        end

        fixed_role = derive_roles(fixed_net);

        % Preserve the variable shapes expected by Main and the WLNM runners.
        net = fixed_net;
        taxonomy = cellstr(fixed_taxonomy(:).');
        mass = fixed_mass(:).';
        role = fixed_role;

        temp_file = [tempname(data_dir) '.mat'];
        save(temp_file, 'net', 'taxonomy', 'mass', 'role');
        movefile(temp_file, mat_file, 'f');

        audit.FoodWeb(i) = foodweb;
        audit.OriginalNodes(i) = original_nodes;
        audit.FixedNodes(i) = fixed_nodes;
        audit.OriginalLinks(i) = original_links;
        audit.FixedLinks(i) = nnz(fixed_net);
        audit.DuplicateNodesMerged(i) = original_nodes - fixed_nodes;
        audit.SelfLinksRemoved(i) = original_self_links;
        audit.MassConflicts(i) = mass_conflicts;
    end

    update_foodweb_index(index_file, audit);
    writetable(audit, fullfile(data_dir, 'duplicate_taxa_fix_audit.csv'));
end

function role = derive_roles(net)
    prey_count = full(sum(net, 1))';
    consumer_count = full(sum(net, 2));

    role = repmat({'isolate'}, size(net, 1), 1);
    role(prey_count == 0 & consumer_count > 0) = {'resource'};
    role(prey_count > 0 & consumer_count == 0) = {'consumer'};
    role(prey_count > 0 & consumer_count > 0) = {'consumer-resource'};
end

function update_foodweb_index(index_file, audit)
    raw = fileread(index_file);
    has_trailing_newline = ~isempty(raw) && raw(end) == newline;
    lines = splitlines(string(raw));

    for i = 1:height(audit)
        prefix = audit.FoodWeb(i) + ",";
        row = startsWith(lines, prefix);
        if nnz(row) ~= 1
            error('fix_duplicate_taxa:IndexRow', ...
                'Expected one index row for %s, found %d.', ...
                audit.FoodWeb(i), nnz(row));
        end

        nodes = audit.FixedNodes(i);
        links = audit.FixedLinks(i);
        fields = split(lines(row), ',');
        ecosystem_type = fields(end);
        connectance = links / max(nodes * (nodes - 1), 1);
        lines(row) = sprintf('%s,%d,%d,%.9f,%s', ...
            audit.FoodWeb(i), nodes, links, connectance, ecosystem_type);
    end

    temp_file = [tempname(fileparts(index_file)) '.csv'];
    output = strjoin(lines, newline);
    if has_trailing_newline && ~endsWith(output, newline)
        output = output + newline;
    end
    file_id = fopen(temp_file, 'w');
    if file_id < 0
        error('fix_duplicate_taxa:IndexWrite', ...
            'Unable to open temporary index file %s.', temp_file);
    end
    cleanup = onCleanup(@() fclose(file_id));
    fwrite(file_id, char(output), 'char');
    clear cleanup;
    movefile(temp_file, index_file, 'f');
end
