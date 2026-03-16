function compute_backbone_link_stats_all()
    %COMPUTE_BACKBONE_LINK_STATS_ALL
    %  Extract structural properties of PF backbone links for all food webs.
    %
    %  Inputs (configured in the local config struct below):
    %   - foodweb_metrics_ecosystem.csv : list of foodweb names + metadata
    %   - .mat files in matFolder       : each with net, p_values_mat, mass,
    %                                     taxonomy, role, params, etc.
    %
    %  Outputs (CSV in backboneStatsDir):
    %   1) backbone_link_stats_long.csv
    %        One row per backbone link with:
    %          Foodweb, Source, Target,
    %          Source_in, Source_out, Source_deg,
    %          Target_in, Target_out, Target_deg,
    %          Source_mass, Target_mass,
    %          Source_taxonomy, Target_taxonomy,
    %          Source_role, Target_role
    %
    %   2) backbone_link_summary.csv
    %        One row per foodweb with:
    %          Foodweb, EcosystemType (if available),
    %          N_nodes, N_edges_total, N_edges_backbone,
    %          Backbone_edge_fraction,
    %          N_backbone_nodes, Backbone_node_fraction,
    %          HighDeg_threshold_total,
    %          N_highdeg_total, N_highdeg_backbone,
    %          Prop_highdeg_total, Prop_highdeg_backbone,
    %          Mean_deg_all_endpoints, Median_deg_all_endpoints,
    %          Mean_deg_backbone_endpoints, Median_deg_backbone_endpoints
    %
    %   3) all_link_stats_long.csv
    %        One row per edge in the full net with:
    %          (same columns as backbone_link_stats_long.csv)
    %          plus:
    %          IsBackbone (logical 0/1)
    %
    %  NOTE:
    %    - Uses backbone_regime(net, p_values_mat, ...) with PF parameters
    %      configured below (same as in Main.m).
    %    - Assumes Statistics & Machine Learning Toolbox is available (for PRCTILE).

    %% ================== CONFIG ==================
    config = struct( ...
        'foodwebCSV',        'data/foodwebs_mat/foodweb_metrics_ecosystem.csv', ...
        'matFolder',         'data/foodwebs_mat_backbones/', ...  % where AEW01_tax_mass.mat etc live
        'backboneStatsDir',  'data/result/backbone_stats/', ...   % output folder
        'backbone_q',        0.05, ...
        'backbone_max_q',    0.25, ...
        'backbone_q_ladder', 2.0, ...
        'alpha_fallback',    [], ...
        'highDeg_top_frac',  0.10 ...   % top 10% of total degree = "high degree"
    );

    % Make sure output directory exists
    if ~exist(config.backboneStatsDir, 'dir')
        mkdir(config.backboneStatsDir);
    end

    % Add paths if needed (adapt to your repo layout)
    addpath(genpath('wlnm_version_runners'));
    addpath(genpath('software'));
    addpath(genpath('logging'));
    addpath(genpath('data'));

    %% ================== LOAD FOODWEB LIST ==================
    meta = readtable(config.foodwebCSV);
    if ~ismember('Foodweb', meta.Properties.VariableNames)
        error('Expected column "Foodweb" in %s', config.foodwebCSV);
    end
    foodweb_names = meta.Foodweb;
    n_foodwebs = numel(foodweb_names);
    fprintf('[INFO] Found %d foodwebs in %s\n', n_foodwebs, config.foodwebCSV);

    % Try to get EcosystemType (optional)
    hasEcosystem = ismember('EcosystemType', meta.Properties.VariableNames);

    %% ================== ACCUMULATORS ==================
    all_link_rows   = {};   % backbone-only links (per-link table)
    all_edge_rows   = {};   % ALL edges (full net) with IsBackbone flag
    summary_rows    = [];   % per-foodweb summary table

    %% ================== MAIN LOOP ==================
    for f_idx = 1:n_foodwebs
        dataname = foodweb_names{f_idx};
        mat_path = fullfile(config.matFolder, strcat(dataname, '.mat'));

        if ~isfile(mat_path)
            fprintf('[WARN] .mat file not found for "%s": %s\n', dataname, mat_path);
            continue;
        end

        fprintf('\n[INFO] Processing foodweb (%d/%d): %s\n', f_idx, n_foodwebs, dataname);

        S = load(mat_path, 'net', 'p_values_mat', 'mass', 'taxonomy', 'role');
        if ~isfield(S, 'net') || ~isfield(S, 'p_values_mat')
            fprintf('[WARN] Missing net or p_values_mat in %s. Skipping.\n', mat_path);
            continue;
        end

        net          = S.net;
        p_values_mat = S.p_values_mat;

        % Ensure adjacency is logical/sparse
        net = sparse(net ~= 0);

        n_nodes = size(net, 1);
        if size(net,1) ~= size(net,2)
            fprintf('[WARN] net is not square (%dx%d) in %s. Skipping.\n', size(net,1), size(net,2), dataname);
            continue;
        end

        % ---- Compute PF backbone mask B ----
        if isempty(p_values_mat)
            fprintf('[WARN] p_values_mat is empty for "%s". Skipping backbone.\n', dataname);
            continue;
        end

        [B, thr, st] = backbone_regime(net, p_values_mat, ...
            'q',              config.backbone_q, ...
            'max_q',          config.backbone_max_q, ...
            'q_ladder',       config.backbone_q_ladder, ...
            'alpha_fallback', config.alpha_fallback);

        B = sparse(B ~= 0);  % ensure logical mask

        % ---- Degrees (node-level) ----
        outdeg = full(sum(net, 2));      % n x 1
        indeg  = full(sum(net, 1))';     % n x 1
        totdeg = indeg + outdeg;         % n x 1

        % ---- Edge sets: all edges vs backbone edges ----
        [src_all, tgt_all] = find(net);
        [src_bb,  tgt_bb]  = find(B & net);

        n_edges_total    = numel(src_all);
        n_edges_backbone = numel(src_bb);

        % Nodes participating in backbone edges
        nodes_bb = unique([src_bb; tgt_bb]);
        n_nodes_backbone = numel(nodes_bb);

        % ---- High-degree threshold (on total degree) ----
        highDeg_threshold = prctile(totdeg, 100*(1 - config.highDeg_top_frac));  % e.g. 90th percentile
        is_high = totdeg >= highDeg_threshold;

        n_high_total    = nnz(is_high);
        n_high_backbone = nnz(is_high(nodes_bb));

        % ---- Degree distributions for endpoints ----
        deg_all_endpoints = totdeg([src_all; tgt_all]);
        deg_bb_endpoints  = totdeg([src_bb; tgt_bb]);

        mean_deg_all     = mean(deg_all_endpoints);
        median_deg_all   = median(deg_all_endpoints);
        mean_deg_bb      = mean(deg_bb_endpoints);
        median_deg_bb    = median(deg_bb_endpoints);

        %% --------- ALL EDGES TABLE (FULL NET + IsBackbone) ---------
        T_all = table;

        T_all.Foodweb = repmat(string(dataname), n_edges_total, 1);
        T_all.Source  = src_all;
        T_all.Target  = tgt_all;

        % Degrees for endpoints
        T_all.Source_in  = indeg(src_all);
        T_all.Source_out = outdeg(src_all);
        T_all.Source_deg = totdeg(src_all);

        T_all.Target_in  = indeg(tgt_all);
        T_all.Target_out = outdeg(tgt_all);
        T_all.Target_deg = totdeg(tgt_all);

        % Optional metadata: mass
        if isfield(S, 'mass') && ~isempty(S.mass)
            mass_vec = S.mass(:); % ensure column
            if numel(mass_vec) >= n_nodes
                T_all.Source_mass = mass_vec(src_all);
                T_all.Target_mass = mass_vec(tgt_all);
            end
        end

        % Optional metadata: taxonomy
        if isfield(S, 'taxonomy') && ~isempty(S.taxonomy)
            tax_vec = S.taxonomy(:); % cell column
            if numel(tax_vec) >= n_nodes
                T_all.Source_taxonomy = string(tax_vec(src_all));
                T_all.Target_taxonomy = string(tax_vec(tgt_all));
            end
        end

        % Optional metadata: role
        if isfield(S, 'role') && ~isempty(S.role)
            role_vec = S.role(:); % cell column
            if numel(role_vec) >= n_nodes
                T_all.Source_role = string(role_vec(src_all));
                T_all.Target_role = string(role_vec(tgt_all));
            end
        end

        % Backbone flag: 1 if this edge is in B, 0 otherwise
        is_bb_vec = full(B(sub2ind(size(B), src_all, tgt_all))) ~= 0;
        T_all.IsBackbone = is_bb_vec;

        % Accumulate ALL edges
        all_edge_rows{end+1} = T_all; %#ok<AGROW>

        %% --------- BACKBONE-ONLY PER-LINK TABLE ---------
        if n_edges_backbone > 0
            T_links = table;

            T_links.Foodweb   = repmat(string(dataname), n_edges_backbone, 1);
            T_links.Source    = src_bb;
            T_links.Target    = tgt_bb;

            T_links.Source_in  = indeg(src_bb);
            T_links.Source_out = outdeg(src_bb);
            T_links.Source_deg = totdeg(src_bb);

            T_links.Target_in  = indeg(tgt_bb);
            T_links.Target_out = outdeg(tgt_bb);
            T_links.Target_deg = totdeg(tgt_bb);

            % Optional metadata: mass (vector), taxonomy (cell), role (cell)
            if isfield(S, 'mass') && ~isempty(S.mass)
                mass_vec = S.mass(:); % ensure column
                if numel(mass_vec) >= n_nodes
                    T_links.Source_mass = mass_vec(src_bb);
                    T_links.Target_mass = mass_vec(tgt_bb);
                end
            end

            if isfield(S, 'taxonomy') && ~isempty(S.taxonomy)
                tax_vec = S.taxonomy(:); % cell column
                if numel(tax_vec) >= n_nodes
                    T_links.Source_taxonomy = string(tax_vec(src_bb));
                    T_links.Target_taxonomy = string(tax_vec(tgt_bb));
                end
            end

            if isfield(S, 'role') && ~isempty(S.role)
                role_vec = S.role(:); % cell column
                if numel(role_vec) >= n_nodes
                    T_links.Source_role = string(role_vec(src_bb));
                    T_links.Target_role = string(role_vec(tgt_bb));
                end
            end

            % Accumulate backbone-only rows
            all_link_rows{end+1} = T_links; %#ok<AGROW>
        else
            fprintf('[INFO] No backbone edges found for "%s" (with current PF thresholding).\n', dataname);
        end

        %% --------- PER-FOODWEB SUMMARY ROW ---------
        sumRow = table;
        sumRow.Foodweb              = string(dataname);
        if hasEcosystem
            sumRow.EcosystemType    = meta.EcosystemType(f_idx);
        end

        sumRow.N_nodes              = n_nodes;
        sumRow.N_edges_total        = n_edges_total;
        sumRow.N_edges_backbone     = n_edges_backbone;
        sumRow.Backbone_edge_fraction = n_edges_backbone / max(n_edges_total, 1);

        sumRow.N_backbone_nodes     = n_nodes_backbone;
        sumRow.Backbone_node_fraction = n_nodes_backbone / max(n_nodes, 1);

        sumRow.HighDeg_threshold_total = highDeg_threshold;
        sumRow.N_highdeg_total      = n_high_total;
        sumRow.N_highdeg_backbone   = n_high_backbone;
        sumRow.Prop_highdeg_total   = n_high_total / max(n_nodes, 1);
        sumRow.Prop_highdeg_backbone = n_high_backbone / max(n_nodes_backbone, 1);

        sumRow.Mean_deg_all_endpoints      = mean_deg_all;
        sumRow.Median_deg_all_endpoints    = median_deg_all;
        sumRow.Mean_deg_backbone_endpoints = mean_deg_bb;
        sumRow.Median_deg_backbone_endpoints = median_deg_bb;

        % Append to summary table
        if isempty(summary_rows)
            summary_rows = sumRow;
        else
            summary_rows = [summary_rows; sumRow]; %#ok<AGROW>
        end
    end

    %% ================== WRITE OUTPUTS ==================
    % 1) Per-link long table (one row per backbone edge)
    if ~isempty(all_link_rows)
        link_table = vertcat(all_link_rows{:});
    else
        link_table = table;
    end

    % 2) ALL edges (full net) with backbone flag
    if ~isempty(all_edge_rows)
        all_edges_table = vertcat(all_edge_rows{:});
    else
        all_edges_table = table;
    end

    links_csv      = fullfile(config.backboneStatsDir, 'backbone_link_stats_long.csv');
    summary_csv    = fullfile(config.backboneStatsDir, 'backbone_link_summary.csv');
    all_edges_csv  = fullfile(config.backboneStatsDir, 'all_link_stats_long.csv');

    fprintf('\n[INFO] Writing per-link backbone stats to: %s\n', links_csv);
    writetable(link_table, links_csv);

    fprintf('[INFO] Writing per-foodweb backbone summary to: %s\n', summary_csv);
    writetable(summary_rows, summary_csv);

    fprintf('[INFO] Writing ALL-link stats (full net + IsBackbone flag) to: %s\n', all_edges_csv);
    writetable(all_edges_table, all_edges_csv);

    fprintf('\n[DONE] Backbone link stats computed for %d foodwebs.\n', height(summary_rows));
end
