function Main()
    % Main Program for WLNM-based Link Prediction
    % Adapted from:
    % - Lu 2011: Link prediction in complex networks
    % - Muhan Zhang, Washington University in St. Louis
    %
    % Author: Jorge Eduardo Castro Cruces
    % Queen Mary University of London
    %
    % NOTE (backbone semantics, 2025-11 update):
    %   BackboneRatio / backboneRatioRange are now interpreted (in STANDARD
    %   backbone mode, inverse_backbone = false) as:
    %       "fraction of BACKBONE edges to place in the TRAIN set"
    %   i.e. BackboneRatio = backboneTrainFrac in [0,1].
    %   The realized fraction of train edges that are backbone will differ
    %   per food web, and can be computed from split_stats.

    %% === CONFIGURATION FLAGS ===

    config = struct( ...
        'useParallel',            true, ...                % Enable/disable parallel pool
        'version',                'WLNM_dir_neg', ...      % e.g. 'WLNM_dir_neg', 'WLNM_original', 'WLNM_dir_neg_kfold', etc.
        'numExperiments',         10, ...                   % Repeated experiments per food web
        'kRange',                 10, ...                  % Number of nodes per subgraph
        'sweepTrainRatios',       false, ...               % Sweep over multiple ratios or fixed
        'ratioTrain',             0.6, ...                 % Default training ratio
        'trainRatioRange',        0.60:0.10:0.80, ...      % Training ratios to test
        'nodeSelection',          'random', ...            % Type of node selection
        'checkConnectivity',      true, ...                % Ensure train graph connectivity
        'adaptiveConnectivity',   true, ...                % Adapt connectivity check based on train ratio
        'use_backbone' ,          false, ...               % Enable backbone extraction
        'inverse_backbone',       false, ...               % Use non-backbone edges instead (keeps old semantics)
        'logBackboneStats',       false, ...               % Enable/disable backbone stats CSV logging
        'evaluate_on_all_unseen', true, ...              % explicit evaluation regime
        'exportBackboneCSV',      false, ...               % only export backbone links if explicitly requested
        'sweepBackboneTrain',     false, ...               % Sweep backbone *train fraction* or use fixed
        'BackboneRatio',          0.50, ...                % Fixed backboneTrainFrac if sweep disabled
        'backboneRatioRange',     [0.40 0.60 0.80], ...    % Fractions of backbone edges to put in TRAIN
        'backbone_q',             0.05, ...                % PF thresholding q
        'backbone_max_q',         0.25, ...                % PF thresholding max q
        'backbone_q_ladder',      2.0, ...                 % PF thresholding q ladder
        'alpha_fallback',         [], ...                  % PF thresholding alpha fallback
        'foodwebCSV',             'data/foodwebs_mat/foodweb_metrics_ecosystem.csv', ...              % CSV with food web names
        'matFolder',              'data/foodwebs_mat_backbones/', ...                                 % Folder with .mat files
        'logDir',                 'data/result/prediction_scores_logs', ...                           % Directory for result logs
        'terminalLogDir',         'data/result/terminal_logs/', ...                                   % Directory for terminal logs
        'backboneStatsFile',      'data/result/backbone_stats/backbone_overview_per_foodweb.csv', ... % CSV for backbone stats
        'cvEnabled',              false, ...               % enable cross-validation mode
        'cvKList',                [3], ...                 % list of K values for K-fold CV
        'cvK',                    3, ...                   % e.g. 5-fold -> 80/20 per fold
        'cvSeed',                 12345, ...               % fold assignment seed
        'cvStratifyBackbone',     false, ...               % stratify folds by backbone/nonbackbone if mask exists
        'cvSaveConfusion',        true ...                 % avoid generating huge CSVs during CV
    );

    %% === SETUP ===
    if config.sweepTrainRatios
        train_ratios = config.trainRatioRange;
    else
        train_ratios = config.ratioTrain;
    end

    % Decide which CV ks to run
    if config.cvEnabled
        % train_ratios = (config.cvK - 1) / config.cvK;

        if isfield(config,'cvKList') && ~isempty(config.cvKList)
            cvKs = config.cvKList;
        else
            cvKs = config.cvK;
        end
    else
        cvKs = []; % not used
    end

    foodweb_list = readtable(config.foodwebCSV);
    foodweb_names = foodweb_list.Foodweb;

    % Track which food webs already have backbone stats logged (for this run)
    backbone_logged = false(numel(foodweb_names), 1);

    % Start logging
    addpath(genpath('wlnm_version_runners'));
    addpath(genpath('software'));
    addpath(genpath('logging'));
    addpath(genpath('metrics'));
    addpath(genpath('data'));

    % Ensure nauty mex function is compiled
    ensure_nauty_mex();

    % Create log directories
    if ~exist(config.logDir, 'dir'); mkdir(config.logDir); end
    if ~exist(config.terminalLogDir, 'dir'); mkdir(config.terminalLogDir); end

    % Start parallel pool if enabled
    pool_created = false;
    if config.useParallel && isempty(gcp('nocreate'))
        parpool(feature('numcores'));
        pool_created = true;
    end

    %% === RESOLVE RUNNER FOR REQUESTED VERSION ===
    registry = get_version_registry();                         % containers.Map
    runner   = resolve_runner(registry, config.version);       % function handle

    %% === MAIN EXECUTION LOOP ===
    if config.cvEnabled
        for cvK = cvKs
            config.cvK = cvK;

            ratioTrain_cv = (cvK - 1) / cvK; % for printing / filenames only
            fprintf('=== CV: %d-fold (Train=%.2f%% / Test=%.2f%%) ===\n', ...
                cvK, ratioTrain_cv*100, (1-ratioTrain_cv)*100);

            for f_idx = 1:numel(foodweb_names)
                dataname = foodweb_names{f_idx};

                diary_file = fullfile(config.terminalLogDir, strcat(dataname, '_terminal_log.txt'));
                diary(diary_file);

                datapath = fullfile(config.matFolder, strcat(dataname, '.mat'));
                if ~isfile(datapath)
                    fprintf('[WARN] File not found: %s\n', datapath);
                    diary off;
                    continue;
                end

                load(datapath, 'net', 'taxonomy', 'mass', 'role', 'p_values_mat');
                fprintf('[INFO] Processing dataset: %s\n', dataname);

                backbone_mask = []; % CV run in non-backbone mode for now

                % Log file includes cvK so files don’t mix different fold regimes
                log_file = fullfile(config.logDir, sprintf('%s_results_%s_%s.csv', dataname, string(config.nodeSelection), lower(string(config.version))));

                init_log_file(log_file, config.use_backbone, config.inverse_backbone);

                data = struct();
                data.dataname      = dataname;
                data.net           = net;
                data.taxonomy      = taxonomy;
                data.mass          = mass;
                data.role          = role;
                data.p_values_mat  = p_values_mat;
                data.backbone_mask = backbone_mask;

                for K = config.kRange
                    fprintf('Processing with K = %d, node selection: %s\n', K, string(config.nodeSelection));

                    results = runner(data, K, ratioTrain_cv, config); % ratioTrain is ignored by CV runner

                    append_results(log_file, results, config.use_backbone);
                end

                diary off;
                clear net taxonomy mass role p_values_mat;
            end
        end
    else
        for ratioTrain = train_ratios
            fprintf('--- Executing train/test split: %.0f%% ---\n', ratioTrain * 100);

            for f_idx = 1:numel(foodweb_names)
                dataname = foodweb_names{f_idx};

                % Set up terminal log file
                diary_file = fullfile(config.terminalLogDir, strcat(dataname, '_terminal_log.txt'));

                % === SKIP if terminal log already exists ===
                % if isfile(diary_file)
                %     fprintf('[RESUME] Skipping "%s" because terminal log exists: %s\n', dataname, diary_file);
                %     continue;  % move to next food web
                % end

                diary(diary_file);

                % Load .mat data
                datapath = fullfile(config.matFolder, strcat(dataname, '.mat'));

                if ~isfile(datapath)
                    fprintf('[WARN] File not found: %s\n', datapath);
                    diary off;
                    continue;
                end

                load(datapath, 'net', 'taxonomy', 'mass', 'role', 'p_values_mat');
                fprintf('[INFO] Processing dataset: %s\n', dataname);

                % ---- Optional: compute backbone once per dataset (controlled from Main) ----
                backbone_mask = [];
                if config.use_backbone
                    if isempty(p_values_mat)
                        warning('[Main] use_backbone=true but p_values_mat is empty for "%s". Falling back to standard split for this dataset.', dataname);
                    else
                        % Build PF backbone (independent of TrainRatio and K)
                        [B, thr, st] = backbone_regime(net, p_values_mat, ...
                                            'q',            config.backbone_q, ...
                                            'max_q',        config.backbone_max_q, ...
                                            'q_ladder',     config.backbone_q_ladder, ...
                                            'alpha_fallback', config.alpha_fallback);

                        backbone_mask = B;

                        % Log high-level stats only once per food web
                        if config.logBackboneStats && ~backbone_logged(f_idx)
                            log_backbone_stats(config.backboneStatsFile, dataname, net, st);
                            backbone_logged(f_idx) = true;
                        end
                    end
                end

                % ---- Existing logging of WLNM results ----
                log_file = fullfile(config.logDir, sprintf('%s_results_%s_%s.csv', dataname, string(config.nodeSelection), lower(string(config.version))));

                init_log_file(log_file, config.use_backbone, config.inverse_backbone);

                % Pack data struct passed to the runner
                data = struct();                 % scalar
                data.dataname      = dataname;
                data.net           = net;
                data.taxonomy      = taxonomy;
                data.mass          = mass;
                data.role          = role;
                data.p_values_mat  = p_values_mat;
                data.backbone_mask = backbone_mask;   % precomputed backbone mask (or [])

                for K = config.kRange
                    fprintf('Processing with K = %d, node selection: %s\n', K, string(config.nodeSelection));

                    % --- Delegate to the selected version runner ---
                    results = runner(data, K, ratioTrain, config);

                    % --- Append results ---
                    append_results(log_file, results, config.use_backbone);
                end

                diary off;
                clear net taxonomy mass role p_values_mat;
            end
        end
    end

    % Close parallel pool if open
    if config.useParallel && pool_created
        delete(gcp('nocreate'));
    end

    fprintf('Execution finished at: %s\n', datestr(now));
end
