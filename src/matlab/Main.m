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
    % matlab -batch "Main"

    %% === CONFIGURATION FLAGS ===

    config = struct( ...
        'useParallel',            true, ...                 % Enable/disable parallel pool
        'version',                'WLNM_dir_neg', ...      % e.g. 'WLNM_dir_neg', 'WLNM_original', 'WLNM_dir_neg_kfold', etc.
        'numExperiments',         20, ...                   % Repeated experiments per food web
        'parallelWorkers',        [], ...                  % [] auto; WLNM runners use useful workers only
        'baseSeed',               12345, ...                % Base seed for repeated holdout experiments
        'resampleSplitsEachExperiment', true, ...           % Resample train/test split for each repeated experiment
        'kRange',                 10, ...                  % Number of nodes per subgraph
        'sweepTrainRatios',       true, ...               % Sweep over multiple ratios or fixed
        'ratioTrain',             0.90, ...                 % Default training ratio
        'trainRatioRange',        0.10:0.10:0.90, ...      % Training ratios to test
        'nodeSelection',          'random', ...            % Type of node selection
        'checkConnectivity',      false, ...               % Allow train/test splits even when removing bridge links
        'adaptiveConnectivity',   true, ...                % Adapt connectivity check based on train ratio
        'use_backbone' ,          false, ...               % Enable backbone extraction
        'inverse_backbone',       false, ...               % Use non-backbone edges instead (keeps old semantics)
        'logBackboneStats',       false, ...               % Enable/disable backbone stats CSV logging
        'evaluate_on_all_unseen', false, ...              % explicit evaluation regime
        'exportBackboneCSV',      false, ...               % only export backbone links if explicitly requested
        'sweepBackboneTrain',     false, ...               % Sweep backbone *train fraction* or use fixed
        'BackboneRatio',          0.50, ...                % Fixed backboneTrainFrac if sweep disabled
        'backboneRatioRange',     [0.40 0.60 0.80], ...    % Fractions of backbone edges to put in TRAIN
        'backbone_q',             0.05, ...                % PF thresholding q
        'backbone_max_q',         0.25, ...                % PF thresholding max q
        'backbone_q_ladder',      2.0, ...                 % PF thresholding q ladder
        'alpha_fallback',         [], ...                  % PF thresholding alpha fallback
        'foodwebCSV',             'data/foodwebs_mat/foodweb_metrics_ecosystem.csv', ...              % CSV with food web names
        'matFolder',              'data/foodwebs_mat/', ...                                 % Folder with .mat files
        'logDir',                 'data/result/fixed_foodwebs_4/prediction_scores_logs', ... % Directory for result logs
        'terminalLogDir',         'data/result/fixed_foodwebs_4/terminal_logs/', ...          % Directory for terminal logs
        'artifactDir',            'data/result/fixed_foodwebs_4/confusion_matrix_csv/', ...   % Directory for auxiliary TP/FP/FN CSVs
        'exportAuxiliaryCSVs',     false, ...                                                        % Set true only for inspection CSVs; false keeps all repeated experiments parallel
        'thresholdMode',          'fixed', ...                                                        % 'fixed' or legacy 'test_f1'
        'fixedThreshold',         0.50, ...                                                           % Used when thresholdMode='fixed'
        'thresholdSweepEnabled',  true, ...                                                           % WLNM runners: write one result row per threshold
        'thresholdSweepRange',    0.10:0.10:0.90, ...                                                 % Thresholds evaluated when thresholdSweepEnabled=true
        'negativeMassPreferenceEnabled', true, ...                                                    % WLNM_dir_neg: prefer non-links where consumer mass < resource mass
        'negativeMassPreferenceThreshold', 1.0, ...                                                    % mass(target) < threshold * mass(source)
        'useGraphEncodingParallel', false, ...                                                        % WLNM runners: only useful when useParallel=false
        'computeEcologicalMetrics', true, ...                                                         % WLNM metric/comparison outputs; required for dir_neg delta t-tests
        'runDeltaTTests',         false, ...                                                          % WLNM_dir_neg: paired-difference t-tests on food-web metric deltas
        'deltaTTestAlpha',        0.05, ...                                                           % Significance level for delta t-tests
        'deltaTTestFile',         'data/result/statistical_tests/wlnm_dir_neg_delta_ttests.csv', ... % Summary CSV for delta t-tests
        'deltaTTestByEcosystemFile', '', ...                                                  % Empty derives *_by_ecosystem.csv from deltaTTestFile
        'runDeltaEquivalenceTests', false, ...                                                        % WLNM_dir_neg: TOST equivalence tests on food-web metric deltas
        'deltaEquivalenceAlpha',  0.05, ...                                                           % TOST alpha; alpha=0.05 gives a 90% CI decision rule
        'deltaEquivalenceFile',   'data/result/statistical_tests/wlnm_dir_neg_delta_equivalence.csv', ...
        'deltaEquivalenceByEcosystemFile', '', ...                                                    % Empty derives *_by_ecosystem.csv from deltaEquivalenceFile
        'deltaEquivalenceMarginsFile', 'data/result/statistical_tests/wlnm_dir_neg_delta_equivalence_margins.csv', ...
        'backboneStatsFile',      'data/result/backbone_stats/backbone_overview_per_foodweb.csv', ... % CSV for backbone stats
        'cvEnabled',              false, ...                % enable cross-validation mode
        'cvKList',                [], ...          % list of K values for K-fold CV
        'cvK',                    3, ...                   % e.g. 5-fold -> 80/20 per fold
        'cvSeed',                 12345, ...               % fold assignment seed
        'cvStratifyBackbone',     false, ...               % stratify folds by backbone/nonbackbone if mask exists
        'cvSaveConfusion',        true ...                 % save confusion matrices for each food web and fold (in terminal log dir)
    );

    config = apply_runtime_overrides(config);

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
    foodweb_indices = resolve_foodweb_indices(height(foodweb_list));
    foodweb_list = foodweb_list(foodweb_indices, :);
    foodweb_names = foodweb_list.Foodweb;
    fprintf('[Main] Processing %d foodweb(s): original index range [%d, %d]\n', ...
        numel(foodweb_names), min(foodweb_indices), max(foodweb_indices));

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

    %% === RESOLVE RUNNER FOR REQUESTED VERSION ===
    registry = get_version_registry();                         % containers.Map
    runner   = resolve_runner(registry, config.version);       % function handle
    version_key = char(lower(string(config.version)));

    if is_experiment_parallel_wlnm(version_key) && config.useParallel && ...
            get_main_config_bool(config, 'useGraphEncodingParallel', false)
        warning(['[Main] useGraphEncodingParallel=true is ignored when useParallel=true ' ...
                 'for this WLNM runner to avoid nested parfor. Disabling graph encoding parallelism.']);
        config.useGraphEncodingParallel = false;
    end

    collect_delta_ttests = strcmp(version_key, 'wlnm_dir_neg') && ...
        get_main_config_bool(config, 'runDeltaTTests', true);
    collect_delta_equivalence = strcmp(version_key, 'wlnm_dir_neg') && ...
        get_main_config_bool(config, 'runDeltaEquivalenceTests', true);
    collect_delta_stats = collect_delta_ttests || collect_delta_equivalence;

    if collect_delta_stats && ~get_main_config_bool(config, 'computeEcologicalMetrics', true)
        warning(['[Main] Delta statistical tests require computeEcologicalMetrics=true ' ...
                 'because they are computed from ecological metric deltas. Enabling it.']);
        config.computeEcologicalMetrics = true;
    end

    % Start parallel pool if enabled
    pool_created = false;
    if config.useParallel && isempty(gcp('nocreate'))
        start_parallel_pool(resolve_parallel_workers(config, version_key));
        pool_created = true;
    end

    delta_metric_rows = struct([]);

    %% === MAIN EXECUTION LOOP ===
    if config.cvEnabled
        for cvK = cvKs
            config.cvK = cvK;

            ratioTrain_cv = (cvK - 1) / cvK; % for printing / filenames only
            fprintf('=== CV: %d-fold (Train=%.2f%% / Test=%.2f%%) ===\n', ...
                cvK, ratioTrain_cv*100, (1-ratioTrain_cv)*100);

            for f_idx = 1:numel(foodweb_names)
                dataname = foodweb_names{f_idx};

                version_key = char(lower(string(config.version)));
                diary_file = fullfile(config.terminalLogDir, sprintf('%s_%s_cvK%d_terminal_log.txt', ...
                    dataname, version_key, cvK));
                diary(diary_file);

                datapath = fullfile(config.matFolder, strcat(dataname, '.mat'));
                if ~isfile(datapath)
                    fprintf('[WARN] File not found: %s\n', datapath);
                    diary off;
                    continue;
                end

                load(datapath, 'net', 'taxonomy', 'mass', 'role', 'p_values_mat');
                if ~exist('p_values_mat', 'var')
                    p_values_mat = [];
                end
                fprintf('[INFO] Processing dataset: %s\n', dataname);

                backbone_mask = []; % CV run in non-backbone mode for now

                % Log file includes cvK so files do not mix different fold regimes.
                log_file = fullfile(config.logDir, sprintf('%s_results_%s_%s_cvK%d.csv', ...
                    dataname, string(config.nodeSelection), version_key, cvK));

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
                    if collect_delta_stats
                        results = attach_foodweb_to_results(results, dataname, ecosystem_type_for_index(foodweb_list, f_idx));
                        delta_metric_rows = append_result_rows(delta_metric_rows, results);
                    end

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
                version_key = char(lower(string(config.version)));
                diary_file = fullfile(config.terminalLogDir, sprintf('%s_%s_terminal_log.txt', ...
                    dataname, version_key));

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
                if ~exist('p_values_mat', 'var')
                    p_values_mat = [];
                end
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
                log_file = fullfile(config.logDir, sprintf('%s_results_%s_%s.csv', ...
                    dataname, string(config.nodeSelection), version_key));

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
                    if collect_delta_stats
                        results = attach_foodweb_to_results(results, dataname, ecosystem_type_for_index(foodweb_list, f_idx));
                        delta_metric_rows = append_result_rows(delta_metric_rows, results);
                    end

                    % --- Append results ---
                    append_results(log_file, results, config.use_backbone);
                end

                diary off;
                clear net taxonomy mass role p_values_mat;
            end
        end
    end

    if collect_delta_ttests
        by_ecosystem_file = resolve_delta_ttest_by_ecosystem_file(config);
        write_delta_ttest_summary(config.deltaTTestFile, delta_metric_rows, ...
            'alpha', config.deltaTTestAlpha, ...
            'version', config.version, ...
            'byEcosystemFile', by_ecosystem_file);
    end

    if collect_delta_equivalence
        by_ecosystem_file = resolve_delta_equivalence_by_ecosystem_file(config);
        write_delta_equivalence_summary(config.deltaEquivalenceFile, delta_metric_rows, ...
            'alpha', config.deltaEquivalenceAlpha, ...
            'version', config.version, ...
            'byEcosystemFile', by_ecosystem_file, ...
            'marginsFile', config.deltaEquivalenceMarginsFile);
    end

    % Close parallel pool if open
    if config.useParallel && pool_created
        delete(gcp('nocreate'));
    end

    fprintf('Execution finished at: %s\n', datestr(now));
end

function value = get_main_config_bool(config, field, default_value)
    if isfield(config, field) && ~isempty(config.(field))
        value = logical(config.(field));
    else
        value = logical(default_value);
    end
end

function config = apply_runtime_overrides(config)
    config.version = get_env_text('WLNM_VERSION', config.version);
    config.foodwebCSV = get_env_text('WLNM_FOODWEB_CSV', config.foodwebCSV);
    config.numExperiments = get_env_number('WLNM_NUM_EXPERIMENTS', config.numExperiments);
    config.parallelWorkers = get_env_number('WLNM_PARALLEL_WORKERS', config.parallelWorkers);
    config.sweepTrainRatios = get_env_bool('WLNM_SWEEP_TRAIN_RATIOS', config.sweepTrainRatios);
    config.ratioTrain = get_env_number('WLNM_RATIO_TRAIN', config.ratioTrain);
    config.trainRatioRange = get_env_number_list('WLNM_TRAIN_RATIO_RANGE', config.trainRatioRange);
    config.checkConnectivity = get_env_bool('WLNM_CHECK_CONNECTIVITY', config.checkConnectivity);
    config.adaptiveConnectivity = get_env_bool('WLNM_ADAPTIVE_CONNECTIVITY', config.adaptiveConnectivity);
    config.cvEnabled = get_env_bool('WLNM_CV_ENABLED', config.cvEnabled);
    config.cvKList = get_env_number_list('WLNM_CV_K_LIST', config.cvKList);
    config.cvSaveConfusion = get_env_bool('WLNM_CV_SAVE_CONFUSION', config.cvSaveConfusion);
    config.exportAuxiliaryCSVs = get_env_bool('WLNM_EXPORT_AUXILIARY_CSVS', config.exportAuxiliaryCSVs);
    config.thresholdMode = get_env_text('WLNM_THRESHOLD_MODE', config.thresholdMode);
    config.fixedThreshold = get_env_number('WLNM_FIXED_THRESHOLD', config.fixedThreshold);
    config.thresholdSweepEnabled = get_env_bool('WLNM_THRESHOLD_SWEEP_ENABLED', config.thresholdSweepEnabled);
    config.thresholdSweepRange = get_env_number_list('WLNM_THRESHOLD_SWEEP_RANGE', config.thresholdSweepRange);
    config.negativeMassPreferenceEnabled = get_env_bool('WLNM_NEGATIVE_MASS_PREFERENCE_ENABLED', config.negativeMassPreferenceEnabled);
    config.negativeMassPreferenceThreshold = get_env_number('WLNM_NEGATIVE_MASS_PREFERENCE_THRESHOLD', config.negativeMassPreferenceThreshold);
    config.computeEcologicalMetrics = get_env_bool('WLNM_COMPUTE_ECOLOGICAL_METRICS', config.computeEcologicalMetrics);
    config.runDeltaTTests = get_env_bool('WLNM_RUN_DELTA_TTESTS', config.runDeltaTTests);
    config.runDeltaEquivalenceTests = get_env_bool('WLNM_RUN_DELTA_EQUIVALENCE', config.runDeltaEquivalenceTests);

    output_root = strtrim(getenv('WLNM_OUTPUT_ROOT'));
    if ~isempty(output_root)
        config.logDir = fullfile(output_root, 'prediction_scores_logs');
        config.terminalLogDir = fullfile(output_root, 'terminal_logs');
        config.artifactDir = fullfile(output_root, 'confusion_matrix_csv');
        config.deltaTTestFile = fullfile(output_root, 'statistical_tests', 'wlnm_dir_neg_delta_ttests.csv');
        config.deltaEquivalenceFile = fullfile(output_root, 'statistical_tests', 'wlnm_dir_neg_delta_equivalence.csv');
        config.deltaEquivalenceMarginsFile = fullfile(output_root, 'statistical_tests', 'wlnm_dir_neg_delta_equivalence_margins.csv');
        config.backboneStatsFile = fullfile(output_root, 'backbone_stats', 'backbone_overview_per_foodweb.csv');
    end
end

function indices = resolve_foodweb_indices(n)
    indices = 1:n;

    single_index = strtrim(getenv('WLNM_FOODWEB_INDEX'));
    if isempty(single_index)
        single_index = strtrim(getenv('SLURM_ARRAY_TASK_ID'));
    end

    if ~isempty(single_index)
        idx = str2double(single_index);
        if isnan(idx) || idx < 1 || idx > n || floor(idx) ~= idx
            error('[Main] Invalid foodweb index %s for n=%d.', single_index, n);
        end
        indices = idx;
        return;
    end

    start_index = get_env_number('WLNM_FOODWEB_START', 1);
    end_index = get_env_number('WLNM_FOODWEB_END', n);
    start_index = max(1, floor(start_index));
    end_index = min(n, floor(end_index));

    if start_index > end_index
        error('[Main] Invalid foodweb range [%d, %d] for n=%d.', start_index, end_index, n);
    end

    indices = start_index:end_index;
end

function workers = resolve_parallel_workers(config, version_key)
    max_workers = feature('numcores');
    slurm_cpus_per_task = str2double(getenv('SLURM_CPUS_PER_TASK'));
    slurm_tasks = str2double(getenv('SLURM_NTASKS'));

    if ~isnan(slurm_cpus_per_task) && slurm_cpus_per_task > 0
        % Leave one allocated CPU for the MATLAB client process.
        max_workers = min(max_workers, max(1, floor(slurm_cpus_per_task) - 1));
    elseif ~isnan(slurm_tasks) && slurm_tasks > 0
        % Leave one allocated CPU for the MATLAB client process.
        max_workers = min(max_workers, max(1, floor(slurm_tasks) - 1));
    end

    if isfield(config, 'parallelWorkers') && ~isempty(config.parallelWorkers)
        workers = min(max_workers, max(1, floor(double(config.parallelWorkers))));
        fprintf('[Main] Starting parallel pool with %d workers (configured).\n', workers);
        return;
    end

    workers = max_workers;

    if is_experiment_parallel_wlnm(version_key) && isfield(config, 'numExperiments')
        experiments_in_parfor = double(config.numExperiments);

        % WLNM runners run experiment 1 serially when auxiliary CSVs are
        % exported, then parallelizes the remaining experiments.
        if get_main_config_bool(config, 'exportAuxiliaryCSVs', false)
            experiments_in_parfor = max(1, experiments_in_parfor - 1);
        end

        workers = min(workers, max(1, floor(experiments_in_parfor)));
    end

    fprintf('[Main] Starting parallel pool with %d workers (auto).\n', workers);
end

function start_parallel_pool(workers)
    cluster = parcluster('Processes');
    job_storage = resolve_matlab_job_storage();

    if ~isempty(job_storage)
        if ~isfolder(job_storage)
            mkdir(job_storage);
        end

        if ~isfolder(job_storage)
            error('[Main] Could not create MATLAB JobStorageLocation: %s', job_storage);
        end

        cluster.JobStorageLocation = job_storage;
        fprintf('[Main] MATLAB parallel JobStorageLocation: %s\n', job_storage);
        parpool(cluster, workers);
        return;
    end

    parpool(workers);
end

function job_storage = resolve_matlab_job_storage()
    job_storage = strtrim(getenv('WLNM_MATLAB_JOB_STORAGE'));
    if ~isempty(job_storage)
        return;
    end

    base_dir = strtrim(getenv('SLURM_TMPDIR'));
    if isempty(base_dir)
        base_dir = strtrim(getenv('TMPDIR'));
    end
    if isempty(base_dir)
        base_dir = tempdir;
    end

    slurm_job_id = strtrim(getenv('SLURM_JOB_ID'));
    slurm_task_id = strtrim(getenv('SLURM_ARRAY_TASK_ID'));

    if isempty(slurm_job_id)
        job_storage = tempname(base_dir);
        return;
    end

    if isempty(slurm_task_id)
        slurm_task_id = 'single';
    end

    job_storage = fullfile(base_dir, 'wlnm_matlab_jobs', ...
        sprintf('%s_%s', slurm_job_id, slurm_task_id));
end

function tf = is_experiment_parallel_wlnm(version_key)
    tf = any(strcmp(version_key, {'wlnm_dir_neg', 'wlnm_dir_neg_kfold', 'wlnm_original', 'wlnm_directed', 'wlnm_negative'}));
end

function results = attach_foodweb_to_results(results, dataname, ecosystem_type)
    if nargin < 3
        ecosystem_type = '';
    end

    for i = 1:numel(results)
        results(i).Foodweb = char(string(dataname));
        results(i).EcosystemType = char(string(ecosystem_type));
    end
end

function value = get_env_text(name, default_value)
    raw = strtrim(getenv(name));
    if isempty(raw)
        value = default_value;
    else
        value = raw;
    end
end

function value = get_env_number(name, default_value)
    raw = strtrim(getenv(name));
    if isempty(raw)
        value = default_value;
        return;
    end

    parsed = str2double(raw);
    if isnan(parsed)
        error('[Main] Environment variable %s must be numeric, got "%s".', name, raw);
    end
    value = parsed;
end

function value = get_env_bool(name, default_value)
    raw = lower(strtrim(getenv(name)));
    if isempty(raw)
        value = logical(default_value);
        return;
    end

    if any(strcmp(raw, {'1', 'true', 'yes', 'on'}))
        value = true;
    elseif any(strcmp(raw, {'0', 'false', 'no', 'off'}))
        value = false;
    else
        error('[Main] Environment variable %s must be boolean, got "%s".', name, raw);
    end
end

function value = get_env_number_list(name, default_value)
    raw = strtrim(getenv(name));
    if isempty(raw)
        value = default_value;
        return;
    end

    raw = strrep(raw, '[', '');
    raw = strrep(raw, ']', '');

    if contains(raw, ':')
        parts = regexp(raw, ':', 'split');
        if numel(parts) ~= 3
            error('[Main] Environment variable %s must be start:step:end, got "%s".', name, raw);
        end
        start_value = str2double(parts{1});
        step_value = str2double(parts{2});
        end_value = str2double(parts{3});
        if any(isnan([start_value, step_value, end_value])) || step_value == 0
            error('[Main] Environment variable %s must be a numeric range, got "%s".', name, raw);
        end
        value = start_value:step_value:end_value;
        return;
    end

    parts = regexp(raw, '[,\s]+', 'split');
    parts = parts(~cellfun('isempty', parts));

    value = zeros(1, numel(parts));
    for i = 1:numel(parts)
        value(i) = str2double(parts{i});
        if isnan(value(i))
            error('[Main] Environment variable %s must be a numeric list, got "%s".', name, raw);
        end
    end
end

function ecosystem_type = ecosystem_type_for_index(foodweb_list, f_idx)
    ecosystem_type = '';
    if ismember('EcosystemType', foodweb_list.Properties.VariableNames)
        ecosystem_type = foodweb_list.EcosystemType(f_idx);
    end
end

function by_ecosystem_file = resolve_delta_ttest_by_ecosystem_file(config)
    by_ecosystem_file = '';
    if isfield(config, 'deltaTTestByEcosystemFile') && ...
            ~isempty(config.deltaTTestByEcosystemFile)
        by_ecosystem_file = char(string(config.deltaTTestByEcosystemFile));
    end

    if isempty(by_ecosystem_file)
        [out_dir, base_name, ext] = fileparts(config.deltaTTestFile);
        if isempty(ext)
            ext = '.csv';
        end
        by_ecosystem_file = fullfile(out_dir, [base_name '_by_ecosystem' ext]);
    end
end

function by_ecosystem_file = resolve_delta_equivalence_by_ecosystem_file(config)
    by_ecosystem_file = '';
    if isfield(config, 'deltaEquivalenceByEcosystemFile') && ...
            ~isempty(config.deltaEquivalenceByEcosystemFile)
        by_ecosystem_file = char(string(config.deltaEquivalenceByEcosystemFile));
    end

    if isempty(by_ecosystem_file)
        [out_dir, base_name, ext] = fileparts(config.deltaEquivalenceFile);
        if isempty(ext)
            ext = '.csv';
        end
        by_ecosystem_file = fullfile(out_dir, [base_name '_by_ecosystem' ext]);
    end
end

function rows = append_result_rows(rows, new_rows)
    if isempty(new_rows)
        return;
    end

    new_rows = new_rows(:);
    if isempty(rows)
        rows = new_rows;
    else
        rows = [rows(:); new_rows];
    end
end
