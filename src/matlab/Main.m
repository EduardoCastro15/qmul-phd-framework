
function Main()
    % Main Program for WLNM-based Link Prediction
    % Adapted from:
    % - Lu 2011: Link prediction in complex networks
    % - Muhan Zhang, Washington University in St. Louis
    %
    % Author: Jorge Eduardo Castro Cruces
    % Queen Mary University of London

    %% === CONFIGURATION FLAGS ===

    config = struct( ...
        'useParallel',          true, ...                % Enable/disable parallel pool
        'version',              'WLNM_original', ...       % e.g. 'WLNM_dir_neg', 'WLNM_original', 'WLNM_directed', 'WLNM_negative, etc.
        'numExperiments',       10, ...                    % Repeated experiments per food web
        'kRange',               10, ...                   % Number of nodes per subgraph
        'sweepTrainRatios',     false, ...                % Sweep over multiple ratios or fixed
        'ratioTrain',           0.8, ...                  % Default training ratio
        'trainRatioRange',      0.10:0.05:0.90, ...       % Training ratios to test
        'nodeSelection',        'random', ...             % Type of node selection
        'checkConnectivity',    true, ...                 % Ensure train graph connectivity
        'adaptiveConnectivity', true, ...                 % Adapt connectivity check based on train ratio
        'foodwebCSV',           'data/foodwebs_mat/foodweb_metrics_ecosystem.csv', ... % CSV with food web names
        'matFolder',            'data/foodwebs_mat/', ...                      % Folder with .mat files
        'logDir',               'data/result/prediction_scores_logs', ...      % Directory for result logs
        'terminalLogDir',       'data/result/terminal_logs/' ...               % Directory for terminal logs
    );

    %% === SETUP ===
    if config.sweepTrainRatios
        train_ratios = config.trainRatioRange;
    else
        train_ratios = config.ratioTrain;
    end

    foodweb_list = readtable(config.foodwebCSV);
    foodweb_names = foodweb_list.Foodweb;

    % Ensure algorithm code is on path (recursively)
    addpath(genpath('wlnm_version_runners'));
    addpath(genpath('software'));
    addpath(genpath('logging'));
    addpath(genpath('data'));

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
    for ratioTrain = train_ratios
        fprintf('--- Executing train/test split: %.0f%% ---\n', ratioTrain * 100);

        for f_idx = 1:numel(foodweb_names)
            dataname = foodweb_names{f_idx};

            % Set up terminal log file
            diary_file = fullfile(config.terminalLogDir, strcat(dataname, '_terminal_log.txt'));
            diary(diary_file);

            % Load .mat data
            datapath = fullfile(config.matFolder, strcat(dataname, '.mat'));

            if ~isfile(datapath)
                fprintf('[WARN] File not found: %s\n', datapath);
                diary off;
                continue;
            end

            load(datapath, 'net', 'taxonomy', 'mass', 'role');
            fprintf('[INFO] Processing dataset: %s\n', dataname);

            log_file = fullfile(config.logDir, sprintf('%s_results_%s.csv', dataname, string(config.nodeSelection)));
            init_log_file(log_file);

            data = struct();                 % scalar
            data.dataname = dataname;
            data.net       = net;
            data.taxonomy  = taxonomy;
            data.mass      = mass;
            data.role      = role;

            for K = config.kRange
                fprintf('Processing with K = %d, node selection: %s\n', K, string(config.nodeSelection));

                % --- Delegate to the selected version runner ---
                results = runner(data, K, ratioTrain, config);

                % --- Append results ---
                append_results(log_file, results);
            end

            diary off;
            clear net taxonomy mass role;
        end
    end

    % Close parallel pool if open
    if config.useParallel && pool_created
        delete(gcp('nocreate'));
    end

    fprintf('Execution finished at: %s\n', datestr(now));
end
