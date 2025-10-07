
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
        'useParallel',          false, ...                % Enable/disable parallel pool
        'numExperiments',       1, ...                    % Repeated experiments per food web
        'kRange',               10, ...                   % Number of nodes per subgraph
        'sweepTrainRatios',     false, ...                % Sweep over multiple ratios or fixed
        'ratioTrain',           0.8, ...                  % Default training ratio
        'trainRatioRange',      0.10:0.05:0.90, ...       % Training ratios to test
        'useDegreeStrategy',    false, ...                % Enable high2low / low2high strategy
        'enableRareLinks',      true, ...                 % master switch
        'rareFractionRange',    0.00:0.10:0.20, ...       % swept if enableRareLinks; 0 used if disabled
        'rareQuotaBasis',       'train', ...              % 'train' or 'total'
        'rarePolicy',           'hm', ...                 % 'hm','sum','product','alpha_beta'
        'alpha',                1, ...                    % for 'alpha_beta'
        'beta',                 1, ...
        'minRareInTest',        0.00, ...                 % reserve a fraction of rare in TEST (0 = off)
        'checkConnectivity',    true, ...
        'adaptiveConnectivity', true, ...
        'foodwebCSV',           'data/foodwebs_mat/foodweb_metrics_ecosystem.csv', ...
        'matFolder',            'data/foodwebs_mat/', ...
        'logDir',               'data/result/prediction_scores_logs', ...
        'terminalLogDir',       'data/result/terminal_logs/' ...
    );

    %% === SETUP ===
    if config.sweepTrainRatios
        train_ratios = config.trainRatioRange;
    else
        train_ratios = config.ratioTrain;
    end

    foodweb_list = readtable(config.foodwebCSV);
    foodweb_names = foodweb_list.Foodweb;

    % Create log directories
    ensureDir(config.logDir);
    ensureDir(config.terminalLogDir);

    % Start parallel pool if enabled
    pool_created = false;
    if config.useParallel && isempty(gcp('nocreate'))
        parpool(feature('numcores'));
        pool_created = true;
    end

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

            % Degree strategies: either ["random"] or ["high2low","low2high"]
            strategies = selectStrategies(config.useDegreeStrategy);

            % Rare sweep setup (when disabled, run a single pass with 0.00)
            if config.enableRareLinks
                rare_sweep = config.rareFractionRange;
            else
                rare_sweep = 0.00;
            end

            for strategy = strategies
                % Prepare per-strategy log file (suffix _rare if rare mode on)
                suffix   = ternary(config.enableRareLinks, '_rare', '');
                log_file = fullfile(config.logDir, sprintf('%s_results_%s%s.csv', dataname, string(strategy), suffix));

                % Create header once with a consistent schema
                if ~isfile(log_file)
                    fid = fopen(log_file, 'w');
                    % Always include RareFraction / RarePolicy / QuotaBasis for consistency
                    fprintf(fid, 'Iteration,AUC,ElapsedTime,K,TrainRatio,RareFraction,RarePolicy,QuotaBasis,BestThreshold,Precision,Recall,F1Score\n');
                    fclose(fid);
                end

                for rareFraction = rare_sweep
                    for K = config.kRange
                        rf_str = sprintf(', rareFraction: %.2f', rareFraction);
                        if ~config.enableRareLinks, rf_str = ''; end
                        fprintf('Processing with K = %d, strategy: %s%s\n', K, string(strategy), rf_str);

                        % Build split; forward rare knobs only when enabled
                        if config.enableRareLinks
                            [train, test, train_nodes, test_nodes] = DivideNet( ...
                                net, ratioTrain, string(strategy), ...
                                false, ...                               % use_original_logic
                                config.checkConnectivity, ...
                                config.adaptiveConnectivity, ...
                                rareFraction, ...                        % positional (kept for compat)
                                'rare_policy',       config.rarePolicy, ...
                                'alpha',             config.alpha, ...
                                'beta',              config.beta, ...
                                'quota_basis',       config.rareQuotaBasis, ...
                                'min_rare_in_test',  config.minRareInTest ...
                            );
                        else
                            % Normal random or degree-based split (no rare enforcement)
                            [train, test, train_nodes, test_nodes] = DivideNet( ...
                                net, ratioTrain, string(strategy), ...
                                false, ...
                                config.checkConnectivity, ...
                                config.adaptiveConnectivity, ...
                                0.00 ...            % rareFraction positional kept as 0.00
                            );
                        end

                        % Run WLNM
                        results = runExperiments(config, dataname, taxonomy, mass, role, ...
                                                 train, test, train_nodes, test_nodes, ...
                                                 K, string(strategy), ratioTrain);

                        % Append with consistent schema
                        appendResultsStable(log_file, results, rareFraction, config);
                    end
                end
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


%% === HELPER FUNCTIONS ===

function ensureDir(folder)
    if ~exist(folder, 'dir'), mkdir(folder); end
end

function strategies = selectStrategies(useDegreeStrategy)
    if useDegreeStrategy
        strategies = ["high2low", "low2high"];
    else
        strategies = ["random"];
    end
end

function results = runExperiments(config, dataname, taxonomy, mass, role, train, test, train_nodes, test_nodes, K, strategy, ratioTrain)
    results = repmat(struct('AUC', 0, 'TimeElapsed', '', 'K', K, ...
                        'TrainRatio', ratioTrain, 'Threshold', 0, ...
                        'Precision', 0, 'Recall', 0, 'F1Score', 0), ...
                        config.numExperiments, 1);
    
    if config.useParallel
        parfor i = 1:config.numExperiments
            results(i) = processExperiment(i, dataname, taxonomy, mass, role, train, test, K, train_nodes, test_nodes, strategy, ratioTrain);
        end
    else
        for i = 1:config.numExperiments
            results(i) = processExperiment(i, dataname, taxonomy, mass, role, train, test, K, train_nodes, test_nodes, strategy, ratioTrain);
        end
    end
end

function appendResultsStable(log_file, results, rareFraction, config)
    % Always writes a stable schema: includes RareFraction/RarePolicy/QuotaBasis
    for i = 1:numel(results)
        fid = fopen(log_file, 'a');
        fprintf(fid, '%d,%.4f,%s,%d,%.0f,%.2f,%s,%s,%.2f,%.4f,%.4f,%.4f\n', ...
            i, results(i).AUC, results(i).TimeElapsed, results(i).K, ...
            results(i).TrainRatio * 100, ...
            rareFraction, ...
            upper(config.rarePolicy), ...
            upper(config.rareQuotaBasis), ...
            results(i).Threshold, results(i).Precision, results(i).Recall, results(i).F1Score);
        fclose(fid);
    end
end

function result = processExperiment(ith_experiment, dataname, taxonomy, mass, role, train, test, K, train_nodes, test_nodes, strategy, ratioTrain)
    t0 = tic;
    disp(['Experiment ', num2str(ith_experiment), ' (strategy: ', char(strategy), ') — Running WLNM...']);
    [auc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM(dataname, train, test, K, taxonomy, mass, role, train_nodes, test_nodes, strategy, ratioTrain);
    elapsed_time_str = datestr(seconds(toc(t0)), 'HH:MM:SS');
    result = struct( ...
        'AUC', auc, ...
        'TimeElapsed', elapsed_time_str, ...
        'K', K, ...
        'TrainRatio', ratioTrain, ...
        'Threshold', best_threshold, ...
        'Precision', best_precision, ...
        'Recall', best_recall, ...
        'F1Score', best_f1_score);
end

function x = ternary(cond, a, b)
    if cond, x = a; else, x = b; end
end
