
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
        'useParallel',        true, ...                % Enable/disable parallel pool
        'kRange',             10, ...                   % Number of nodes per subgraph
        'numExperiments',     1, ...                    % Repeated experiments per food web
        'ratioTrain',         0.8, ...                  % Default training ratio
        'sweepTrainRatios',   true, ...                % Sweep over multiple ratios or fixed
        'useDegreeStrategy',  false, ...                % Enable high2low / low2high strategy
        'trainRatioRange',    0.10:0.05:0.30, ...       % Training ratios to test
        'useRareFractionSweep', false, ...               % Enable rare fraction sweep
        'rareFractionRange',  0.01:0.01:0.10, ...       % Fraction of rare links to include in training
        'foodwebCSV',         'data/foodwebs_mat/foodweb_metrics_ecosystem.csv', ...
        'matFolder',          'data/foodwebs_mat/', ...
        'logDir',             'data/result/prediction_scores_logs', ...
        'terminalLogDir',     'data/result/terminal_logs/' ...
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

        for f_idx = 1:length(foodweb_names)
            dataname = foodweb_names{f_idx};

            % Set up terminal log file
            diary_file = fullfile(config.terminalLogDir, strcat(dataname, '_terminal_log.txt'));
            diary(diary_file);

            % Load .mat data
            datapath = fullfile(config.matFolder, strcat(dataname, '.mat'));

            if ~isfile(datapath)
                fprintf(['File not found: ', datapath]);
                diary off;
                continue;
            end

            load(datapath, 'net', 'taxonomy', 'mass', 'role');
            fprintf(['Processing dataset: ', dataname]);

            strategies = selectStrategies(config.useDegreeStrategy, config.useRareFractionSweep);

            for strategy = strategies

                if strcmpi(strategy, 'rarelinks')
                    if config.useRareFractionSweep
                        rare_fractions = config.rareFractionRange;
                    else
                        fprintf('[INFO] Skipping rarelinks strategy as rare fraction sweep is disabled.\n');
                        continue;
                    end
                else
                    rare_fractions = 1; % Only one iteration
                end                

                for rareFraction = rare_fractions
                    for K = config.kRange
                        if strcmpi(strategy, 'rarelinks')
                            rf_str = sprintf(', rareFraction: %.2f', rareFraction);
                        else
                            rf_str = '';
                        end
                        fprintf('Processing with K = %d, strategy: %s%s\n', K, strategy, rf_str);

                        log_file = fullfile(config.logDir, sprintf('%s_results_%s.csv', dataname, strategy));                                                     

                        if ~isfile(log_file)
                            fid = fopen(log_file, 'w');
                            if strcmpi(strategy, 'rarelinks')
                                fprintf(fid, 'Iteration,AUC,ElapsedTime,K,TrainRatio,RareFraction,BestThreshold,Precision,Recall,F1Score\n');
                            else
                                fprintf(fid, 'Iteration,AUC,ElapsedTime,K,TrainRatio,BestThreshold,Precision,Recall,F1Score\n');
                            end
                            fclose(fid);
                        end                        

                        if strcmpi(strategy, 'rarelinks')
                            [train, test, train_nodes, test_nodes] = DivideNet(net, ratioTrain, strategy, false, true, true, rareFraction);
                        else
                            % Normal random or degree-based strategy
                            [train, test, train_nodes, test_nodes] = DivideNet(net, ratioTrain, strategy, false, true, true);
                        end

                        results = runExperiments(config, dataname, taxonomy, mass, role, train, test, train_nodes, test_nodes, K, strategy, ratioTrain);

                        if strcmpi(strategy, 'rarelinks')
                            appendResults(log_file, results, rareFraction);
                        else
                            appendResults(log_file, results);
                        end
                    end
                end
            end

            diary off;
            clear net;
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

    if ~exist(folder, 'dir');
        mkdir(folder);
    end

end


function strategies = selectStrategies(useDegreeStrategy, useRareFractionSweep)

    if useRareFractionSweep
        strategies = ["rarelinks"];
    else
        if useDegreeStrategy
            strategies = ["high2low", "low2high"];
        else
            strategies = ["random"];
        end
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


function appendResults(log_file, results, rareFraction)

    if nargin < 3
        rareFraction = NaN; % For strategies without rare fraction
    end
    
    for i = 1:numel(results)
        fid = fopen(log_file, 'a');
        if isnan(rareFraction)
            fprintf(fid, '%d,%.4f,%s,%d,%.0f,%.2f,%.4f,%.4f,%.4f\n', ...
                i, results(i).AUC, results(i).TimeElapsed, results(i).K, ...
                results(i).TrainRatio * 100, results(i).Threshold, ...
                results(i).Precision, results(i).Recall, results(i).F1Score);
        else
            fprintf(fid, '%d,%.4f,%s,%d,%.0f,%.2f,%.2f,%.4f,%.4f,%.4f\n', ...
                i, results(i).AUC, results(i).TimeElapsed, results(i).K, ...
                results(i).TrainRatio * 100, rareFraction, results(i).Threshold, ...
                results(i).Precision, results(i).Recall, results(i).F1Score);
        end
        fclose(fid);
    end

end


function result = processExperiment(ith_experiment, dataname, taxonomy, mass, role, train, test, K, train_nodes, test_nodes, strategy, ratioTrain)

    iteration_start_time = tic;

    % WLNM
    disp(['Experiment ', num2str(ith_experiment), ' (strategy: ', strategy, ') — Running WLNM...']);
    [auc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM(dataname, train, test, K, taxonomy, mass, role, train_nodes, test_nodes, strategy, ratioTrain);

    % Time formatting
    elapsed_time_str = datestr(seconds(toc(iteration_start_time)), 'HH:MM:SS');

    % Return as structured result
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
