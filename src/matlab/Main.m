%  Main Program. Partly adapted from the codes of
%  Lu 2011, Link prediction in complex networks: A survey.
%  Muhan Zhang, Washington University in St. Louis
%
%  *author: Jorge Eduardo Castro Cruces, Queen Mary University of London

%% Configuration
useParallel = false;         % Flag to enable or disable parallel pool
kRange = 10;                % Define the interval of K values to execute
numOfExperiment = 1;
ratioTrain = 0.8;
useDegreeStrategy = false;   % NEW: Enable or disable the degree-based strategy loop

%% Load food web list from a CSV file or a predefined list
foodweb_list = readtable('data/foodwebs_mat/foodweb_metrics_small.csv');
foodweb_names = foodweb_list.Foodweb;

%% Set up logging directories
log_dir = 'data/result/prediction_scores_logs';
terminal_log_dir = 'data/result/terminal_logs/';
if ~exist(log_dir, 'dir'); mkdir(log_dir); end
if ~exist(terminal_log_dir, 'dir'); mkdir(terminal_log_dir); end

%% Start parallel pool if enabled
pool_created = false;  % NEW: track if we open the pool ourselves
if useParallel && isempty(gcp('nocreate'))
    parpool(feature('numcores'));
    pool_created = true;
end

%% Iterate over all food webs in the list
for f_idx = 1:length(foodweb_names)
    dataname = foodweb_names{f_idx};

    % Set up terminal log file
    diary_file = fullfile(terminal_log_dir, strcat(dataname, '_terminal_log.txt'));
    diary(diary_file);

    %% Load .mat data
    addpath(genpath('utils'));
    datapath = 'data/foodwebs_mat/';
    thisdatapath = fullfile(datapath, strcat(dataname, '.mat'));

    if ~isfile(thisdatapath)
        disp(['File not found: ', thisdatapath]);
        diary off;
        continue;
    end

    load(thisdatapath, 'net', 'taxonomy', 'mass', 'role');
    disp(['Processing dataset: ', dataname]);

    %% Strategy loop: Degree-based or default
    if useDegreeStrategy
        strategies = ["high2low", "low2high"];
    else
        strategies = ["random"];  % Single fallback strategy if degree-based loop is disabled
    end

    for strategy = strategies
        for K = kRange
            disp(['Processing with K = ', num2str(K), ' using strategy: ', strategy]);

            log_file = fullfile(log_dir, strcat(dataname, '_results_', strategy, '.csv'));
            if ~isfile(log_file)
                fid = fopen(log_file, 'w');
                fprintf(fid, 'Iteration,AUC,ElapsedTime,K,TrainRatio,BestThreshold,Precision,Recall,F1Score\n');
                fclose(fid);
            end

            results = repmat(struct('AUC', 0, 'TimeElapsed', '', 'K', K, ...
                                    'TrainRatio', ratioTrain, 'Threshold', 0, ...
                                    'Precision', 0, 'Recall', 0, 'F1Score', 0), ...
                                    numOfExperiment, 1);

            % Get split and nodes from DivideNet
            [train, test, train_nodes, test_nodes] = DivideNet(net, ratioTrain, strategy);

            if useParallel
                parfor ith_experiment = 1:numOfExperiment
                    results(ith_experiment) = processExperiment(ith_experiment, dataname, taxonomy, mass, role, train, test, K, train_nodes, test_nodes, strategy);
                end
            else
                for ith_experiment = 1:numOfExperiment
                    results(ith_experiment) = processExperiment(ith_experiment, dataname, taxonomy, mass, role, train, test, K, train_nodes, test_nodes, strategy);
                end
            end

            % Append to CSV log file
            for i = 1:numOfExperiment
                fid = fopen(log_file, 'a');
                fprintf(fid, '%d,%.4f,%s,%d,%.0f,%.2f,%.4f,%.4f,%.4f\n', ...
                    i, results(i).AUC, results(i).TimeElapsed, results(i).K, ...
                    results(i).TrainRatio * 100, results(i).Threshold, ...
                    results(i).Precision, results(i).Recall, results(i).F1Score);
                fclose(fid);
            end
        end
    end

    diary off;
    clear net;
end

% Close parallel pool if we opened it
if useParallel && pool_created
    delete(gcp('nocreate'));
end

disp(['Execution finished at: ', datestr(now)]);


%% Helper Function
function result = processExperiment(ith_experiment, dataname, taxonomy, mass, role, train, test, K, train_nodes, test_nodes, strategy)
    iteration_start_time = tic;

    % WLNM
    disp(['Experiment ', num2str(ith_experiment), ' (strategy: ', strategy, ') — Running WLNM...']);
    [auc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM(dataname, train, test, K, taxonomy, mass, role, train_nodes, test_nodes, strategy);

    % Time formatting
    elapsed_time_str = datestr(seconds(toc(iteration_start_time)), 'HH:MM:SS');

    % Return as structured result
    result = struct( ...
        'AUC', auc, ...
        'TimeElapsed', elapsed_time_str, ...
        'K', K, ...
        'TrainRatio', 0.8, ...
        'Threshold', best_threshold, ...
        'Precision', best_precision, ...
        'Recall', best_recall, ...
        'F1Score', best_f1_score);
end
