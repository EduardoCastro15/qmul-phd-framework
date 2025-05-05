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

%% Load food web list from a CSV file or a predefined list
foodweb_list = readtable('data/foodwebs_mat/foodweb_metrics_small.csv');
foodweb_names = foodweb_list.Foodweb;

%% Set up logging directories
log_dir = 'data/result/';
terminal_log_dir = 'data/result/terminal_logs/';
if ~exist(log_dir, 'dir'); mkdir(log_dir); end
if ~exist(terminal_log_dir, 'dir'); mkdir(terminal_log_dir); end

%% Start parallel pool if enabled
pool_created = false;  % NEW: track if we open the pool ourselves
if useParallel
    if isempty(gcp('nocreate'))
        poolobj = parpool(feature('numcores'));
        pool_created = true;
        % parpool('local', str2double(getenv('NSLOTS')));
    end
end

%% Iterate over all food webs in the list
for f_idx = 1:length(foodweb_names)
    dataname = foodweb_names{f_idx};
    log_file = fullfile(log_dir, strcat(dataname, '_results.csv'));

    % Create CSV header if the file does not exist
    if ~isfile(log_file)
        fid = fopen(log_file, 'w');
        fprintf(fid, 'Iteration,AUC,ElapsedTime,K,TrainRatio,BestThreshold,Precision,Recall,F1Score\n');
        fclose(fid);
    else
        disp(['Skipping ', dataname, ' as it already has a log file.']);
        continue;
    end

    % Set up terminal log file
    diary_file = fullfile(terminal_log_dir, strcat('terminal_log_', dataname, '.txt'));
    diary(diary_file);

    %% Load data
    addpath(genpath('utils'));
    datapath = 'data/foodwebs_mat/';
    thisdatapath = fullfile(datapath, strcat(dataname, '.mat'));

    if ~isfile(thisdatapath)
        disp(['File not found: ', thisdatapath]);
        diary off;
        continue;
    end

    load(thisdatapath, 'net', 'taxonomy', 'mass');
    disp(['Processing dataset: ', dataname]);

    % Loop over values of K
    for K = kRange
        disp(['Processing with K = ', num2str(K)]);

        % Pre-allocate struct array
        results = repmat(struct('AUC', 0, 'TimeElapsed', '', 'K', K, ...
                                'TrainRatio', ratioTrain, 'Threshold', 0, ...
                                'Precision', 0, 'Recall', 0, 'F1Score', 0), ...
                                numOfExperiment, 1);

        if useParallel
            parfor ith_experiment = 1:numOfExperiment
                results(ith_experiment) = processExperiment(ith_experiment, dataname, net, taxonomy, mass, ratioTrain, K);
            end
        else
            for ith_experiment = 1:numOfExperiment
                results(ith_experiment) = processExperiment(ith_experiment, dataname, net, taxonomy, mass, ratioTrain, K);
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

    diary off;
    clear net;
end

% Close parallel pool if we opened it
if useParallel && pool_created
    delete(gcp('nocreate'));
end
disp(['Execution finished at: ', datestr(now)]);


%% Helper Function
function result = processExperiment(ith_experiment, dataname, net, taxonomy, mass, ratioTrain, K)
    % Usage: Train and test the network using WLNM and log the results
    % --Input--
    % - ith_experiment: experiment index, for parallel computing
    % - dataname: name of the food web
    % - net: adjacency matrix representing the network
    % - taxonomy: vector of species names
    % - mass: vector of species masses
    % - ratioTrain: proportion of edges to keep in the training set
    % - K: number of vertices in an enclosing subgraph
    % --Output--
    % - result: a struct containing the AUC, elapsed time, K, train ratio, threshold, precision, recall, and F1 score

    iteration_start_time = tic;

    % Train/test split (preserve directionality)
    [train, test] = DivideNet(net, ratioTrain);
    train = sparse(train);  % do NOT add train + train'
    test = sparse(test);

    % WLNM
    disp(['Experiment ', num2str(ith_experiment), ': Running WLNM...']);
    [auc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM(dataname, train, test, K, taxonomy, mass);

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
