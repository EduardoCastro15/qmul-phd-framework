function results = run_wlnm_dir_neg(data, K, ratioTrain, config)
    %RUN_WLNM_DIR_NEG Runner for WLNM with directed + negative sampling.
    %
    % INPUTS
    %   data:   struct with fields dataname, net, taxonomy, mass, role
    %   K:      int (subgraph size)
    %   ratioTrain: double in (0,1)
    %   config: struct with fields (nodeSelection, useParallel, numExperiments, checkConnectivity, adaptiveConnectivity)
    %
    % OUTPUT
    %   results: struct array with fields:
    %       AUC, TimeElapsed, K, TrainRatio, Threshold, Precision, Recall, F1Score

    % Split train/test once per (food web, split ratio)

    [train, test, ~, ~] = DivideNet_dir_neg(data.net, ratioTrain, string(config.nodeSelection), false, config.checkConnectivity, config.adaptiveConnectivity);

    % Preallocate result objects
    results = repmat(struct( ...
        'AUC', 0, ...
        'TimeElapsed', '', ...
        'K', ...
        K, ...
        'TrainRatio', ...
        ratioTrain, ...
        'Threshold', 0, ...
        'Precision', 0, ...
        'Recall', 0, ...
        'F1Score', 0), ...
        config.numExperiments, ...
        1);

    % Broadcast-friendly locals (helps parfor classification)
    dataname      = data.dataname;
    taxonomy      = data.taxonomy;
    mass          = data.mass;
    role          = data.role;
    nodeSelection = config.nodeSelection;

    % Execute experiments
    if config.useParallel
        parfor i = 1:config.numExperiments
            results(i) = one_experiment_dir_neg(i, dataname, train, test, K, ratioTrain, taxonomy, mass, role, nodeSelection);
        end
    else
        for i = 1:config.numExperiments
            results(i) = one_experiment_dir_neg(i, dataname, train, test, K, ratioTrain, taxonomy, mass, role, nodeSelection);
        end
    end
end

function r = one_experiment_dir_neg(i, dataname, train, test, K, ratioTrain, taxonomy, mass, role, nodeSelection)
    t0 = tic;
    disp(['Experiment ', num2str(i), ' (node selection: ', char(nodeSelection), ') - Running WLNM_dir_neg...']);

    [auc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM_dir_neg(dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain);

    r = struct( ...
        'AUC', auc, ...
        'TimeElapsed', datestr(seconds(toc(t0)), 'HH:MM:SS'), ...
        'K', K, ...
        'TrainRatio', ratioTrain, ...
        'Threshold', best_threshold, ...
        'Precision', best_precision, ...
        'Recall', best_recall, ...
        'F1Score', best_f1_score);
end
