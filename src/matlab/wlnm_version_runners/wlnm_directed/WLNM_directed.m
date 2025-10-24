function [auc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM_directed(dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain)
    %  Usage: the main program for Weisfeiler-Lehman Neural Machine (WLNM)
    %  --Input--
    %  -train: a sparse matrix of training links (1: link, 0: otherwise)
    %  -test: a sparse matrix of testing links (1: link, 0: otherwise)
    %  -K: number of vertices in an enclosing subgraph
    %  -ith_experiment: exp index, for parallel computing
    %  --Output--
    %  -auc: the AUC score of WLNM
    %
    %  *author: Muhan Zhang, Washington University in St. Louis
    %%

    a = 2;  % how many times of negative links (w.r.t. pos links) to sample
    portion = 1;  % if specified, only a portion of the sampled train and test links be returned
    evaluate_on_all_unseen = false;  % evaluate on all unseen links
    use_role_filter = false;  % preserve graph direction and filter neg_links based on role constraints
    use_original_wlnm = false;  % use original WLNM logic for subgraph extraction
    useParallel = false;

    % Remove triu function: Use the entire adjacency matrix for directed graphs
    htrain = train;  % Use the full adjacency matrix
    htest = test;

    % sample negative links for train and test sets
    [train_pos, train_neg, test_pos, test_neg] = sample_neg_directed(htrain, htest, role, a, portion, evaluate_on_all_unseen, use_role_filter);

    % Sanity check
    if isempty(train_pos) || isempty(train_neg) || isempty(test_pos) || isempty(test_neg)
        warning('[WLNM] Skipping due to empty filtered sets.');
        auc = NaN;
        best_threshold = NaN;
        best_precision = NaN;
        best_recall = NaN;
        best_f1_score = NaN;
        return;
    end

    % Convert graphs to feature vectors
    [train_data, train_label] = graph2vector_directed(train_pos, train_neg, train, K, useParallel, dataname, use_original_wlnm);
    [test_data, test_label] = graph2vector_directed(test_pos, test_neg, train, K, useParallel, dataname, use_original_wlnm);

    % train a model
    model = 3;
    switch model
        case 1  % logistic regression
            addpath('software/liblinear-2.1/matlab');  % need to install liblinear
            train_data = sparse(train_data);
            test_data = sparse(test_data);
            [~, optim_c] = evalc('liblinear_train(train_label, train_data, ''-s 0 -C -q'');');
            model = liblinear_train(train_label, train_data, sprintf('-s 0 -c %d -q', optim_c(1)));
            [~, acc, scores] = liblinear_predict(test_label, test_data, model, '-b 1 -q');
            acc
            l1 = find(model.Label == 1);
            scores = scores(:, l1);
        case 2 % train a feedforward neural network in Torch
            addpath('software/liblinear-2.1/matlab');  % need to install liblinear
            train_data = sparse(train_data);
            test_data = sparse(test_data);
            if exist('tempdata') ~= 7
                !mkdir tempdata
            end
            % libsvmwrite(sprintf('tempdata/traindata_%d', ith_experiment), train_label, train_data);
            % libsvmwrite(sprintf('tempdata/testdata_%d', ith_experiment), test_label, test_data);  % prepare data
            % Convert sparse matrix to full matrix before writing
            train_data_full = full(train_data);
            test_data_full = full(test_data);
            % Write to CSV files
            writematrix([train_label, train_data_full], sprintf('tempdata/traindata_%d.csv', ith_experiment));
            writematrix([test_label, test_data_full], sprintf('tempdata/testdata_%d.csv', ith_experiment));
    
            cmd = sprintf('th nDNN.lua -inputdim %d -ith_experiment %d', K * (K - 1) / 2, ith_experiment);
            [status, cmdout] = system(cmd, '-echo');  % Capture the status and output of the command
            if status ~= 0
                error('External command failed: %s', cmdout);
            end
    
            scores = load(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));
            delete(sprintf('tempdata/traindata_%d', ith_experiment));  % to delete temporal train and test data
            delete(sprintf('tempdata/testdata_%d', ith_experiment));
            delete(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));
        case 3 % train a feedforward neural network in MATLAB
            layers = [imageInputLayer([K*(K-1)/2 1 1], 'Normalization','none')
                fullyConnectedLayer(32)
                reluLayer
                fullyConnectedLayer(32)
                reluLayer
                fullyConnectedLayer(16)
                reluLayer
                fullyConnectedLayer(2)
                softmaxLayer
                classificationLayer];
            opts = trainingOptions('sgdm', 'InitialLearnRate', 0.1, 'MaxEpochs', 200, 'MiniBatchSize', 128, ...
                'LearnRateSchedule','piecewise', 'LearnRateDropFactor', 0.9, 'L2Regularization', 0, ...
                'ExecutionEnvironment', 'cpu');
            net = trainNetwork(reshape(train_data', K*(K-1)/2, 1, 1, size(train_data, 1)), categorical(train_label), layers, opts);
            [~, scores] = classify(net, reshape(test_data', K*(K-1)/2, 1, 1, size(test_data, 1)));
            scores(:, 1) = [];
        case 4 % train a neural network with sklearn
            addpath('software/liblinear-2.1/matlab');  % need to install liblinear
            train_data = sparse(train_data);
            test_data = sparse(test_data);
            if exist('tempdata') ~= 7
                !mkdir tempdata
            end
            libsvmwrite(sprintf('tempdata/traindata_%d', ith_experiment), train_label, train_data);
            libsvmwrite(sprintf('tempdata/testdata_%d', ith_experiment), test_label, test_data);  % prepare data
            cmd = sprintf('python3 nDNN.py %d %d', K * (K - 1) / 2, ith_experiment);
            system(cmd, '-echo');
            scores = load(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));
            delete(sprintf('tempdata/traindata_%d', ith_experiment));  % to delete temporal train and test data
            delete(sprintf('tempdata/testdata_%d', ith_experiment));
            delete(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));
    end

    % Compute AUC
    [~, ~, ~, auc] = perfcurve(test_label', scores', 1);

    % Optimize classification threshold
    thresholds = 0.1:0.05:0.9;
    best_f1_score = 0;
    best_threshold = 0;
    best_precision = 0;
    best_recall = 0;

    for t = thresholds
        binary_predictions = scores' > t;
        TP = sum((binary_predictions == 1) & (test_label' == 1));
        FP = sum((binary_predictions == 1) & (test_label' == 0));
        FN = sum((binary_predictions == 0) & (test_label' == 1));

        precision = TP / max(TP + FP, eps);
        recall = TP / max(TP + FN, eps);
        f1_score = 2 * (precision * recall) / max(precision + recall, eps);

        if f1_score > best_f1_score
            best_f1_score = f1_score;
            best_threshold = t;
            best_precision = precision;
            best_recall = recall;
        end
    end

    fprintf('Best Threshold: %.2f, Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n', best_threshold, best_precision, best_recall, best_f1_score);
    fprintf('AUC: %.4f\n', auc);

    % === Augmented Output for TP, FP, FN analysis ===
    binary_predictions = scores' > best_threshold;
    test_pairs = [test_pos; test_neg];

    predicted_links = test_pairs(binary_predictions == 1, :);  % predicted as existing
    true_links      = test_pairs(test_label == 1, :);          % actual positives

    TP_links = intersect(predicted_links, true_links, 'rows');
    FP_links = setdiff(predicted_links, true_links, 'rows');
    FN_links = setdiff(true_links, predicted_links, 'rows');

    % Save files
    % exp_id = sprintf('%s_K_%d_%s', dataname, K, nodeSelection);
    exp_id = sprintf('%s_K_%d_%s_ratio%.0f', dataname, K, nodeSelection, ratioTrain * 100);
    results_dir = 'data/result/confusion_matrix_csv/';
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end

    % === Save scores and labels to CSV ===
    scores_labels_table = table(scores, test_label, 'VariableNames', {'Score', 'Label'});
    writetable(scores_labels_table, fullfile(results_dir, ...
        sprintf('%s_scores_labels.csv', exp_id)));

    % Save enriched CSVs
    export_augmented_links(TP_links, [exp_id '_TP_links.csv'], taxonomy, mass, results_dir);
    export_augmented_links(FP_links, [exp_id '_FP_links.csv'], taxonomy, mass, results_dir);
    export_augmented_links(FN_links, [exp_id '_FN_links.csv'], taxonomy, mass, results_dir);
    export_augmented_links(train_pos, [exp_id '_train_links.csv'], taxonomy, mass, results_dir);
end

% === Save TP/FP/FN links with metadata ===
function export_augmented_links(links, filename, taxonomy, mass, results_dir)
    if isempty(links)
        T = cell2table(cell(0,4), 'VariableNames', {'Prey', 'Predator', 'PreyMass', 'PredatorMass'});
    else
        % Ensure links are 2 columns
        if size(links, 2) ~= 2
            links = reshape(links, [], 2);
        end

        % Explicitly reshape all to column vectors
        prey_names = reshape(taxonomy(links(:,1)), [], 1);
        predator_names = reshape(taxonomy(links(:,2)), [], 1);
        prey_mass = reshape(mass(links(:,1)), [], 1);
        predator_mass = reshape(mass(links(:,2)), [], 1);

        T = table(prey_names, predator_names, prey_mass, predator_mass, ...
            'VariableNames', {'Prey', 'Predator', 'PreyMass', 'PredatorMass'});
        
        % Optional: sort by predator mass
        T = sortrows(T, 'PredatorMass');
    end
    writetable(T, fullfile(results_dir, filename));
end
