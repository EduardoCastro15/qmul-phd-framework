function [auc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM(dataname, train, test, K, taxonomy, mass)
    %  Usage: the main program for Weisfeiler-Lehman Neural Machine (WLNM)
    %  --Input--
    %  -dataname: name of the food web
    %  -train: a sparse matrix of training links (1: link, 0: otherwise)
    %  -test: a sparse matrix of testing links (1: link, 0: otherwise)
    %  -K: number of vertices in an enclosing subgraph
    %  -taxonomy: vector of species names
    %  -mass: vector of species masses
    %  --Output--
    %  -auc: the AUC score of WLNM
    %  -best_threshold: the best threshold for classification
    %  -best_precision: the best precision for classification
    %  -best_recall: the best recall for classification
    %  -best_f1_score: the best F1 score for classification
    %
    %  Partly adapted from the codes of
    %  Lu 2011, Link prediction in complex networks: A survey.
    %  Muhan Zhang, Washington University in St. Louis
    %
    %  *author: Jorge Eduardo Castro Cruces, Queen Mary University of London

    useParallel = false;            % Flag to enable or disable parallel pool
    htrain = train;                 % NEW: keep directed structure
    htest = test;

    % Sample negative links
    [train_pos, train_neg, test_pos, test_neg] = sample_neg(htrain, htest, 2, 1, true);

    % Encode subgraphs into vectors
    [train_data, train_label] = graph2vector(train_pos, train_neg, train, K, useParallel);
    [test_data, test_label] = graph2vector(test_pos, test_neg, train, K, useParallel);

    % Train feedforward neural network
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

    net = trainNetwork(reshape(train_data', K*(K-1)/2, 1, 1, size(train_data, 1)), ...
        categorical(train_label), layers, opts);

    % Predict probabilities
    [~, scores] = classify(net, reshape(test_data', K*(K-1)/2, 1, 1, size(test_data, 1)));
    scores(:, 1) = [];
    disp(scores);
    disp(size(scores));

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

    fprintf('Best Threshold: %.2f, Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n', ...
        best_threshold, best_precision, best_recall, best_f1_score);
    fprintf('AUC: %.4f\n', auc);

    % === Augmented Output for TP, FP, FN analysis ===
    binary_predictions = scores' > best_threshold;
    test_pairs = [test_pos; test_neg];

    predicted_links = test_pairs(binary_predictions == 1, :);  % predicted as existing
    true_links      = test_pairs(test_label == 1, :);          % actual positives

    TP_links = intersect(predicted_links, true_links, 'rows');
    FP_links = setdiff(predicted_links, true_links, 'rows');
    FN_links = setdiff(true_links, predicted_links, 'rows');

    % === Save path setup ===
    exp_id = sprintf('%s_K_%d', dataname, K);
    results_dir = 'data/result/confusion_matrix_csv/';
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end

    % === Save scores and labels to CSV ===
    scores_labels_table = table(scores, test_label, ...
        'VariableNames', {'Score', 'Label'});
    writetable(scores_labels_table, fullfile(results_dir, ...
        sprintf('%s_scores_labels.csv', exp_id)));

    % === Save TP/FP/FN links with metadata ===
    function export_augmented_links(links, filename)
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

    % Save enriched CSVs
    export_augmented_links(TP_links, [exp_id '_TP_links.csv']);
    export_augmented_links(FP_links, [exp_id '_FP_links.csv']);
    export_augmented_links(FN_links, [exp_id '_FN_links.csv']);
end
