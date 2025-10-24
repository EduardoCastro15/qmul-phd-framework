function [auc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM_negative(dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain)
    %WLNM_NEGATIVE Baseline WLNM with original negative sampling & encoders.
    % Preserves:
    %   - Augmented TP/FP/FN analysis
    %   - Save files
    %   - Save scores & labels to CSV
    %   - Save TP/FP/FN links with metadata
    %   - Save enriched CSVs

    %#ok<INUSD> % role not used in original pipeline, kept for signature parity

    a = 2;  % how many times of negative links (w.r.t. pos links) to sample
    portion = 1;  % if specified, only a portion of the sampled train and test links be returned
    evaluate_on_all_unseen = false;  % evaluate on all unseen links
    use_role_filter = true;  % preserve graph direction and filter neg_links based on role constraints
    use_original_wlnm = false;  % use original WLNM logic for subgraph extraction
    useParallel = false;

    % === Original half-matrix setup (undirected) ===
    htrain = triu(train, 1);
    htest  = triu(test, 1);

    % === Original negative sampling ===
    [train_pos, train_neg, test_pos, test_neg] = sample_neg_negative(htrain, htest, role, a, portion, evaluate_on_all_unseen, use_role_filter);

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

    % === Original graph encoders ===
    [train_data, train_label] = graph2vector_negative(train_pos, train_neg, train, K, dataname);
    [test_data, test_label] = graph2vector_negative(test_pos, test_neg, train, K, dataname);

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
    % scores is N×1

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
