function [roc_auc, pr_auc, best_threshold, best_precision, best_recall, best_f1_score, aux] = ...
    WLNM_original(dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain, varargin)
    %WLNM_ORIGINAL Baseline WLNM with original negative sampling & encoders.

    p = inputParser;
    addParameter(p, 'save_confusion', false);
    addParameter(p, 'artifact_tag', '');
    addParameter(p, 'artifact_dir', 'data/result/confusion_matrix_csv/');
    addParameter(p, 'threshold_mode', 'fixed');
    addParameter(p, 'fixed_threshold', 0.5);
    addParameter(p, 'threshold_sweep_enabled', false);
    addParameter(p, 'threshold_sweep_range', 0.10:0.10:0.90);
    addParameter(p, 'encode_parallel', false);
    addParameter(p, 'compute_ecological_metrics', true);
    parse(p, varargin{:});
    opt = p.Results;

    a = 2;
    portion = 1;
    evaluate_on_all_unseen = false;
    use_role_filter = false;
    useParallel = logical(opt.encode_parallel);
    compute_ecological_metrics = logical(opt.compute_ecological_metrics);
    threshold_sweep_enabled = logical(opt.threshold_sweep_enabled);
    if threshold_sweep_enabled
        threshold_sweep_range = normalize_threshold_values_original_wlnm(opt.threshold_sweep_range);
    else
        threshold_sweep_range = [];
    end

    % === Original half-matrix setup (undirected) ===
    htrain = triu(train, 1);
    htest  = triu(test, 1);
    Aund = spones(htrain + htrain');

    % === Original negative sampling ===
    [train_pos, train_neg, test_pos, test_neg] = sample_neg_original( ...
        htrain, htest, a, portion, evaluate_on_all_unseen, use_role_filter);

    % Sanity check
    if isempty(train_pos) || isempty(train_neg) || isempty(test_pos) || isempty(test_neg)
        warning('[WLNM] Skipping due to empty filtered sets.');
        roc_auc = NaN;
        pr_auc = NaN;
        if threshold_sweep_enabled
            best_threshold = threshold_sweep_range(:);
        else
            best_threshold = NaN;
        end
        best_precision = NaN(size(best_threshold));
        best_recall = NaN(size(best_threshold));
        best_f1_score = NaN(size(best_threshold));
        aux = repmat(struct(), numel(best_threshold), 1);
        return;
    end

    % === Original graph encoders ===
    [train_data, train_label] = graph2vector_original(train_pos, train_neg, Aund, K, dataname, useParallel);
    [test_data, test_label] = graph2vector_original(test_pos, test_neg, Aund, K, dataname, useParallel);

    % Train feedforward neural network
    feature_dim = K * (K - 1) / 2;
    layers = [imageInputLayer([feature_dim 1 1], 'Normalization','none')
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(16)
        reluLayer
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];

    opts = trainingOptions('sgdm', 'InitialLearnRate', 0.1, 'MaxEpochs', 200, ...
        'MiniBatchSize', 128, 'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor', 0.9, 'L2Regularization', 0, ...
        'ExecutionEnvironment', 'cpu');

    net = trainNetwork( ...
        reshape(train_data', feature_dim, 1, 1, size(train_data, 1)), ...
        categorical(train_label), layers, opts);

    % Predict probabilities
    [~, scores] = classify(net, reshape(test_data', feature_dim, 1, 1, size(test_data, 1)));
    scores(:, 1) = [];
    scores = double(scores);

    % Compute ROC-AUC
    [~, ~, ~, roc_auc] = perfcurve(test_label', scores', 1);

    % Compute PR-AUC
    [~, ~, ~, pr_auc] = perfcurve(test_label', scores', 1, 'XCrit', 'reca', 'YCrit', 'prec');

    [primary_threshold, primary_precision, primary_recall, primary_f1_score] = ...
        compute_threshold_metrics(scores, test_label, opt.threshold_mode, opt.fixed_threshold);

    fprintf('Threshold mode: %s | Threshold: %.2f, Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n', ...
        char(string(opt.threshold_mode)), primary_threshold, primary_precision, primary_recall, primary_f1_score);
    if threshold_sweep_enabled
        fprintf('Threshold sweep enabled | %d thresholds from %.2f to %.2f\n', ...
            numel(threshold_sweep_range), threshold_sweep_range(1), threshold_sweep_range(end));
    end
    fprintf('ROC-AUC: %.4f\n', roc_auc);
    fprintf('PR-AUC: %.4f\n', pr_auc);

    % === Augmented Output for TP, FP, FN analysis ===
    test_pairs = [test_pos; test_neg];
    true_links = test_pairs(test_label == 1, :);

    if threshold_sweep_enabled
        best_threshold = threshold_sweep_range(:);
    else
        best_threshold = primary_threshold;
    end

    [primary_predicted_links, ~] = links_and_metrics_at_threshold_original( ...
        scores, test_label, test_pairs, primary_threshold);

    if opt.save_confusion
        TP_links = intersect(primary_predicted_links, true_links, 'rows');
        FP_links = setdiff(primary_predicted_links, true_links, 'rows');
        FN_links = setdiff(true_links, primary_predicted_links, 'rows');
    else
        TP_links = zeros(0, 2);
        FP_links = zeros(0, 2);
        FN_links = zeros(0, 2);
    end

    % ------------------------------------------------------------
    % Build empirical full graph and pseudo full graph (undirected)
    % ------------------------------------------------------------
    n = size(Aund, 1);

    empirical_full = spones(htrain + htrain' + htest + htest');
    empirical_full = empirical_full - spdiags(diag(empirical_full), 0, n, n);
    empirical_full = spones(empirical_full);

    num_thresholds = numel(best_threshold);
    best_precision = NaN(num_thresholds, 1);
    best_recall = NaN(num_thresholds, 1);
    best_f1_score = NaN(num_thresholds, 1);
    aux = repmat(struct(), num_thresholds, 1);

    for t = 1:num_thresholds
        [predicted_links_t, test_metrics] = links_and_metrics_at_threshold_original( ...
            scores, test_label, test_pairs, best_threshold(t));

        if compute_ecological_metrics
            pseudo_full = build_pseudo_full_original(htrain, predicted_links_t);
            cmp_metrics = compare_binary_graphs_original(empirical_full, pseudo_full);
        else
            cmp_metrics = struct();
        end

        best_precision(t) = test_metrics.Precision;
        best_recall(t) = test_metrics.Recall;
        best_f1_score(t) = test_metrics.F1Score;

        aux(t).comparison_metrics     = cmp_metrics;
        aux(t).test_metrics           = test_metrics;
        aux(t).NumPredictedNovelLinks = size(predicted_links_t, 1);
        aux(t).NumTrueNovelLinks      = size(true_links, 1);
        aux(t).EvaluateOnAllUnseen    = evaluate_on_all_unseen;
    end

    % Save files
    base_id = sprintf('%s_K_%d_%s_ratio%.0f', dataname, K, nodeSelection, ratioTrain * 100);
    if isempty(opt.artifact_tag)
        exp_id = base_id;
    else
        exp_id = sprintf('%s_%s', base_id, char(string(opt.artifact_tag)));
    end

    results_dir = char(string(opt.artifact_dir));
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end

    if opt.save_confusion
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
end

% === Save TP/FP/FN links with metadata ===
function export_augmented_links(links, filename, taxonomy, mass, results_dir)
    if isempty(links)
        T = cell2table(cell(0,4), 'VariableNames', {'Prey', 'Predator', 'PreyMass', 'PredatorMass'});
    else
        if size(links, 2) ~= 2
            links = reshape(links, [], 2);
        end

        prey_names = reshape(taxonomy(links(:,1)), [], 1);
        predator_names = reshape(taxonomy(links(:,2)), [], 1);
        prey_mass = reshape(mass(links(:,1)), [], 1);
        predator_mass = reshape(mass(links(:,2)), [], 1);

        T = table(prey_names, predator_names, prey_mass, predator_mass, ...
            'VariableNames', {'Prey', 'Predator', 'PreyMass', 'PredatorMass'});

        T = sortrows(T, 'PredatorMass');
    end
    writetable(T, fullfile(results_dir, filename));
end

function thresholds = normalize_threshold_values_original_wlnm(values)
    thresholds = double(values(:)');
    thresholds = thresholds(isfinite(thresholds));
    thresholds = unique(thresholds);
    thresholds = thresholds(thresholds >= 0 & thresholds <= 1);

    if isempty(thresholds)
        error('WLNM_original:InvalidThresholdSweepRange', ...
              'threshold_sweep_range must contain at least one finite value in [0, 1].');
    end
end

function [predicted_links, metrics] = links_and_metrics_at_threshold_original(scores, labels, test_pairs, threshold)
    binary_predictions = double(scores(:)) > threshold;
    metrics = compute_binary_classification_metrics_original(binary_predictions, labels);
    predicted_links = test_pairs(binary_predictions == 1, :);
end

function pseudo_full = build_pseudo_full_original(htrain, predicted_links)
    n = size(htrain, 1);

    if isempty(predicted_links)
        predicted_sparse = sparse(n, n);
    else
        predicted_sparse = sparse(predicted_links(:,1), predicted_links(:,2), 1, n, n);
        predicted_sparse = spones(predicted_sparse + predicted_sparse');
    end

    pseudo_full = spones(htrain + htrain' + predicted_sparse);
    pseudo_full = pseudo_full - spdiags(diag(pseudo_full), 0, n, n);
    pseudo_full = spones(pseudo_full);
end

% ============================================================
% Compare empirical vs pseudo graph and derive exported metrics
% ============================================================
function cmp = compare_binary_graphs_original(empirical_full, pseudo_full)

    if isempty(empirical_full) || isempty(pseudo_full)
        error('compare_binary_graphs_original:EmptyInput', ...
              'Both empirical_full and pseudo_full must be non-empty.');
    end

    if ~isequal(size(empirical_full), size(pseudo_full))
        error('compare_binary_graphs_original:SizeMismatch', ...
              'empirical_full and pseudo_full must have the same size.');
    end

    E = spones(sparse(empirical_full));
    P = spones(sparse(pseudo_full));

    n = size(E, 1);
    E = E - spdiags(diag(E), 0, n, n);
    P = P - spdiags(diag(P), 0, n, n);
    E = spones(E);
    P = spones(P);

    TP = nnz(E & P);
    FP = nnz(P) - TP;
    FN = nnz(E) - TP;
    TN = numel(E) - TP - FP - FN;

    TPR = TP / max(TP + FN, eps);
    TNR = TN / max(TN + FP, eps);
    FPR = FP / max(FP + TN, eps);
    FNR = FN / max(FN + TP, eps);
    Precision = TP / max(TP + FP, eps);
    Recall = TPR;
    F1Score = 2 * (Precision * Recall) / max(Precision + Recall, eps);
    MCCDenominator = sqrt(double(TP + FP) * double(TP + FN) * ...
                          double(TN + FP) * double(TN + FN));
    MCC = (double(TP) * double(TN) - double(FN) * double(FP)) / max(MCCDenominator, eps);
    TSS = TPR + TNR - 1;
    JaccardLinks = TP / max(TP + FP + FN, eps);

    cmp = struct( ...
        'TP', TP, ...
        'FP', FP, ...
        'FN', FN, ...
        'TN', TN, ...
        'TPR', TPR, ...
        'TNR', TNR, ...
        'FPR', FPR, ...
        'FNR', FNR, ...
        'Precision', Precision, ...
        'Recall', Recall, ...
        'F1Score', F1Score, ...
        'MCC', MCC, ...
        'TSS', TSS, ...
        'JaccardLinks', JaccardLinks ...
    );
end

function metrics = compute_binary_classification_metrics_original(predictions, labels)
    predictions = logical(predictions(:));
    labels = double(labels(:)) == 1;

    TP = sum(predictions & labels);
    FP = sum(predictions & ~labels);
    FN = sum(~predictions & labels);
    TN = sum(~predictions & ~labels);

    TPR = TP / max(TP + FN, eps);
    TNR = TN / max(TN + FP, eps);
    FPR = FP / max(FP + TN, eps);
    FNR = FN / max(FN + TP, eps);

    Precision = TP / max(TP + FP, eps);
    Recall = TPR;
    F1Score = 2 * (Precision * Recall) / max(Precision + Recall, eps);

    Total = TP + FP + FN + TN;
    Accuracy = (TP + TN) / max(Total, eps);
    ExpectedAccuracy = ((TP + FP) * (TP + FN) + ...
                        (FN + TN) * (FP + TN)) / max(Total^2, eps);
    Kappa = (Accuracy - ExpectedAccuracy) / max(1 - ExpectedAccuracy, eps);

    MCCDenominator = sqrt(double(TP + FP) * double(TP + FN) * ...
                          double(TN + FP) * double(TN + FN));
    MCC = (double(TP) * double(TN) - double(FN) * double(FP)) / max(MCCDenominator, eps);

    metrics = struct( ...
        'TP', TP, ...
        'FP', FP, ...
        'FN', FN, ...
        'TN', TN, ...
        'TPR', TPR, ...
        'TNR', TNR, ...
        'FPR', FPR, ...
        'FNR', FNR, ...
        'Precision', Precision, ...
        'Recall', Recall, ...
        'F1Score', F1Score, ...
        'Accuracy', Accuracy, ...
        'ExpectedAccuracy', ExpectedAccuracy, ...
        'Kappa', Kappa, ...
        'MCC', MCC, ...
        'TSS', TPR + TNR - 1 ...
    );
end
