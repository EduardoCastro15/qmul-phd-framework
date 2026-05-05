function [roc_auc, pr_auc, best_threshold, best_precision, best_recall, best_f1_score, aux] = ...
    WLNM_original(dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain, varargin)
    %WLNM_ORIGINAL Baseline WLNM with original negative sampling & encoders.

    p = inputParser;
    addParameter(p, 'save_confusion', false);
    addParameter(p, 'artifact_tag', '');
    addParameter(p, 'artifact_dir', 'data/result/confusion_matrix_csv/');
    addParameter(p, 'threshold_mode', 'fixed');
    addParameter(p, 'fixed_threshold', 0.5);
    parse(p, varargin{:});
    opt = p.Results;

    aux = struct();

    a = 2;
    portion = 1;
    evaluate_on_all_unseen = false;
    use_role_filter = false;
    use_original_wlnm = false;
    useParallel = false;

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
        best_threshold = NaN;
        best_precision = NaN;
        best_recall = NaN;
        best_f1_score = NaN;
        return;
    end

    % === Original graph encoders ===
    [train_data, train_label] = graph2vector_original(train_pos, train_neg, Aund, K, dataname);
    [test_data, test_label] = graph2vector_original(test_pos, test_neg, Aund, K, dataname);

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

    % Compute ROC-AUC
    [~, ~, ~, roc_auc] = perfcurve(test_label', scores', 1);

    % Compute PR-AUC
    [~, ~, ~, pr_auc] = perfcurve(test_label', scores', 1, 'XCrit', 'reca', 'YCrit', 'prec');

    [best_threshold, best_precision, best_recall, best_f1_score] = ...
        compute_threshold_metrics(scores, test_label, opt.threshold_mode, opt.fixed_threshold);

    fprintf('Threshold mode: %s | Threshold: %.2f, Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n', ...
        char(string(opt.threshold_mode)), best_threshold, best_precision, best_recall, best_f1_score);
    fprintf('ROC-AUC: %.4f\n', roc_auc);
    fprintf('PR-AUC: %.4f\n', pr_auc);

    % === Augmented Output for TP, FP, FN analysis ===
    binary_predictions = scores' > best_threshold;
    test_pairs = [test_pos; test_neg];

    predicted_links = test_pairs(binary_predictions == 1, :);
    true_links      = test_pairs(test_label == 1, :);

    TP_links = intersect(predicted_links, true_links, 'rows');
    FP_links = setdiff(predicted_links, true_links, 'rows');
    FN_links = setdiff(true_links, predicted_links, 'rows');

    % ------------------------------------------------------------
    % Build empirical full graph and pseudo full graph (undirected)
    % ------------------------------------------------------------
    n = size(Aund, 1);

    empirical_full = spones(htrain + htrain' + htest + htest');
    empirical_full = empirical_full - spdiags(diag(empirical_full), 0, n, n);
    empirical_full = spones(empirical_full);

    if isempty(predicted_links)
        predicted_sparse = sparse(n, n);
    else
        predicted_sparse = sparse(predicted_links(:,1), predicted_links(:,2), 1, n, n);
        predicted_sparse = spones(predicted_sparse + predicted_sparse');
    end

    pseudo_full = spones(htrain + htrain' + predicted_sparse);
    pseudo_full = pseudo_full - spdiags(diag(pseudo_full), 0, n, n);
    pseudo_full = spones(pseudo_full);

    cmp_metrics = compare_binary_graphs_original(empirical_full, pseudo_full);

    aux.comparison_metrics     = cmp_metrics;
    aux.NumPredictedNovelLinks = size(predicted_links, 1);
    aux.NumTrueNovelLinks      = size(true_links, 1);
    aux.EvaluateOnAllUnseen    = evaluate_on_all_unseen;

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

% ============================================================
% Compare empirical vs pseudo graph and derive exported metrics
% ============================================================
function cmp = compare_binary_graphs_original(empirical_full, pseudo_full)

    E = logical(full(empirical_full));
    P = logical(full(pseudo_full));

    TP = sum(E(:) & P(:));
    FP = sum(~E(:) & P(:));
    FN = sum(E(:) & ~P(:));
    TN = sum(~E(:) & ~P(:));

    TPR = TP / max(TP + FN, eps);
    TNR = TN / max(TN + FP, eps);
    FPR = FP / max(FP + TN, eps);
    FNR = FN / max(FN + TP, eps);
    Precision = TP / max(TP + FP, eps);
    Recall = TPR;
    F1Score = 2 * (Precision * Recall) / max(Precision + Recall, eps);
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
        'TSS', TSS, ...
        'JaccardLinks', JaccardLinks ...
    );
end
