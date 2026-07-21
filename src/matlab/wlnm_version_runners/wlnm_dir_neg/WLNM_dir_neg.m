function [roc_auc, pr_auc, best_threshold, best_precision, best_recall, best_f1_score, aux] = ...
    WLNM_dir_neg(dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain, varargin)
    %WLNM_DIR_NEG Directed WLNM with ecology-aware negative sampling.
    %
    % Returns standard predictive metrics plus auxiliary ecological metrics
    % comparing the empirical full web against the reconstructed pseudo web.
    %
    % Assumes adjacency orientation:
    %   A(i,j) = 1 means prey/resource i -> predator/consumer j

    p = inputParser;
    addParameter(p, 'cv_tag', '');
    addParameter(p, 'save_confusion', false);
    addParameter(p, 'backbone_mask', []);          % n×n sparse logical
    addParameter(p, 'export_backbone', false);      % toggle if needed
    addParameter(p, 'evaluate_on_all_unseen', false);
    addParameter(p, 'artifact_tag', '');
    addParameter(p, 'artifact_dir', 'data/result/confusion_matrix_csv/');
    addParameter(p, 'threshold_mode', 'fixed');
    addParameter(p, 'fixed_threshold', 0.5);
    addParameter(p, 'threshold_sweep_enabled', false);
    addParameter(p, 'threshold_sweep_range', 0.10:0.10:0.90);
    addParameter(p, 'encode_parallel', false);
    addParameter(p, 'compute_ecological_metrics', true);
    addParameter(p, 'use_role_filter', true);
    addParameter(p, 'negative_mass_eligibility_enabled', []);
    addParameter(p, 'negative_mass_eligibility_threshold', []);
    addParameter(p, 'negative_mass_preference_enabled', []);   % legacy alias
    addParameter(p, 'negative_mass_preference_threshold', []); % legacy alias
    parse(p, varargin{:});
    opt = p.Results;

    a = 2;                      % negative sampling multiplier for training
    portion = 1;
    evaluate_on_all_unseen = logical(opt.evaluate_on_all_unseen);
    use_role_filter = logical(opt.use_role_filter); % preserve graph direction and filter negatives by role
    if ~isempty(opt.negative_mass_eligibility_enabled)
        negative_mass_eligibility_enabled = logical(opt.negative_mass_eligibility_enabled);
    elseif ~isempty(opt.negative_mass_preference_enabled)
        negative_mass_eligibility_enabled = logical(opt.negative_mass_preference_enabled);
    else
        negative_mass_eligibility_enabled = false;
    end
    if ~isempty(opt.negative_mass_eligibility_threshold)
        negative_mass_eligibility_threshold = double(opt.negative_mass_eligibility_threshold);
    elseif ~isempty(opt.negative_mass_preference_threshold)
        negative_mass_eligibility_threshold = double(opt.negative_mass_preference_threshold);
    else
        negative_mass_eligibility_threshold = 1.0;
    end
    use_original_wlnm = false;
    useParallel = logical(opt.encode_parallel);
    compute_ecological_metrics = logical(opt.compute_ecological_metrics);
    threshold_sweep_enabled = logical(opt.threshold_sweep_enabled);
    if threshold_sweep_enabled
        threshold_sweep_range = normalize_threshold_values(opt.threshold_sweep_range);
    else
        threshold_sweep_range = [];
    end

    htrain = train;
    htest  = test;

    % ------------------------------------------------------------
    % Sample negative links
    % ------------------------------------------------------------
    [train_pos, train_neg, test_pos, test_neg] = sample_neg_dir_neg( ...
        htrain, htest, role, a, portion, evaluate_on_all_unseen, use_role_filter, ...
        mass, negative_mass_eligibility_enabled, negative_mass_eligibility_threshold);

    % ------------------------------------------------------------
    % Sanity check
    % ------------------------------------------------------------
    if isempty(train_pos) || isempty(train_neg) || isempty(test_pos) || isempty(test_neg)
        warning('[WLNM_dir_neg] Skipping due to empty filtered sets.');

        roc_auc         = NaN;
        pr_auc          = NaN;
        if threshold_sweep_enabled
            best_threshold = threshold_sweep_range(:);
        else
            best_threshold = NaN;
        end
        best_precision  = NaN(size(best_threshold));
        best_recall     = NaN(size(best_threshold));
        best_f1_score   = NaN(size(best_threshold));
        aux = repmat(struct(), numel(best_threshold), 1);
        return;
    end

    % ------------------------------------------------------------
    % Encode subgraphs
    % ------------------------------------------------------------
    [train_data, train_label] = graph2vector_dir_neg( ...
        train_pos, train_neg, train, K, useParallel, dataname, use_original_wlnm);

    [test_data, test_label] = graph2vector_dir_neg( ...
        test_pos, test_neg, train, K, useParallel, dataname, use_original_wlnm);

    % ------------------------------------------------------------
    % Train feedforward neural network
    % ------------------------------------------------------------
    feature_dim = K * (K - 1);

    layers = [ ...
        imageInputLayer([feature_dim 1 1], 'Normalization','none')
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(16)
        reluLayer
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];

    opts = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.1, ...
        'MaxEpochs', 200, ...
        'MiniBatchSize', 128, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.9, ...
        'L2Regularization', 0, ...
        'ExecutionEnvironment', 'cpu', ...
        'Verbose', false);

    net = trainNetwork( ...
        reshape(train_data', feature_dim, 1, 1, size(train_data, 1)), ...
        categorical(train_label), ...
        layers, opts);

    % ------------------------------------------------------------
    % Predict probabilities
    % ------------------------------------------------------------
    [~, scores] = classify(net, ...
        reshape(test_data', feature_dim, 1, 1, size(test_data, 1)));

    % Keep probability of positive class
    scores(:,1) = [];
    scores = double(scores);   % ensure numeric

    % ------------------------------------------------------------
    % Compute ROC-AUC and PR-AUC
    % ------------------------------------------------------------
    [~, ~, ~, roc_auc] = perfcurve(test_label', scores', 1);
    [~, ~, ~, pr_auc]  = perfcurve(test_label', scores', 1, ...
        'XCrit', 'reca', 'YCrit', 'prec');

    % ------------------------------------------------------------
    % Thresholded metrics
    % ------------------------------------------------------------
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

    % ------------------------------------------------------------
    % Link-level outputs and threshold sweep rows
    % ------------------------------------------------------------
    test_pairs = [test_pos; test_neg];
    true_links = test_pairs(test_label == 1, :); % actual positives

    if threshold_sweep_enabled
        best_threshold = threshold_sweep_range(:);
    else
        best_threshold = primary_threshold;
    end

    [primary_predicted_links, ~] = links_and_metrics_at_threshold( ...
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
    % Build empirical full web once; pseudo full web is threshold-specific
    % ------------------------------------------------------------
    n = size(train, 1);

    % empirical_full = train positives + held-out true positives
    empirical_full = spones(train + test);

    % Remove self-loops defensively
    empirical_full = empirical_full - spdiags(diag(empirical_full), 0, n, n);

    empirical_full = spones(empirical_full);
    train_full = train - spdiags(diag(train), 0, n, n);
    train_full = spones(train_full);

    % ------------------------------------------------------------
    % Ecological / structural metrics
    % ------------------------------------------------------------
    if compute_ecological_metrics
        emp_metrics = compute_foodweb_metrics(empirical_full);
        train_metrics = compute_foodweb_metrics(train_full);
    else
        emp_metrics = struct();
        train_metrics = struct();
    end

    num_thresholds = numel(best_threshold);
    best_precision = NaN(num_thresholds, 1);
    best_recall = NaN(num_thresholds, 1);
    best_f1_score = NaN(num_thresholds, 1);
    aux = repmat(struct(), num_thresholds, 1);

    for t = 1:num_thresholds
        [predicted_links_t, test_metrics] = links_and_metrics_at_threshold( ...
            scores, test_label, test_pairs, best_threshold(t));

        if compute_ecological_metrics
            pseudo_full = build_pseudo_full(train, predicted_links_t);
            pseudo_metrics = compute_foodweb_metrics(pseudo_full);
            cmp_metrics = compare_empirical_pseudo_webs_sparse(empirical_full, pseudo_full);
        else
            pseudo_metrics = struct();
            cmp_metrics = struct();
        end

        best_precision(t) = test_metrics.Precision;
        best_recall(t) = test_metrics.Recall;
        best_f1_score(t) = test_metrics.F1Score;

        aux(t).empirical_metrics      = emp_metrics;
        aux(t).train_metrics          = train_metrics;
        aux(t).pseudo_metrics         = pseudo_metrics;
        aux(t).comparison_metrics     = cmp_metrics;
        aux(t).test_metrics           = test_metrics;
        aux(t).NumPredictedNovelLinks = size(predicted_links_t, 1);
        aux(t).NumTrueNovelLinks      = size(true_links, 1);
        aux(t).EvaluateOnAllUnseen    = evaluate_on_all_unseen;
    end

    % ------------------------------------------------------------
    % Save files
    % ------------------------------------------------------------
    base_id = sprintf('%s_K_%d_%s_ratio%.0f', dataname, K, nodeSelection, ratioTrain * 100);
    tags = {};
    if ~isempty(opt.artifact_tag)
        tags{end+1} = char(string(opt.artifact_tag));
    end
    if ~isempty(opt.cv_tag)
        tags{end+1} = char(string(opt.cv_tag));
    end
    if isempty(tags)
        exp_id = base_id;
    else
        exp_id = sprintf('%s_%s', base_id, strjoin(tags, '_'));
    end

    results_dir = char(string(opt.artifact_dir));
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end

    if opt.save_confusion
        % Save scores and labels
        scores_labels_table = table(scores, test_label, ...
            'VariableNames', {'Score', 'Label'});
        writetable(scores_labels_table, ...
            fullfile(results_dir, sprintf('%s_scores_labels.csv', exp_id)));

        % Save enriched link CSVs
        export_augmented_links(TP_links,       [exp_id '_TP_links.csv'],        taxonomy, mass, results_dir);
        export_augmented_links(FP_links,       [exp_id '_FP_links.csv'],        taxonomy, mass, results_dir);
        export_augmented_links(FN_links,       [exp_id '_FN_links.csv'],        taxonomy, mass, results_dir);
        export_augmented_links(train_pos,      [exp_id '_train_links.csv'],     taxonomy, mass, results_dir);
        export_augmented_links(primary_predicted_links,[exp_id '_predicted_links.csv'], taxonomy, mass, results_dir);

        if opt.export_backbone && ~isempty(opt.backbone_mask)
            Bmask = opt.backbone_mask;
            if ~issparse(Bmask)
                Bmask = sparse(Bmask);
            end
            Bmask = logical(Bmask);

            [bi, bj] = find(Bmask);
            backbone_links = [bi, bj];

            export_augmented_links(backbone_links, ...
                [exp_id '_backbone_links.csv'], taxonomy, mass, results_dir);
        end
    end
end

% ============================================================
% Helper: Save TP / FP / FN / predicted links with metadata
% ============================================================
function export_augmented_links(links, filename, taxonomy, mass, results_dir)
    if isempty(links)
        T = cell2table(cell(0,4), ...
            'VariableNames', {'Prey', 'Predator', 'PreyMass', 'PredatorMass'});
    else
        if size(links, 2) ~= 2
            links = reshape(links, [], 2);
        end

        prey_names     = reshape(taxonomy(links(:,1)), [], 1);
        predator_names = reshape(taxonomy(links(:,2)), [], 1);
        prey_mass      = reshape(mass(links(:,1)), [], 1);
        predator_mass  = reshape(mass(links(:,2)), [], 1);

        T = table(prey_names, predator_names, prey_mass, predator_mass, ...
            'VariableNames', {'Prey', 'Predator', 'PreyMass', 'PredatorMass'});

        % Optional ordering for readability
        T = sortrows(T, 'PredatorMass');
    end

    writetable(T, fullfile(results_dir, filename));
end

function thresholds = normalize_threshold_values(values)
    thresholds = double(values(:)');
    thresholds = thresholds(isfinite(thresholds));
    thresholds = unique(thresholds);
    thresholds = thresholds(thresholds >= 0 & thresholds <= 1);

    if isempty(thresholds)
        error('WLNM_dir_neg:InvalidThresholdSweepRange', ...
              'threshold_sweep_range must contain at least one finite value in [0, 1].');
    end
end

function [predicted_links, metrics] = links_and_metrics_at_threshold(scores, labels, test_pairs, threshold)
    binary_predictions = double(scores(:)) > threshold;
    metrics = compute_binary_classification_metrics(binary_predictions, labels);
    predicted_links = test_pairs(binary_predictions == 1, :);
end

function pseudo_full = build_pseudo_full(train, predicted_links)
    n = size(train, 1);

    if isempty(predicted_links)
        predicted_sparse = sparse(n, n);
    else
        predicted_sparse = sparse(predicted_links(:,1), predicted_links(:,2), 1, n, n);
    end

    pseudo_full = spones(train + predicted_sparse);
    pseudo_full = pseudo_full - spdiags(diag(pseudo_full), 0, n, n);
    pseudo_full = spones(pseudo_full);
end

function cmp = compare_empirical_pseudo_webs_sparse(empirical_full, pseudo_full)
    if isempty(empirical_full) || isempty(pseudo_full)
        error('compare_empirical_pseudo_webs_sparse:EmptyInput', ...
              'Both empirical_full and pseudo_full must be non-empty.');
    end

    if ~isequal(size(empirical_full), size(pseudo_full))
        error('compare_empirical_pseudo_webs_sparse:SizeMismatch', ...
              'empirical_full and pseudo_full must have the same size.');
    end

    empirical_full = spones(sparse(empirical_full));
    pseudo_full    = spones(sparse(pseudo_full));

    n = size(empirical_full, 1);

    empirical_full = empirical_full - spdiags(diag(empirical_full), 0, n, n);
    pseudo_full    = pseudo_full    - spdiags(diag(pseudo_full),    0, n, n);

    empirical_links = nnz(empirical_full);
    pseudo_links = nnz(pseudo_full);

    TP = nnz(empirical_full & pseudo_full);
    FP = pseudo_links - TP;
    FN = empirical_links - TP;
    TN = n * max(0, n - 1) - TP - FP - FN;

    TPR = TP / max(TP + FN, eps);
    TNR = TN / max(TN + FP, eps);
    FPR = FP / max(FP + TN, eps);
    FNR = FN / max(FN + TP, eps);

    precision = TP / max(TP + FP, eps);
    recall = TPR;
    f1_score = 2 * (precision * recall) / max(precision + recall, eps);
    mcc_den = sqrt(double(TP + FP) * double(TP + FN) * ...
                   double(TN + FP) * double(TN + FN));
    mcc = (double(TP) * double(TN) - double(FN) * double(FP)) / max(mcc_den, eps);

    union_links = TP + FP + FN;
    if union_links > 0
        jaccard_links = TP / union_links;
    else
        jaccard_links = 0;
    end

    cmp = struct();
    cmp.TP = TP;
    cmp.FP = FP;
    cmp.FN = FN;
    cmp.TN = TN;
    cmp.TPR = TPR;
    cmp.TNR = TNR;
    cmp.FPR = FPR;
    cmp.FNR = FNR;
    cmp.Precision = precision;
    cmp.Recall = recall;
    cmp.F1Score = f1_score;
    cmp.MCC = mcc;
    cmp.TSS = TPR + TNR - 1;
    cmp.JaccardLinks = jaccard_links;
    cmp.EmpiricalLinks = empirical_links;
    cmp.PseudoLinks = pseudo_links;
    cmp.LinkDelta = pseudo_links - empirical_links;
end

function metrics = compute_binary_classification_metrics(predictions, labels)
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

    precision = TP / max(TP + FP, eps);
    recall = TPR;
    f1_score = 2 * (precision * recall) / max(precision + recall, eps);

    n_total = TP + FP + FN + TN;
    accuracy = (TP + TN) / max(n_total, eps);
    expected_accuracy = ((TP + FP) * (TP + FN) + ...
                         (FN + TN) * (FP + TN)) / max(n_total^2, eps);
    kappa = (accuracy - expected_accuracy) / max(1 - expected_accuracy, eps);

    mcc_den = sqrt(double(TP + FP) * double(TP + FN) * ...
                   double(TN + FP) * double(TN + FN));
    mcc = (double(TP) * double(TN) - double(FN) * double(FP)) / max(mcc_den, eps);

    metrics = struct();
    metrics.TP = TP;
    metrics.FP = FP;
    metrics.FN = FN;
    metrics.TN = TN;
    metrics.TPR = TPR;
    metrics.TNR = TNR;
    metrics.FPR = FPR;
    metrics.FNR = FNR;
    metrics.Precision = precision;
    metrics.Recall = recall;
    metrics.F1Score = f1_score;
    metrics.Accuracy = accuracy;
    metrics.ExpectedAccuracy = expected_accuracy;
    metrics.Kappa = kappa;
    metrics.MCC = mcc;
    metrics.TSS = TPR + TNR - 1;
end
