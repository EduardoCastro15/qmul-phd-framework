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
    addParameter(p, 'encode_parallel', false);
    addParameter(p, 'compute_ecological_metrics', true);
    parse(p, varargin{:});
    opt = p.Results;

    % Default aux output (filled later if possible)
    aux = struct();

    a = 2;                      % negative sampling multiplier for training
    portion = 1;
    evaluate_on_all_unseen = logical(opt.evaluate_on_all_unseen);
    use_role_filter = true;     % preserve graph direction and filter negatives by role
    use_original_wlnm = false;
    useParallel = logical(opt.encode_parallel);
    compute_ecological_metrics = logical(opt.compute_ecological_metrics);

    htrain = train;
    htest  = test;

    % ------------------------------------------------------------
    % Sample negative links
    % ------------------------------------------------------------
    [train_pos, train_neg, test_pos, test_neg] = sample_neg_dir_neg( ...
        htrain, htest, role, a, portion, evaluate_on_all_unseen, use_role_filter);

    % ------------------------------------------------------------
    % Sanity check
    % ------------------------------------------------------------
    if isempty(train_pos) || isempty(train_neg) || isempty(test_pos) || isempty(test_neg)
        warning('[WLNM_dir_neg] Skipping due to empty filtered sets.');

        roc_auc         = NaN;
        pr_auc          = NaN;
        best_threshold  = NaN;
        best_precision  = NaN;
        best_recall     = NaN;
        best_f1_score   = NaN;
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
    [best_threshold, best_precision, best_recall, best_f1_score] = ...
        compute_threshold_metrics(scores, test_label, opt.threshold_mode, opt.fixed_threshold);

    fprintf('Threshold mode: %s | Threshold: %.2f, Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n', ...
        char(string(opt.threshold_mode)), best_threshold, best_precision, best_recall, best_f1_score);
    fprintf('ROC-AUC: %.4f\n', roc_auc);
    fprintf('PR-AUC: %.4f\n', pr_auc);

    % ------------------------------------------------------------
    % Link-level outputs (TP / FP / FN)
    % ------------------------------------------------------------
    binary_predictions = scores > best_threshold;
    test_pairs = [test_pos; test_neg];

    predicted_links = test_pairs(binary_predictions == 1, :); % predicted present
    true_links      = test_pairs(test_label == 1, :);         % actual positives

    if opt.save_confusion
        TP_links = intersect(predicted_links, true_links, 'rows');
        FP_links = setdiff(predicted_links, true_links, 'rows');
        FN_links = setdiff(true_links, predicted_links, 'rows');
    else
        TP_links = zeros(0, 2);
        FP_links = zeros(0, 2);
        FN_links = zeros(0, 2);
    end

    % ------------------------------------------------------------
    % Build empirical full web and pseudo full web
    % ------------------------------------------------------------
    n = size(train, 1);

    % empirical_full = train positives + held-out true positives
    empirical_full = spones(train + test);

    % pseudo_full = train positives + all predicted positive test pairs
    if isempty(predicted_links)
        predicted_sparse = sparse(n, n);
    else
        predicted_sparse = sparse(predicted_links(:,1), predicted_links(:,2), 1, n, n);
    end
    pseudo_full = spones(train + predicted_sparse);

    % Remove self-loops defensively
    empirical_full = empirical_full - spdiags(diag(empirical_full), 0, n, n);
    pseudo_full    = pseudo_full    - spdiags(diag(pseudo_full),    0, n, n);

    empirical_full = spones(empirical_full);
    pseudo_full    = spones(pseudo_full);

    % ------------------------------------------------------------
    % Ecological / structural metrics
    % ------------------------------------------------------------
    if compute_ecological_metrics
        emp_metrics    = compute_foodweb_metrics(empirical_full);
        pseudo_metrics = compute_foodweb_metrics(pseudo_full);
        cmp_metrics    = compare_empirical_pseudo_webs_sparse(empirical_full, pseudo_full);
    else
        emp_metrics    = struct();
        pseudo_metrics = struct();
        cmp_metrics    = struct();
    end

    aux.empirical_metrics      = emp_metrics;
    aux.pseudo_metrics         = pseudo_metrics;
    aux.comparison_metrics     = cmp_metrics;
    aux.NumPredictedNovelLinks = size(predicted_links, 1);
    aux.NumTrueNovelLinks      = size(true_links, 1);
    aux.EvaluateOnAllUnseen    = evaluate_on_all_unseen;

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
        export_augmented_links(predicted_links,[exp_id '_predicted_links.csv'], taxonomy, mass, results_dir);

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
    cmp.TSS = TPR + TNR - 1;
    cmp.JaccardLinks = jaccard_links;
    cmp.EmpiricalLinks = empirical_links;
    cmp.PseudoLinks = pseudo_links;
    cmp.LinkDelta = pseudo_links - empirical_links;
end
