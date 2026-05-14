function [roc_auc, pr_auc, best_threshold, best_precision, best_recall, best_f1_score, aux] = ...
    WLNM_directed(dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain, varargin)
    %WLNM_DIRECTED Directed WLNM preserving graph direction.

    p = inputParser;
    addParameter(p, 'save_confusion', false);
    addParameter(p, 'artifact_tag', '');
    addParameter(p, 'artifact_dir', 'data/result/confusion_matrix_csv/');
    addParameter(p, 'threshold_mode', 'fixed');
    addParameter(p, 'fixed_threshold', 0.5);
    addParameter(p, 'ith_experiment', 0);
    addParameter(p, 'evaluate_on_all_unseen', false);
    addParameter(p, 'encode_parallel', false);
    addParameter(p, 'compute_ecological_metrics', true);
    parse(p, varargin{:});
    opt = p.Results;
    ith_experiment = opt.ith_experiment;

    aux = struct();

    a = 2;
    portion = 1;
    evaluate_on_all_unseen = logical(opt.evaluate_on_all_unseen);
    use_role_filter = false;
    use_original_wlnm = false;
    useParallel = logical(opt.encode_parallel);
    compute_ecological_metrics = logical(opt.compute_ecological_metrics);

    % Full directed adjacency
    htrain = train;
    htest  = test;

    % sample negative links for train and test sets
    [train_pos, train_neg, test_pos, test_neg] = sample_neg_directed( ...
        htrain, htest, role, a, portion, evaluate_on_all_unseen, use_role_filter);

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

    % Convert graphs to feature vectors
    [train_data, train_label] = graph2vector_directed( ...
        train_pos, train_neg, train, K, useParallel, dataname, use_original_wlnm);
    [test_data, test_label] = graph2vector_directed( ...
        test_pos, test_neg, train, K, useParallel, dataname, use_original_wlnm);

    % train a model
    model = 3;
    switch model
        case 1
            addpath('software/liblinear-2.1/matlab');
            train_data = sparse(train_data);
            test_data = sparse(test_data);
            [~, optim_c] = evalc('liblinear_train(train_label, train_data, ''-s 0 -C -q'');');
            model = liblinear_train(train_label, train_data, sprintf('-s 0 -c %d -q', optim_c(1)));
            [~, ~, scores] = liblinear_predict(test_label, test_data, model, '-b 1 -q');
            l1 = find(model.Label == 1);
            scores = scores(:, l1);

        case 2
            addpath('software/liblinear-2.1/matlab');
            train_data = sparse(train_data);
            test_data = sparse(test_data);
            if exist('tempdata') ~= 7
                !mkdir tempdata
            end
            train_data_full = full(train_data);
            test_data_full = full(test_data);
            writematrix([train_label, train_data_full], sprintf('tempdata/traindata_%d.csv', ith_experiment));
            writematrix([test_label, test_data_full], sprintf('tempdata/testdata_%d.csv', ith_experiment));

            cmd = sprintf('th nDNN.lua -inputdim %d -ith_experiment %d', K * (K - 1), ith_experiment);
            [status, cmdout] = system(cmd, '-echo');
            if status ~= 0
                error('External command failed: %s', cmdout);
            end

            scores = load(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));
            delete(sprintf('tempdata/traindata_%d', ith_experiment));
            delete(sprintf('tempdata/testdata_%d', ith_experiment));
            delete(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));

        case 3
            feature_dim = K * (K - 1);
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
            [~, scores] = classify(net, reshape(test_data', feature_dim, 1, 1, size(test_data, 1)));
            scores(:, 1) = [];

        case 4
            addpath('software/liblinear-2.1/matlab');
            train_data = sparse(train_data);
            test_data = sparse(test_data);
            if exist('tempdata') ~= 7
                !mkdir tempdata
            end
            libsvmwrite(sprintf('tempdata/traindata_%d', ith_experiment), train_label, train_data);
            libsvmwrite(sprintf('tempdata/testdata_%d', ith_experiment), test_label, test_data);
            cmd = sprintf('python3 nDNN.py %d %d', K * (K - 1), ith_experiment);
            system(cmd, '-echo');
            scores = load(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));
            delete(sprintf('tempdata/traindata_%d', ith_experiment));
            delete(sprintf('tempdata/testdata_%d', ith_experiment));
            delete(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));
    end

    scores = scores(:);

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
    binary_predictions = scores > best_threshold;
    test_pairs = [test_pos; test_neg];

    predicted_links = test_pairs(binary_predictions == 1, :);
    true_links      = test_pairs(test_label == 1, :);

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
    % Build empirical full web and pseudo full web (directed)
    % ------------------------------------------------------------
    n = size(train, 1);

    empirical_full = spones(train + test);
    empirical_full = empirical_full - spdiags(diag(empirical_full), 0, n, n);
    empirical_full = spones(empirical_full);

    if isempty(predicted_links)
        predicted_sparse = sparse(n, n);
    else
        predicted_sparse = sparse(predicted_links(:,1), predicted_links(:,2), 1, n, n);
    end

    pseudo_full = spones(train + predicted_sparse);
    pseudo_full = pseudo_full - spdiags(diag(pseudo_full), 0, n, n);
    pseudo_full = spones(pseudo_full);

    % ------------------------------------------------------------
    % Ecological / structural metrics
    % ------------------------------------------------------------
    if compute_ecological_metrics
        emp_metrics    = compute_foodweb_metrics(empirical_full);
        pseudo_metrics = compute_foodweb_metrics(pseudo_full);
        cmp_metrics    = compare_empirical_pseudo_webs_directed_sparse(empirical_full, pseudo_full);
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

function cmp = compare_empirical_pseudo_webs_directed_sparse(empirical_full, pseudo_full)
    if isempty(empirical_full) || isempty(pseudo_full)
        error('compare_empirical_pseudo_webs_directed_sparse:EmptyInput', ...
              'Both empirical_full and pseudo_full must be non-empty.');
    end

    if ~isequal(size(empirical_full), size(pseudo_full))
        error('compare_empirical_pseudo_webs_directed_sparse:SizeMismatch', ...
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
    TN = n * max(0, n - 1) - TP - FP - FN;

    TPR = TP / max(TP + FN, eps);
    TNR = TN / max(TN + FP, eps);
    FPR = FP / max(FP + TN, eps);
    FNR = FN / max(FN + TP, eps);

    precision = TP / max(TP + FP, eps);
    recall = TPR;
    f1_score = 2 * (precision * recall) / max(precision + recall, eps);
    TSS = TPR + TNR - 1;

    union_links = TP + FP + FN;
    if union_links > 0
        jaccard_links = TP / union_links;
    else
        jaccard_links = 0;
    end

    empirical_links = nnz(E);
    pseudo_links = nnz(P);

    cmp = struct( ...
        'TP', TP, ...
        'FP', FP, ...
        'FN', FN, ...
        'TN', TN, ...
        'TPR', TPR, ...
        'TNR', TNR, ...
        'FPR', FPR, ...
        'FNR', FNR, ...
        'Precision', precision, ...
        'Recall', recall, ...
        'F1Score', f1_score, ...
        'TSS', TSS, ...
        'JaccardLinks', jaccard_links, ...
        'EmpiricalLinks', empirical_links, ...
        'PseudoLinks', pseudo_links, ...
        'LinkDelta', pseudo_links - empirical_links ...
    );
end
