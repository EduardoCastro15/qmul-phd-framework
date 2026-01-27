function results = run_wlnm_dir_neg_kfold(data, K, ratioTrain_unused, config)
%RUN_WLNM_DIR_NEG_KFOLD k-fold CV wrapper for WLNM_dir_neg.
% Train positives are all positives excluding the fold.
% Test positives are the fold.
%
% This preserves the assumption in sample_neg_dir_neg: net = train + test.

    %#ok<NASGU> ratioTrain_unused

    if ~isfield(config,'cvK'), config.cvK = 5; end
    if ~isfield(config,'cvSeed'), config.cvSeed = 12345; end
    if ~isfield(config,'cvStratifyBackbone'), config.cvStratifyBackbone = true; end
    if ~isfield(config,'cvSaveConfusion'), config.cvSaveConfusion = false; end
    if ~isfield(config,'numExperiments'), config.numExperiments = 1; end

    dataname      = data.dataname;
    net           = sparse(data.net);
    taxonomy      = data.taxonomy;
    mass          = data.mass;
    role          = data.role;
    nodeSelection = config.nodeSelection;

    backbone_mask = [];
    if isfield(data,'backbone_mask') && ~isempty(data.backbone_mask)
        backbone_mask = data.backbone_mask;
    end

    % Build folds on positive edges
    stratify = config.cvStratifyBackbone && ~isempty(backbone_mask);
    folds = kfold_edges_dir(net, config.cvK, config.cvSeed, backbone_mask, stratify);

    k = folds.k;
    ratioTrain = (k - 1) / k; % for reporting only (e.g. 0.8 if k=5)

    R = k * config.numExperiments;
    results = repmat(struct( ...
        'AUC',NaN, 'TimeElapsed','', 'K',K, ...
        'TrainRatio',ratioTrain, ...
        'CvK',k, ...
        'BackboneRatio',0, ...
        'FoldID',0, 'NumFolds',k, ...
        'ExperimentID',0, 'Seed',0, ...
        'Threshold',NaN, 'Precision',NaN, 'Recall',NaN, 'F1Score',NaN, ...
        'TotalLinks',0, 'TrainLinks',0, 'TestLinks',0, ...
        'BackboneTotal',0, 'NonBackboneTotal',0, ...
        'BackboneTrainLinks',0, 'NonBackboneTrainLinks',0, ...
        'BackboneTestLinks',0, 'NonBackboneTestLinks',0 ...
    ), R, 1);

    row = 0;

    for f = 1:k
        idx = find(folds.fold_id == f);

        test  = sparse(folds.i(idx), folds.j(idx), 1, folds.n, folds.n);
        train = net - test;

        % Sanity check: disjoint + covers all positives
        if nnz(train & test) ~= 0
            error('[CV] Train/Test overlap detected in fold %d.', f);
        end
        if nnz(train + test) ~= nnz(net)
            error('[CV] Train+Test does not cover all positives in fold %d.', f);
        end

        st = cv_split_stats(net, train, test, backbone_mask);

        for e = 1:config.numExperiments
            row = row + 1;

            % deterministic seed per (fold, experiment)
            seed = config.cvSeed + 1000*f + e;
            rng(seed, 'twister');

            t0 = tic;
            cv_tag = sprintf('cv_k%d_fold%02d', k, f);

            [auc, thr, prec, rec, f1] = WLNM_dir_neg( ...
                dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain, ...
                'cv_tag', cv_tag, ...
                'save_confusion', config.cvSaveConfusion);

            results(row).AUC         = auc;
            results(row).TimeElapsed = datestr(seconds(toc(t0)), 'HH:MM:SS');
            results(row).K           = K;
            results(row).TrainRatio  = ratioTrain;
            results(row).CvK         = k;
            results(row).BackboneRatio = 0;

            results(row).FoldID      = f;
            results(row).NumFolds    = k;
            results(row).ExperimentID= e;
            results(row).Seed        = seed;

            results(row).Threshold   = thr;
            results(row).Precision   = prec;
            results(row).Recall      = rec;
            results(row).F1Score     = f1;

            results(row).TotalLinks            = st.TotalLinks;
            results(row).TrainLinks            = st.TrainLinks;
            results(row).TestLinks             = st.TestLinks;
            results(row).BackboneTotal         = st.BackboneTotal;
            results(row).NonBackboneTotal      = st.NonBackboneTotal;
            results(row).BackboneTrainLinks    = st.BackboneTrainLinks;
            results(row).NonBackboneTrainLinks = st.NonBackboneTrainLinks;
            results(row).BackboneTestLinks     = st.BackboneTestLinks;
            results(row).NonBackboneTestLinks  = st.NonBackboneTestLinks;

            fprintf('[CV] %s | K=%d | fold %d/%d | exp %d/%d | AUC=%.4f\n', ...
                dataname, K, f, k, e, config.numExperiments, auc);
        end
    end
end
