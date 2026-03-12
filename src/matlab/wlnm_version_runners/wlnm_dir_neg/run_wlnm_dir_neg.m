function results = run_wlnm_dir_neg(data, K, ratioTrain, config)
    % Runner for WLNM with directed + negative sampling + backbone sweep.
    %
    % Adds 'BackboneRatio' to each result row.
    %
    % Inputs (as before)
    %   data.net            : (n x n) binary directed adjacency
    %   data.p_values_mat   : (n x n) PF p-values (sparse or double)
    %
    % Config additions (with defaults if missing):
    %   config.use_backbone        : true/false
    %   config.inverse_backbone    : true/false
    %   config.sweepBackboneTrain  : true/false
    %   config.backboneRatioRange  : e.g. [0.20 0.50 0.80]
    %   config.BackboneRatio       : scalar, used if sweepBackboneTrain=false
    %   config.backbone_q          : 0.05
    %   config.backbone_max_q      : 0.25
    %   config.backbone_q_ladder   : 2.0
    %   config.alpha_fallback      : [] or scalar in (0,1)
    %
    % NOTE (2025-11 update, STANDARD backbone mode):
    %   In DivideNet_dir_neg, BackboneRatio is now interpreted as
    %       backboneTrainFrac = fraction of BACKBONE edges placed in TRAIN
    %   when inverse_backbone = false. In inverse_backbone=true, the old
    %   semantics are kept (BackboneRatio ≈ fraction of TOTAL edges
    %   targeted from the primary set).

    % ---- defaults for backbone knobs ----
    if ~isfield(config,'useParallel'),        config.useParallel        = false;            end
    if ~isfield(config,'use_backbone'),       config.use_backbone       = false;            end
    if ~isfield(config,'inverse_backbone'),   config.inverse_backbone   = false;            end
    if ~isfield(config,'sweepBackboneTrain'), config.sweepBackboneTrain = false;            end
    if ~isfield(config,'backboneRatioRange'), config.backboneRatioRange = [0.20 0.50 0.80]; end
    if ~isfield(config,'backbone_q'),         config.backbone_q         = 0.05;             end
    if ~isfield(config,'backbone_max_q'),     config.backbone_max_q     = 0.25;             end
    if ~isfield(config,'backbone_q_ladder'),  config.backbone_q_ladder  = 2.0;              end
    if ~isfield(config,'alpha_fallback'),     config.alpha_fallback     = [];               end

    % ---- Decide backbone mode based on Main + data ----
    has_mask  = isfield(data,'backbone_mask') && ~isempty(data.backbone_mask);
    has_pvals = isfield(data,'p_values_mat')  && ~isempty(data.p_values_mat);

    use_backbone = config.use_backbone && (has_mask || has_pvals);
    if config.use_backbone && ~use_backbone
        warning('[run_wlnm_dir_neg] use_backbone=true but neither backbone_mask nor p_values_mat provided. Falling back to standard split.');
    end

    if config.inverse_backbone && ~use_backbone
        warning('[run_wlnm_dir_neg] inverse_backbone=true but backbone info is missing; inverse mode will be ignored for this dataset.');
    end

    % Backbone "ratio" list (now ≈ backboneTrainFrac when inverse=false)
    if use_backbone
        if config.sweepBackboneTrain
            rb_list = config.backboneRatioRange;
        else
            if isfield(config,'ratioBackbone')
                rb_list = config.ratioBackbone;
            elseif isfield(config,'BackboneRatio')
                rb_list = config.BackboneRatio;
            else
                rb_list = 0.50;  % default: use ~50% of backbone edges in TRAIN
            end
        end
        rb_list = rb_list(:)';  % row vector
    else
        rb_list = 0;            % single run, no backbone
    end

    % --- Preallocate results ---
    R = numel(rb_list) * config.numExperiments;
    results = repmat(struct( ...
        'ROC_AUC',0, 'PR_AUC',0, 'TimeElapsed','', 'K',K, ...
        'TrainRatio',ratioTrain, 'BackboneRatio',0, ...
        'Threshold',0, 'Precision',0,'Recall',0,'F1Score',0, ...
        'TotalLinks',0, 'TrainLinks',0, 'TestLinks',0, ...
        'BackboneTotal',0, 'NonBackboneTotal',0, ...
        'BackboneTrainLinks',0, 'NonBackboneTrainLinks',0, ...
        'BackboneTestLinks',0, 'NonBackboneTestLinks',0 ...
    ), R, 1);

    % --- Locals ---
    dataname      = data.dataname;
    net           = data.net;
    p_values_mat  = [];
    backbone_mask = [];
    if has_pvals,  p_values_mat  = data.p_values_mat;  end
    if has_mask,   backbone_mask = data.backbone_mask; end
    taxonomy      = data.taxonomy;
    mass          = data.mass;
    role          = data.role;
    nodeSelection = config.nodeSelection;

    % --- Confusion/backbone export control ---
    if isfield(config,'cvSaveConfusion')
        save_confusion_flag = logical(config.cvSaveConfusion);
    elseif isfield(config,'save_confusion')
        save_confusion_flag = logical(config.save_confusion);
    else
        save_confusion_flag = true;
    end

    row = 0;

    for rb = rb_list
        % ------------------------------------------------------------
        % Backbone mask: obtain once per rb (for export + optional split)
        % ------------------------------------------------------------
        bb_mask = [];

        % Prefer precomputed mask from Main
        if ~isempty(backbone_mask)
            bb_mask = backbone_mask;

        % Otherwise, if p-values exist, compute backbone once here
        elseif has_pvals
            [bb_mask, ~, ~] = backbone_regime(net, p_values_mat, ...
                'q',              config.backbone_q, ...
                'max_q',          config.backbone_max_q, ...
                'q_ladder',       config.backbone_q_ladder, ...
                'alpha_fallback', config.alpha_fallback);
        end

        % If we are in backbone split mode but still no mask, warn once
        if use_backbone && isempty(bb_mask)
            warning('[run_wlnm_dir_neg] backbone split requested but bb_mask is empty. Falling back to standard split.');
        end

        % ---- Obtain train/test split (backbone or standard) ----
        if use_backbone && ~isempty(bb_mask)
            fprintf('[SweepBackbone] ratioTrain=%.2f | BackboneRatio=%.2f | inverse_backbone=%d\n', ...
                    ratioTrain, rb, config.inverse_backbone);

            args = { ...
                'use_backbone',      true, ...
                'ratioBackbone',     rb, ...
                'inverse_backbone',  config.inverse_backbone, ...
                'backbone_mask',     bb_mask ...
            };

            [train, test, split_stats] = DivideNet_dir_neg(net, ratioTrain, false, false, args{:});
        else
            [train, test, split_stats] = DivideNet_dir_neg(net, ratioTrain, false, false);
        end

        % ---- WLNM experiments ----
        if config.useParallel
            % Preallocate res_block with the SAME fields as `results`
            res_block(config.numExperiments) = struct( ...
                'ROC_AUC',0, 'PR_AUC',0, 'TimeElapsed','', 'K',K, ...
                'TrainRatio',ratioTrain, 'BackboneRatio',rb, ...
                'Threshold',0, 'Precision',0, 'Recall',0, 'F1Score',0, ...
                'TotalLinks',            split_stats.TotalLinks, ...
                'TrainLinks',            split_stats.TrainLinks, ...
                'TestLinks',             split_stats.TestLinks, ...
                'BackboneTotal',         split_stats.BackboneTotal, ...
                'NonBackboneTotal',      split_stats.NonBackboneTotal, ...
                'BackboneTrainLinks',    split_stats.BackboneTrainLinks, ...
                'NonBackboneTrainLinks', split_stats.NonBackboneTrainLinks, ...
                'BackboneTestLinks',     split_stats.BackboneTestLinks, ...
                'NonBackboneTestLinks',  split_stats.NonBackboneTestLinks ...
            );

            % ---- Run ONE serial experiment if we need to export confusion/backbone ----
            start_idx = 1;
            if save_confusion_flag
                t0 = tic;
                [roc_auc, pr_auc, thr, prec, rec, f1] = WLNM_dir_neg( ...
                    dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain, ...
                    'save_confusion', true, ...
                    'backbone_mask', bb_mask, ...
                    'export_backbone', true ...
                );

                res_block(1).ROC_AUC     = roc_auc;
                res_block(1).PR_AUC      = pr_auc;
                res_block(1).TimeElapsed = datestr(seconds(toc(t0)), 'HH:MM:SS');
                res_block(1).Threshold   = thr;
                res_block(1).Precision   = prec;
                res_block(1).Recall      = rec;
                res_block(1).F1Score     = f1;

                start_idx = 2; % remaining runs go to parfor without file output
            end

            % ---- Remaining experiments in parallel, with NO file exports ----
            if start_idx <= config.numExperiments
                parfor i = start_idx:config.numExperiments
                    t0 = tic;
                    [roc_auc, pr_auc, thr, prec, rec, f1] = WLNM_dir_neg( ...
                        dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain, ...
                        'save_confusion', false, ...
                        'backbone_mask', bb_mask, ...
                        'export_backbone', false ...
                    );

                    res_block(i).ROC_AUC     = roc_auc;
                    res_block(i).PR_AUC      = pr_auc;
                    res_block(i).TimeElapsed = datestr(seconds(toc(t0)), 'HH:MM:SS');
                    res_block(i).Threshold   = thr;
                    res_block(i).Precision   = prec;
                    res_block(i).Recall      = rec;
                    res_block(i).F1Score     = f1;
                end
            end

            % Copy into results
            for i = 1:config.numExperiments
                row = row + 1;
                results(row) = res_block(i);
            end
        else
            for i = 1:config.numExperiments
                do_export = save_confusion_flag && (i == 1);

                t0 = tic;
                [roc_auc, pr_auc, thr, prec, rec, f1] = WLNM_dir_neg( ...
                    dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain, ...
                    'save_confusion', do_export, ...
                    'backbone_mask', bb_mask, ...
                    'export_backbone', do_export ...
                );

                row = row + 1;

                results(row) = struct( ...
                    'ROC_AUC',roc_auc, ...
                    'PR_AUC',pr_auc, ...
                    'TimeElapsed',datestr(seconds(toc(t0)), 'HH:MM:SS'), ...
                    'K',K, ...
                    'TrainRatio',ratioTrain, ...
                    'BackboneRatio',rb, ...
                    'Threshold',thr, ...
                    'Precision',prec, ...
                    'Recall',rec, ...
                    'F1Score',f1, ...
                    'TotalLinks',            split_stats.TotalLinks, ...
                    'TrainLinks',            split_stats.TrainLinks, ...
                    'TestLinks',             split_stats.TestLinks, ...
                    'BackboneTotal',         split_stats.BackboneTotal, ...
                    'NonBackboneTotal',      split_stats.NonBackboneTotal, ...
                    'BackboneTrainLinks',    split_stats.BackboneTrainLinks, ...
                    'NonBackboneTrainLinks', split_stats.NonBackboneTrainLinks, ...
                    'BackboneTestLinks',     split_stats.BackboneTestLinks, ...
                    'NonBackboneTestLinks',  split_stats.NonBackboneTestLinks ...
                );
            end
        end
    end
end
