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
    %   config.sweepBackboneTrain  : true/false
    %   config.backboneRatioRange  : e.g. 0.10:0.10:0.90
    %   config.backbone_q          : 0.05
    %   config.backbone_max_q      : 0.25
    %   config.backbone_q_ladder   : 2.0
    %   config.alpha_fallback      : [] or scalar in (0,1)

    % ---- defaults for new knobs ----
    if ~isfield(config,'sweepBackboneTrain'), config.sweepBackboneTrain = true; end
    if ~isfield(config,'backboneRatioRange'), config.backboneRatioRange = 0.10:0.10:0.90; end
    if ~isfield(config,'backbone_q'),        config.backbone_q        = 0.05; end
    if ~isfield(config,'backbone_max_q'),    config.backbone_max_q    = 0.25; end
    if ~isfield(config,'backbone_q_ladder'), config.backbone_q_ladder = 2.0;  end
    if ~isfield(config,'alpha_fallback'),    config.alpha_fallback    = [];   end

    % --- Detect backbone mode automatically ---
    use_backbone = isfield(data,'p_values_mat') && ~isempty(data.p_values_mat) && ...
                   (config.sweepBackboneTrain || isfield(config,'ratioBackbone'));

    if use_backbone
        if config.sweepBackboneTrain
            rb_list = config.backboneRatioRange;
        else
            if isfield(config,'ratioBackbone'), rb_list = config.ratioBackbone; else, rb_list = 0.20; end
        end
        rb_list = rb_list(:)';  % row vector
    else
        rb_list = 0;            % single run, no backbone
    end

    % --- Preallocate results ---
    R = numel(rb_list) * config.numExperiments;
    results = repmat(struct('AUC',0,'TimeElapsed','', 'K',K, 'TrainRatio',ratioTrain, ...
                            'BackboneRatio',0,'Threshold',0,'Precision',0,'Recall',0,'F1Score',0), R, 1);

    % --- Locals ---
    dataname      = data.dataname;
    net           = data.net;
    p_values_mat  = [];
    if use_backbone, p_values_mat = data.p_values_mat; end
    taxonomy      = data.taxonomy; 
    mass          = data.mass; 
    role          = data.role; 
    nodeSelection = config.nodeSelection;

    row = 0;

    for rb = rb_list
        if use_backbone
            fprintf('[SweepBackbone] ratioTrain=%.2f | ratioBackbone=%.2f\n', ratioTrain, rb);
            [train, test] = DivideNet_dir_neg(net, ratioTrain, false, false, ...
                'use_backbone',      true, ...
                'ratioBackbone',     rb, ...
                'p_values_mat',      p_values_mat, ...
                'backbone_q',        config.backbone_q, ...
                'backbone_max_q',    config.backbone_max_q, ...
                'backbone_q_ladder', config.backbone_q_ladder, ...
                'alpha_fallback',    config.alpha_fallback);
        else
            [train, test] = DivideNet_dir_neg(net, ratioTrain, false, false);
        end

        if config.useParallel
            % preallocate for parfor
            res_block(config.numExperiments) = struct('AUC',0,'TimeElapsed','', 'K',K, ...
                                                      'TrainRatio',ratioTrain,'BackboneRatio',rb, ...
                                                      'Threshold',0,'Precision',0,'Recall',0,'F1Score',0); %#ok<AGROW>
            parfor i = 1:config.numExperiments
                t0 = tic;
                [auc, thr, prec, rec, f1] = WLNM_dir_neg(dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain);
                res_block(i).AUC = auc;
                res_block(i).TimeElapsed = datestr(seconds(toc(t0)), 'HH:MM:SS');
                res_block(i).Threshold = thr;
                res_block(i).Precision = prec;
                res_block(i).Recall = rec;
                res_block(i).F1Score = f1;
            end
            for i = 1:config.numExperiments
                row = row + 1; results(row) = res_block(i);
            end
        else
            for i = 1:config.numExperiments
                t0 = tic;
                [auc, thr, prec, rec, f1] = WLNM_dir_neg(dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain);
                row = row + 1;
                results(row) = struct('AUC',auc,'TimeElapsed',datestr(seconds(toc(t0)), 'HH:MM:SS'), ...
                                      'K',K,'TrainRatio',ratioTrain,'BackboneRatio',rb, ...
                                      'Threshold',thr,'Precision',prec,'Recall',rec,'F1Score',f1);
            end
        end
    end
end

function r = one_experiment_dir_neg(i, dataname, train, test, K, ratioTrain, taxonomy, mass, role, nodeSelection)
    t0 = tic;
    disp(['Experiment ', num2str(i), ' (node selection: ', char(nodeSelection), ') - Running WLNM_dir_neg...']);

    [auc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM_dir_neg(dataname, train, test, K, taxonomy, mass, role, nodeSelection, ratioTrain);

    r = struct( ...
        'AUC', auc, ...
        'TimeElapsed', datestr(seconds(toc(t0)), 'HH:MM:SS'), ...
        'K', K, ...
        'TrainRatio', ratioTrain, ...
        'Threshold', best_threshold, ...
        'Precision', best_precision, ...
        'Recall', best_recall, ...
        'F1Score', best_f1_score);
end
