function results = run_wlnm_original(data, K, ratioTrain, config)
    % (1) Make your split (if needed)
    [train, test, ~, ~] = DivideNet( ...
        data.net, ratioTrain, string(config.nodeSelection), ...
        false, config.checkConnectivity, config.adaptiveConnectivity);

    % (2) Preallocate results as in the template
    results = repmat(struct('AUC',0,'TimeElapsed','', 'K',K,'TrainRatio',ratioTrain,...
                            'Threshold',0,'Precision',0,'Recall',0,'F1Score',0), ...
                     config.numExperiments, 1);

    % (3) Loop (parfor/for) and call your algorithm:
    for i = 1:config.numExperiments
        t0 = tic;
        % [auc, thr, prec, rec, f1] = YOUR_ALGO(...);
        elapsed_time_str = datestr(seconds(toc(t0)), 'HH:MM:SS');
        results(i).AUC        = auc;
        results(i).TimeElapsed= elapsed_time_str;
        results(i).Threshold  = thr;
        results(i).Precision  = prec;
        results(i).Recall     = rec;
        results(i).F1Score    = f1;
    end
end
