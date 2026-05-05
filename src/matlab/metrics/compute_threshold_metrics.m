function [threshold, precision, recall, f1_score] = compute_threshold_metrics(scores, labels, threshold_mode, fixed_threshold)
%COMPUTE_THRESHOLD_METRICS Choose a threshold and compute binary metrics.
%
% threshold_mode:
%   'fixed'   - use fixed_threshold, default 0.5
%   'test_f1' - legacy behavior: choose the threshold that maximizes F1 on labels

    if nargin < 3 || isempty(threshold_mode)
        threshold_mode = 'fixed';
    end
    if nargin < 4 || isempty(fixed_threshold)
        fixed_threshold = 0.5;
    end

    scores = double(scores(:));
    labels = double(labels(:));
    mode = lower(string(threshold_mode));

    switch mode
        case "fixed"
            threshold = fixed_threshold;
            [precision, recall, f1_score] = metrics_at_threshold(scores, labels, threshold);

        case {"test_f1", "test_optimal", "test-optimal"}
            thresholds = 0.1:0.05:0.9;
            best_f1 = -Inf;
            threshold = NaN;
            precision = NaN;
            recall = NaN;

            for t = thresholds
                [p, r, f1] = metrics_at_threshold(scores, labels, t);
                if f1 > best_f1
                    best_f1 = f1;
                    threshold = t;
                    precision = p;
                    recall = r;
                end
            end
            f1_score = best_f1;

        otherwise
            error('compute_threshold_metrics:UnknownMode', ...
                'Unknown threshold_mode "%s". Use "fixed" or "test_f1".', char(mode));
    end
end

function [precision, recall, f1_score] = metrics_at_threshold(scores, labels, threshold)
    pred = scores > threshold;

    TP = sum((pred == 1) & (labels == 1));
    FP = sum((pred == 1) & (labels == 0));
    FN = sum((pred == 0) & (labels == 1));

    precision = TP / max(TP + FP, eps);
    recall = TP / max(TP + FN, eps);
    f1_score = 2 * (precision * recall) / max(precision + recall, eps);
end
