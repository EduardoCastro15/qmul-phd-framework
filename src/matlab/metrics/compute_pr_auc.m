function pr_auc = compute_pr_auc(labels, scores)
%COMPUTE_PR_AUC Compute non-interpolated average precision.
%   PR_AUC is the precision at each distinct score threshold, weighted by
%   the increase in recall at that threshold:
%
%       PR_AUC = sum((recall(k) - recall(k-1)) * precision(k))
%
%   Equal scores are processed as one threshold. This avoids imposing an
%   arbitrary ordering within ties and matches sklearn's binary
%   average_precision_score definition.

    if ~isvector(labels) || ~isvector(scores) || isempty(labels) || ...
            numel(labels) ~= numel(scores)
        error('WLNMMetrics:InvalidPRAUCInput', ...
            'Labels and scores must be non-empty vectors of equal length.');
    end

    labels = double(labels(:));
    scores = double(scores(:));

    if any(~isfinite(labels)) || any(~isfinite(scores)) || ...
            any(~ismember(labels, [0 1]))
        error('WLNMMetrics:InvalidPRAUCInput', ...
            'Labels must be binary and labels/scores must be finite.');
    end

    positive_count = sum(labels == 1);
    negative_count = sum(labels == 0);
    if positive_count == 0 || negative_count == 0
        pr_auc = NaN;
        return;
    end

    [sorted_scores, order] = sort(scores, 'descend');
    sorted_labels = labels(order);

    cumulative_true_positives = cumsum(sorted_labels == 1);
    cumulative_false_positives = cumsum(sorted_labels == 0);

    % Evaluate the curve only after every observation in an equal-score
    % group has been included.
    threshold_ends = [find(diff(sorted_scores) ~= 0); numel(sorted_scores)];
    true_positives = cumulative_true_positives(threshold_ends);
    false_positives = cumulative_false_positives(threshold_ends);

    recall = true_positives ./ positive_count;
    precision = true_positives ./ (true_positives + false_positives);
    recall_increment = [recall(1); diff(recall)];

    pr_auc = sum(recall_increment .* precision);
end
