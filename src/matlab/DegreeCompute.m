function [train_pos, train_neg, test_pos, test_neg] = DegreeCompute(train_pos, train_neg, test_pos, test_neg, train_nodes, test_nodes)

    % Filter train links: both nodes in train set
    train_mask_pos = ismember(train_pos(:,1), train_nodes) & ismember(train_pos(:,2), train_nodes);
    train_mask_neg = ismember(train_neg(:,1), train_nodes) & ismember(train_neg(:,2), train_nodes);

    % Filter test links: at least one node in test set
    test_mask_pos = ismember(test_pos(:,1), test_nodes) | ismember(test_pos(:,2), test_nodes);
    test_mask_neg = ismember(test_neg(:,1), test_nodes) | ismember(test_neg(:,2), test_nodes);

    train_pos = train_pos(train_mask_pos, :);
    train_neg = train_neg(train_mask_neg, :);
    test_pos  = test_pos(test_mask_pos, :);
    test_neg  = test_neg(test_mask_neg, :);
end
