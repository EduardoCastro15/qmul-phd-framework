function aligned = align_node_ecology_features(canonical_node_ids, node_features)
%ALIGN_NODE_ECOLOGY_FEATURES Flatten canonical node features node-major.
% Zero IDs are padding and produce an all-zero seven-channel slot.

    ids = double(canonical_node_ids);
    if size(ids,2) ~= 10
        error('WLNMDirNegNodeEcology:CanonicalIDShape', ...
            'canonical_node_ids must have exactly K=10 columns.');
    end
    if size(node_features,2) ~= 7 || any(~isfinite(node_features(:)))
        error('WLNMDirNegNodeEcology:NodeFeatureShape', ...
            'node_features must be a finite N-by-7 matrix.');
    end
    if any(ids(:) < 0) || any(ids(:) ~= floor(ids(:))) || ...
            any(ids(:) > size(node_features,1))
        error('WLNMDirNegNodeEcology:CanonicalNodeIDs', ...
            'Canonical IDs must be zero padding or valid node indices.');
    end

    M = size(ids,1);
    feature_dim = size(node_features,2);
    aligned = zeros(M, 10 * feature_dim);
    for slot = 1:10
        columns = (slot-1) * feature_dim + (1:feature_dim);
        valid = ids(:,slot) > 0;
        if any(valid)
            aligned(valid,columns) = node_features(ids(valid,slot),:);
        end
    end

    if any(~isfinite(aligned(:)))
        error('WLNMDirNegNodeEcology:InvalidAlignedFeatures', ...
            'Aligned node features contain non-finite values.');
    end
end
