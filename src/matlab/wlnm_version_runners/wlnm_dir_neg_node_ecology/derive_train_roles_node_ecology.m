function roles = derive_train_roles_node_ecology(train)
%DERIVE_TRAIN_ROLES_NODE_ECOLOGY Derive roles from training edges only.
% Orientation: train(resource,consumer)=1.

    if size(train,1) ~= size(train,2)
        error('WLNMDirNegNodeEcology:AdjacencyShape', ...
            'The training adjacency matrix must be square.');
    end

    prey_count = full(sum(train,1))';
    consumer_count = full(sum(train,2));
    roles = repmat("isolate", size(train,1), 1);
    roles(prey_count == 0 & consumer_count > 0) = "resource";
    roles(prey_count > 0 & consumer_count == 0) = "consumer";
    roles(prey_count > 0 & consumer_count > 0) = "consumer-resource";
end
