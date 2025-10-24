function [train, test, train_nodes, test_nodes] = DivideNet_negative(net, ratioTrain)
    % Usage: Divide the network into training and testing sets
    % --Input--
    % - net: adjacency matrix representing the network
    % - ratioTrain: proportion of edges to keep in the training set
    % --Output--
    % - train: adjacency matrix of training links (1: link, 0: otherwise)
    % - test: adjacency matrix of testing links (1: link, 0: otherwise)
    %
    %  *problem identified: duplicate test links due to non-triangular matrix
    %%
    
    % Convert adjacency matrix to upper triangular form (no self-loops)
    net = triu(net) - diag(diag(net));
    
    % Calculate the number of edges for the test set
    num_testlinks = ceil((1-ratioTrain) * nnz(net));
    
    % Identify all edges in the network and store them in linklist
    [xindex, yindex] = find(net);
    linklist = [xindex yindex];
    
    clear xindex yindex;  % Remove unnecessary variables
    
    % Initialize the test set as a sparse matrix of the same size as net
    test = sparse(size(net,1), size(net,2));
    
    % Randomly select edges to add to the test set until the desired count is reached
    while (nnz(test) < num_testlinks)
        if length(linklist) <= 2
            break;
        end
    
        % Randomly choose an edge from the link list
        index_link = ceil(rand(1) * length(linklist));
    
        % Identify the nodes connected by the selected edge
        uid1 = linklist(index_link, 1);
        uid2 = linklist(index_link, 2);
        net(uid1, uid2) = 0;  % Temporarily remove the edge from the network
    
        %% Check if nodes uid1 and uid2 remain reachable
        tempvector = net(uid1, :);  % Get nodes reachable from uid1 in one step
        sign = 0;  % Default: edge cannot be removed
    
        % Calculate reachability within two steps
        uid1TOuid2 = tempvector * net + tempvector;
        if uid1TOuid2(uid2) > 0
            sign = 1;  % Mark as reachable
        else
            % Check reachability for more than two steps until stable
            while (nnz(spones(uid1TOuid2) - tempvector) ~= 0)
                tempvector = spones(uid1TOuid2);
                uid1TOuid2 = tempvector * net + tempvector;
                if uid1TOuid2(uid2) > 0
                    sign = 1;  % Mark as reachable
                    break;
                end
            end
        end
    
        % Modified: Allow all selected edges in the test set, regardless of connectivity
        sign = 1;
    
        %% Add edge to the test set or restore it in the training network
        if sign == 1  % Edge can be deleted
            linklist(index_link, :) = [];  % Remove edge from linklist
            test(uid1, uid2) = 1;  % Mark as test edge
        else
            linklist(index_link, :) = [];
            net(uid1, uid2) = 1;  % Restore edge in the network
        end
    end
    
    % Generate the symmetric training and testing adjacency matrices
    train = net + net';
    test = test + test';

    % === NEW: extract node sets (do not change existing splitting logic) ===
    [ti, tj] = find(train);
    train_nodes = unique([ti; tj]);   % column vector of nodes appearing in TRAIN

    [si, sj] = find(test);
    test_nodes  = unique([si; sj]);   % column vector of nodes appearing in TEST
end
    