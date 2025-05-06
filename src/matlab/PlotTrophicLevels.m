% Load food web
load('data/foodwebs_mat/SF1M2_tax_mass.mat');

% Compute trophic levels
tl = computeTrophicLevels(net);

% Create directed graph (prey → predator)
G = digraph(net);

% Manually define node positions
Y = tl;  % Y-axis = trophic level
N = length(tl);
X = linspace(1, N, N);  % space nodes evenly across X-axis

% Plot the graph
figure;
p = plot(G, 'XData', X, 'YData', Y, ...
         'NodeLabel', taxonomy, ...
         'MarkerSize', 10, ...
         'ArrowSize', 15, ...
         'NodeColor', [0.3 0.6 0.9]);

title('SF1M2 Food Web (Manual Layout by Trophic Level)');
ylabel('Trophic Level');
axis tight;
grid on;

% --- Subfunction ---
function tl = computeTrophicLevels(adj)
    n = size(adj,1);
    tl = ones(n,1);  % basal species start at level 1
    for iter = 1:100
        new_tl = 1 + (adj' * tl) ./ max(sum(adj, 1)', 1);
        if norm(new_tl - tl) < 1e-6
            break;
        end
        tl = new_tl;
    end
end
